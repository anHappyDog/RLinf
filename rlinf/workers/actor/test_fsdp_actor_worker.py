# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import numpy as np
import torch
from omegaconf.omegaconf import DictConfig

from rlinf.algorithms.registry import policy_loss
from rlinf.config import SupportedModel
from rlinf.utils.data_iter_utils import get_iterator_k_split
from rlinf.utils.distributed import all_reduce_dict, masked_normalization
from rlinf.utils.metric_utils import (
    append_to_dict,
)
from rlinf.utils.nested_dict_process import (
    put_tensor_device,
)
from rlinf.utils.utils import (
    clear_memory,
    masked_mean,
    reshape_entropy,
)
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


def process_nested_dict_for_train(nested_dict: dict, shuffle_id: int) -> dict:
    ret_dict = {}
    for key, value in nested_dict.items():
        if key in ["dones", "terminations", "truncations", "prev_values"]:
            value = value[:-1]
        if "env_info" in key:
            raise NotImplementedError
        if value is None:
            ret_dict[key] = None
        if isinstance(value, torch.Tensor):
            if shuffle_id is not None:
                ret_dict[key] = value.reshape(-1, *value.shape[2:])[shuffle_id]
            else:
                ret_dict[key] = value.reshape(-1, *value.shape[2:])
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_train(value, shuffle_id)
    return ret_dict


class TestEmbodiedFSDPActorWorker(EmbodiedFSDPActor):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.version = 0

    def set_version(self, version: int) -> None:
        self.version = version

    @torch.no_grad()
    def compute_proximal_logprobs(self) -> None:
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)

        # # flatten here
        rollout_size = (self.rollout_batch["dones"].shape[0] - 1) * self.rollout_batch[
            "dones"
        ].shape[1]
        g = torch.Generator()
        g.manual_seed(self.cfg.actor.seed + self._rank)
        shuffle_id = torch.randperm(rollout_size, generator=g)

        self.rollout_batch = process_nested_dict_for_train(
            self.rollout_batch, shuffle_id
        )

        self.model.eval()

        total_size = self.rollout_batch["dones"].shape[0]
        micro_batch_size = self.cfg.actor.micro_batch_size

        num_splits = (total_size + micro_batch_size - 1) // micro_batch_size

        iterator = get_iterator_k_split(
            self.rollout_batch, num_splits, enforce_divisible_batch=False
        )

        all_proximal_logprobs = []

        for micro_batch in iterator:
            if SupportedModel(self.cfg.actor.model.model_type) in [
                SupportedModel.OPENVLA,
                SupportedModel.OPENVLA_OFT,
            ]:
                micro_batch["temperature"] = (
                    self.cfg.algorithm.sampling_params.temperature_train
                )
                micro_batch["top_k"] = self.cfg.algorithm.sampling_params.top_k

            device_micro_batch = put_tensor_device(micro_batch, self.device)
            output_dict = self.model(
                data=device_micro_batch,
                compute_logprobs=True,
                compute_entropy=False,
                compute_values=False,
                use_cache=True,
            )
            all_proximal_logprobs.append(output_dict["logprobs"].cpu())

        # Store result directly into the flattened batch
        self.rollout_batch["proximal_logprobs"] = torch.cat(all_proximal_logprobs)

    def run_training(self) -> None:
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)

        rollout_size = (self.rollout_batch["dones"].shape[0] - 1) * self.rollout_batch[
            "dones"
        ].shape[1]
        g = torch.Generator()
        g.manual_seed(self.cfg.actor.seed + self._rank)
        shuffle_id = torch.randperm(rollout_size, generator=g)

        # for key, value in self.rollout_batch.items():
        #     if isinstance(value, torch.Tensor):
        #         print(f"in self.rollout_batch, {key} shape; {value.shape}",flush=True)

        self.rollout_batch = process_nested_dict_for_train(
            self.rollout_batch, shuffle_id
        )
        if self.cfg.algorithm.normalize_advantages:
            self.rollout_batch["advantages"] = masked_normalization(
                self.rollout_batch["advantages"],
                self.rollout_batch.get("loss_mask", None),
            )

        self.model.train()

        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        rollout_size = self.rollout_batch["prev_logprobs"].size(0)
        # print(f"rollout_size:{rollout_size}",flush=True)
        batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        assert rollout_size % batch_size_per_rank == 0, (
            f"{rollout_size} is not divisible by {batch_size_per_rank}"
        )
        metrics = {}
        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for _ in range(update_epoch):
            rollout_dataloader_iter = get_iterator_k_split(
                self.rollout_batch,
                rollout_size // batch_size_per_rank,
            )
            for idx, train_global_batch in enumerate(rollout_dataloader_iter):
                # split batch into micro_batches
                train_global_batch_size = train_global_batch["prev_logprobs"].shape[0]
                assert (
                    train_global_batch_size
                    == self.cfg.actor.global_batch_size
                    // torch.distributed.get_world_size()
                )
                assert train_global_batch_size % self.cfg.actor.micro_batch_size == 0, (
                    f"{train_global_batch_size=}, {self.cfg.actor.micro_batch_size}"
                )

                train_micro_batches = list(
                    get_iterator_k_split(
                        train_global_batch,
                        train_global_batch_size // self.cfg.actor.micro_batch_size,
                    )
                )

                self.optimizer.zero_grad()
                for idx, data in enumerate(train_micro_batches):
                    data = put_tensor_device(
                        data, f"cuda:{int(os.environ['LOCAL_RANK'])}"
                    )
                    backward_ctx = self.before_micro_batch(
                        self.model,
                        is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
                    )
                    advantages = data["advantages"]
                    prev_logprobs = data["prev_logprobs"]
                    returns = data.get("returns", None)
                    prev_values = data.get("prev_values", None)
                    loss_mask = data.get("loss_mask", None)
                    loss_mask_sum = data.get("loss_mask_sum", None)
                    versions = data.get("versions", None)
                    if SupportedModel(self.cfg.actor.model.model_type) in [
                        SupportedModel.OPENVLA,
                        SupportedModel.OPENVLA_OFT,
                    ]:
                        data["temperature"] = (
                            self.cfg.algorithm.sampling_params.temperature_train
                        )
                        data["top_k"] = self.cfg.algorithm.sampling_params.top_k

                    compute_values = (
                        True if self.cfg.algorithm.adv_type == "gae" else False
                    )

                    with self.amp_context:
                        output_dict = self.model(
                            data=data,
                            compute_logprobs=True,
                            compute_entropy=self.cfg.algorithm.entropy_bonus > 0,
                            compute_values=compute_values,
                            use_cache=False,
                        )

                    if SupportedModel(self.cfg.actor.model.model_type) in [
                        SupportedModel.GR00T
                    ]:
                        prev_logprobs = output_dict["prev_logprobs"]

                    current_version = self.version + 1
                    proximal_logprobs = data.get("proximal_logprobs", None)

                    kwargs = {
                        "loss_type": self.cfg.algorithm.loss_type,
                        "logprob_type": self.cfg.algorithm.logprob_type,
                        "reward_type": self.cfg.algorithm.reward_type,
                        "single_action_dim": self.cfg.actor.model.get("action_dim", 7),
                        "proximal_logprobs": proximal_logprobs,
                        "logprobs": output_dict["logprobs"],
                        "values": output_dict.get("values", None),
                        "old_logprobs": prev_logprobs,
                        "advantages": advantages,
                        "returns": returns,
                        "prev_values": prev_values,
                        "versions": versions,
                        "current_version": current_version,
                        "clip_ratio_c": self.cfg.algorithm.get("clip_ratio_c", 3.0),
                        "clip_ratio_high": self.cfg.algorithm.clip_ratio_high,
                        "clip_ratio_low": self.cfg.algorithm.clip_ratio_low,
                        "value_clip": self.cfg.algorithm.get("value_clip", None),
                        "huber_delta": self.cfg.algorithm.get("huber_delta", None),
                        "loss_mask": loss_mask,
                        "loss_mask_sum": loss_mask_sum,
                        "max_episode_steps": self.cfg.env.train.max_episode_steps,
                        "task_type": self.cfg.runner.task_type,
                        "critic_warmup": self.optimizer_steps
                        < self.critic_warmup_steps,
                        "behave_weight_threshold": self.cfg.algorithm.get(
                            "behave_weight_threshold", None
                        ),
                    }
                    loss, metrics_data = policy_loss(**kwargs)

                    entropy_loss = torch.tensor(0.0, device=torch.cuda.current_device())
                    if (
                        self.cfg.algorithm.entropy_bonus > 0
                        and not kwargs["critic_warmup"]
                    ):
                        entropy = output_dict["entropy"]
                        entropy = reshape_entropy(
                            entropy,
                            entropy_type=self.cfg.algorithm.entropy_type,
                            action_dim=self.cfg.actor.model.get("action_dim", 7),
                            batch_size=output_dict["logprobs"].shape[0],
                        )
                        entropy_loss = masked_mean(entropy, mask=loss_mask)
                        loss -= self.cfg.algorithm.entropy_bonus * entropy_loss
                    metrics_data["entropy_loss"] = entropy_loss.detach().item()

                    loss /= self.gradient_accumulation
                    with backward_ctx:
                        self.grad_scaler.scale(loss).backward()

                    metrics_data["loss"] = loss.detach().item()
                    append_to_dict(metrics, metrics_data)
                torch.cuda.empty_cache()

                grad_norm, lr_list = self.optimizer_step()
                data = {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                }
                if len(lr_list) > 1:
                    data["critic/lr"] = lr_list[1]
                append_to_dict(metrics, data)
        # put LR scheduler step here
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        clear_memory()
        mean_metric_dict = {key: np.mean(value) for key, value in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )

        return mean_metric_dict
