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


import torch
from omegaconf import DictConfig
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.tensor import DTensor

from rlinf.data.io_struct import RolloutResult
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.scheduler import Worker
from rlinf.scheduler.channel import Channel
from rlinf.utils.data_iter_utils import get_iterator_k_split
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.utils.utils import (
    compute_logprobs_from_logits,
    cpu_weight_swap,
    retrieve_model_state_dict_in_cpu,
)
from rlinf.workers.inference.utils import (
    _get_full_numel,
    _get_local_tensor,
    _shard_range_1d,
)


class FSDPInference(FSDPModelManager, Worker):
    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
    ):
        Worker.__init__(self)
        super().__init__(cfg.inference, self._world_size, self._rank)
        self.cfg = cfg
        self._actor_group_name = cfg.actor.group_name
        self._component_placement = placement
        # algorithms
        self.kl_beta = cfg.algorithm.get("kl_beta", 0)
        self.reinpp_kl_beta = cfg.algorithm.get("reinpp_kl_beta", 0)
        self.combine_reference_model = cfg.actor.get("combine_reference_model", True)

        self.response_len = (
            self.cfg.actor.model.encoder_seq_length - self.cfg.data.max_prompt_length
        )

        self.total_batch_size_per_dp = (
            self.cfg.data.rollout_batch_size
            * self.cfg.algorithm.group_size
            // self._world_size
        )

    def init_worker(self) -> None:
        # create and wrap model with FSDP's strategy
        model = self.model_provider_func()
        self.model = self._strategy.wrap_model(
            model=model, device_mesh=self._device_mesh
        )

        # Get Ref model if needed.
        ref_policy_state_dict = None
        if (
            self.kl_beta > 0 or self.reinpp_kl_beta > 0
        ) and self.combine_reference_model:
            ref_policy_state_dict = retrieve_model_state_dict_in_cpu(self.model[0])
        self.ref_policy_state_dict = ref_policy_state_dict
        self._setup_actor_weight_src_ranks()

    def _setup_actor_weight_src_ranks(self) -> None:
        self.actor_world_size = self._component_placement.get_world_size("actor")
        self.inference_world_size = self._component_placement.get_world_size(
            "inference"
        )

        assert (
            self.actor_world_size % self.inference_world_size == 0
            or self.inference_world_size % self.actor_world_size == 0
        ), "Actor and Inference world sizes must be multiples of each other."

        if self.actor_world_size >= self.inference_world_size:
            actor_ranks_per_inference = (
                self.actor_world_size // self.inference_world_size
            )
            self._actor_weight_src_ranks = [
                self._rank * actor_ranks_per_inference + i
                for i in range(actor_ranks_per_inference)
            ]
        else:
            inference_ranks_per_actor = (
                self.inference_world_size // self.actor_world_size
            )
            self._actor_weight_src_ranks = [self._rank // inference_ranks_per_actor]
            self.sync_model_idx = self._rank % inference_ranks_per_actor

    @torch.no_grad()
    def load_from_actor_by_intersection(
        self, cur_state_dict: dict[str, torch.Tensor | DTensor | ShardedTensor]
    ) -> None:
        """
        Synchronize the model weights from actor workers to the inference workers by computing the intersection
        of the sharded weights sent from actor workers and load them into the current rank's state_dict.

        Args:
            cur_state_dict(dict[str, torch.Tensor|DTensor|ShardedTensor]): The current rank's state_dict to be updated.
        """
        receiving_jobs = [
            self.recv(
                src_rank=rank, src_group_name=self._actor_group_name, async_op=True
            )
            for rank in self._actor_weight_src_ranks
        ]
        print("Waiting for receiving model weights from actors...")
        received_state_dicts: list[dict[str, torch.Tensor]] = [
            job.wait() for job in receiving_jobs
        ]

        for k, cur_tensor in cur_state_dict.items():
            dst_local = _get_local_tensor(cur_tensor)
            if dst_local is None:
                continue

            full_numel = _get_full_numel(cur_tensor)
            dst_start, dst_end, _ = _shard_range_1d(
                full_numel, self._world_size, self._rank
            )

            dst_flat = dst_local.view(-1)

            for src_dict, src_actor_rank in zip(
                received_state_dicts, self._actor_weight_src_ranks
            ):
                if k not in src_dict:
                    self.logger.warning(
                        f"FSDPInference sync model weight: Key {k} not found in actor rank {src_actor_rank}'s state_dict."
                    )
                    continue

                src_tensor = src_dict[k]
                src_flat = src_tensor.to(
                    device=dst_flat.device, dtype=dst_flat.dtype, non_blocking=True
                ).view(-1)

                src_start, src_end, src_shard_size = _shard_range_1d(
                    full_numel, self.actor_world_size, src_actor_rank
                )

                inter_start = max(dst_start, src_start)
                inter_end = min(dst_end, src_end)
                if inter_start >= inter_end:
                    self.logger.error(
                        f"FSDPInference sync model weight: No intersection for key {k} between actor rank {src_actor_rank} and inference rank {self._rank}."
                    )
                    continue

                n = inter_end - inter_start
                dst_off = inter_start - dst_start
                src_off = inter_start - src_start

                assert dst_off + n <= dst_flat.numel(), (
                    f"{k}: dst overflow in FSDPInference's sync_model_from_actor, dst_off={dst_off}, n={n}, dst_flat.numel()={dst_flat.numel()}"
                )
                assert src_off + n <= src_flat.numel(), (
                    f"{k}: src overflow in FSDPInference's sync_model_from_actor, src_off={src_off}, n={n}, src_flat.numel()={src_flat.numel()}"
                )

                dst_flat[dst_off : dst_off + n].copy_(src_flat[src_off : src_off + n])

    def sync_model_from_actor(self) -> None:
        opts = StateDictOptions(cpu_offload=False, full_state_dict=False)
        current_rank_state_dict = get_model_state_dict(model=self.model, options=opts)
        self.load_from_actor_by_intersection(cur_state_dict=current_rank_state_dict)
        set_model_state_dict(
            model=self.model, model_state_dict=current_rank_state_dict, options=opts
        )

    def get_batch(
        self, channel: Channel
    ) -> tuple[dict[str, torch.Tensor], RolloutResult]:
        result: RolloutResult = channel.get()

        batch = result.to_actor_batch(
            self.cfg.data.max_prompt_length,
            self.cfg.actor.model.encoder_seq_length,
            self.tokenizer.eos_token_id,
        )
        return batch, result

    @torch.no_grad()
    def inference_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.model.eval()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        position_ids = batch["position_ids"]

        multi_modal_inputs = {}
        if "multi_modal_inputs" in batch.keys():
            for key in batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in batch["multi_modal_inputs"]],
                    dim=0,
                ).cuda()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            **multi_modal_inputs,
        )

        logits = outputs.logits
        logits = logits[:, -self.response_len - 1 : -1, :]
        logits = logits / self.cfg.algorithm.sampling_params.temperature

        responses = input_ids[:, -self.response_len :]
        logprobs = compute_logprobs_from_logits(logits, responses)
        return logprobs

    def run_inference(
        self,
        input_channel: Channel,
        output_channel: Channel,
        compute_ref_logprobs: bool,
    ) -> None:
        """
        Compute prev/ref logprobs using the actor Model's forward.

        Args:
            input_channel: The input channel to read from.
            output_channel: The output channel to send results to.
            compute_ref_logprobs: Whether to compute reference logprobs.
        """
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            batch, rollout_result = self.get_batch(input_channel)
            recv_batch_size += rollout_result.num_sequence

            num_splits = (
                rollout_result.num_sequence
                // self.cfg.algorithm.logprob_forward_micro_batch_size
            )
            micro_batches_iter = get_iterator_k_split(
                batch,
                num_splits=num_splits,
            )
            micro_batches = list(micro_batches_iter)

            prev_logprobs = []
            with self.worker_timer():
                for micro_batch in micro_batches:
                    prev_logprobs.append(self.inference_step(micro_batch).cpu())

                if rollout_result.rollout_logprobs is not None:
                    # Rollout has returned logprobs, store the recomputed logprobs in recompute_prev_logprobs
                    rollout_result.recompute_prev_logprobs = torch.cat(prev_logprobs)
                else:
                    # Otherwise, directly store the logprobs in prev_logprobs (the final logprobs used for training)
                    rollout_result.prev_logprobs = torch.cat(prev_logprobs)

            if compute_ref_logprobs:
                assert self.ref_policy_state_dict is not None, (
                    "Reference policy state dict is None but compute_ref_logprobs is True"
                )
                ref_logprobs = []
                with cpu_weight_swap(self.model, self.ref_policy_state_dict):
                    for micro_batch in micro_batches:
                        ref_logprobs.append(self.inference_step(micro_batch).cpu())
                    rollout_result.ref_logprobs = torch.cat(ref_logprobs)

            output_channel.put(rollout_result)

        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )
