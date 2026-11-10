# Copyright 2026 The RLinf Authors.
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

"""RLinf contracts around the unmodified upstream PyTorch model."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType


def load_exported_policy(
    train_config: Any,
    checkpoint_dir: str | Path,
    *,
    pytorch_device: str = "cpu",
    sample_kwargs: dict | None = None,
) -> Any:
    """Load an RLinf export without rounding its trained FP32 norm weights."""
    from openpi.policies.policy_config import create_trained_policy
    from safetensors.torch import load_model

    policy = create_trained_policy(
        train_config,
        checkpoint_dir,
        pytorch_device=pytorch_device,
        sample_kwargs=sample_kwargs,
    )
    # The upstream post-load BF16 cast rounds norm weights before returning them
    # to FP32. Restore values after the final parameter dtypes are established.
    load_model(
        policy._model,
        str(Path(checkpoint_dir) / "model.safetensors"),
        strict=True,
        device="cpu",
    )
    return policy


class OfficialOpenPiPolicy(nn.Module, BasePolicy):
    """Train and sample with an upstream core and its checkpoint transforms.

    Training batches contain an already transformed observation dictionary and
    normalized actions. Prediction accepts a list of raw upstream feature dicts
    and returns the selected profile's action units; robot conversion belongs to
    the environment adapter.
    """

    def __init__(self, policy: Any, *, freeze_vision: bool = False):
        super().__init__()
        self.core = policy._model
        self._input_transform = policy._input_transform
        self._output_transform = policy._output_transform
        self._sample_kwargs = dict(policy._sample_kwargs)
        # The shared embedding/lm_head weight must stay in the root FSDP unit.
        self._fsdp_wrap_embeddings = False
        heads = ["action_in_proj", "action_out_proj"]
        if self.core.pi05:
            heads += ["time_mlp_in", "time_mlp_out"]
        else:
            heads += ["state_proj", "action_time_mlp_in", "action_time_mlp_out"]
        for name in heads:
            getattr(self.core, name)._fsdp_wrap_name = "openpi_fp32_projection"
        if freeze_vision:
            self.core.paligemma_with_expert.paligemma.vision_tower.requires_grad_(False)

    def forward(
        self,
        forward_type: ForwardType = ForwardType.DEFAULT,
        *,
        data: dict | None = None,
        observation: dict | None = None,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        num_steps: int = 10,
        return_loss_statistics: bool = False,
        loss_statistic_slices: tuple[slice, ...] = (slice(None),),
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        from openpi.models.model import Observation

        if forward_type == ForwardType.SFT:
            losses = self.core(
                Observation.from_dict(data["observation"]),
                data["actions"].to(dtype=torch.float32),
                noise=noise,
                time=time,
            )
            if "action_loss_mask" not in data:
                loss = losses.mean()
                if not return_loss_statistics:
                    return loss
                mask = torch.ones_like(losses)
            else:
                mask = data["action_loss_mask"].to(
                    device=losses.device, dtype=losses.dtype
                )
            if mask.shape != losses.shape:
                raise ValueError(
                    f"Expected action mask shape {losses.shape}, found {mask.shape}"
                )
            per_sample = (losses * mask).sum(dim=(-2, -1)) / mask.sum(dim=(-2, -1))
            loss = per_sample.mean() if "action_loss_mask" in data else loss
            if not return_loss_statistics:
                return loss
            # Each component follows the objective's per-sample masked mean.
            # Statistics carry independent sums/counts; an absent target has count zero.
            statistics = []
            with torch.no_grad():
                detached = losses.detach().double()
                for component in loss_statistic_slices:
                    weights = mask[..., component].double()
                    counts = weights.sum(dim=(-2, -1))
                    means = (detached[..., component] * weights).sum(dim=(-2, -1))
                    means = means / counts.clamp_min(1)
                    statistics.append(
                        torch.stack((means, (counts > 0).double(), counts), dim=-1)
                    )
            return {"loss": loss, "loss_statistics": torch.stack(statistics, dim=1)}
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(observation, noise=noise, num_steps=num_steps)
        raise NotImplementedError(f"Official OpenPI does not implement {forward_type}")

    def default_forward(
        self,
        observation: dict,
        *,
        noise: torch.Tensor | None = None,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Sample normalized action chunks from a transformed observation batch."""
        from openpi.models.model import Observation

        return self.core.sample_actions(
            observation["state"].device,
            Observation.from_dict(observation),
            noise=noise,
            num_steps=num_steps,
        )

    @torch.no_grad()
    def predict_action_batch(
        self, env_obs: list[dict], *, noise: np.ndarray | None = None
    ) -> tuple[np.ndarray, dict]:
        """Apply the upstream transforms to canonical, unbatched feature dicts."""
        import jax

        device = next(self.core.parameters()).device
        inputs = [
            self._input_transform(jax.tree.map(lambda x: x, obs)) for obs in env_obs
        ]
        batch = jax.tree.map(
            lambda *values: torch.as_tensor(np.stack(values), device=device), *inputs
        )
        noise_tensor = None if noise is None else torch.as_tensor(noise, device=device)
        actions = self(
            observation=batch,
            noise=noise_tensor,
            num_steps=self._sample_kwargs.get("num_steps", 10),
        )
        outputs = []
        for index in range(len(inputs)):
            output_features = {
                "state": batch["state"][index].detach().cpu().numpy(),
                "actions": actions[index].detach().cpu().numpy(),
            }
            if "source_schema" in env_obs[index]:
                output_features["source_schema"] = env_obs[index]["source_schema"]
            output = self._output_transform(output_features)
            outputs.append(output["actions"])
        return np.stack(outputs), {}

    def gradient_checkpointing_enable(self) -> None:
        """Delegate activation checkpointing to the upstream model."""
        self.core.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        """Disable upstream activation checkpointing."""
        self.core.gradient_checkpointing_disable()
