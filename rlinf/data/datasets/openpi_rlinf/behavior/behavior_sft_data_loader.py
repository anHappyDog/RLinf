# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import dataclasses
import logging
import multiprocessing
import typing

import numpy as np
import torch
from openpi.transforms import DataTransformFn, compose

from rlinf.data.datasets.openpi_rlinf.behavior.behavior_sft_dataset import (
    BehaviorSftDataset,
)
from rlinf.data.storage.lerobot import (
    resolve_lerobot_repo_id,
)
from rlinf.models.embodiment.openpi_rlinf.pi0_model.model import Observation
from rlinf.models.embodiment.openpi_rlinf.transforms_pipeline import (
    build_openpi_transforms,
)

logger = logging.getLogger(__name__)

__all__ = [
    "BehaviorSftDataConfig",
    "BehaviorSftDataLoader",
    "build_behavior_sft_dataloader",
    "create_behavior_sft_data_loader",
]

# Camera views resolved by the BEHAVIOR pi05 transform.
_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

# Raw LeRobot frame keys the streaming dataset yields.
_LEROBOT_IMAGE_KEY = "observation.images.rgb.head"
_LEROBOT_LEFT_WRIST_KEY = "observation.images.rgb.left_wrist"
_LEROBOT_RIGHT_WRIST_KEY = "observation.images.rgb.right_wrist"
_LEROBOT_STATE_KEY = "observation.state"


@dataclasses.dataclass(frozen=True)
class _Repack(DataTransformFn):
    """Map raw LeRobot frame keys to the ``observation/*`` names openpi's
    ``BehaviorInputs`` expects (an ``openpi.transforms`` transform, so it composes
    directly in front of the shared pipeline).

    The two wrist views are stacked into a single ``observation/wrist_image``
    ``[2, C, H, W]`` array in ``(left, right)`` order — the layout openpi's
    ``BehaviorInputs`` consumes (it splits index ``0``/``1`` into the left/right
    wrist). Images stay ``(C, H, W)`` float; ``BehaviorInputs._parse_image`` does
    the channel reorder and the float→uint8 conversion.
    """

    def __call__(self, frame: dict) -> dict:
        left_wrist = np.asarray(frame[_LEROBOT_LEFT_WRIST_KEY])
        right_wrist = np.asarray(frame[_LEROBOT_RIGHT_WRIST_KEY])
        data: dict = {
            "observation/image": np.asarray(frame[_LEROBOT_IMAGE_KEY]),
            "observation/wrist_image": np.stack([left_wrist, right_wrist], axis=0),
            "observation/state": np.asarray(frame[_LEROBOT_STATE_KEY]),
        }

        actions = frame.get("action")
        if actions is not None:
            data["actions"] = np.asarray(actions)

        prompt = frame.get("prompt", frame.get("task"))
        if prompt is None:
            raise ValueError(
                "BEHAVIOR SFT frame is missing both 'prompt' and 'task'; the "
                "streaming dataset must set the per-frame task text."
            )
        if not isinstance(prompt, str):
            prompt = prompt.item() if hasattr(prompt, "item") else str(prompt)
        data["prompt"] = prompt
        return data


class _TransformedStreamingDataset(torch.utils.data.Dataset):
    """Apply the composed openpi input transform to each streamed frame.

    The transform (``compose([_Repack(), *input_transforms])``) is built once in
    the main process and picklable, so ``spawn`` workers receive it directly.
    ``__len__`` only drives torch's default index sampler so iteration proceeds;
    the streaming dataset ignores ``idx`` and partitions chunks internally.
    """

    def __init__(
        self,
        dataset: BehaviorSftDataset,
        transform,
        *,
        history_state_dim: int,
    ):
        self._dataset = dataset
        self._transform = transform
        self._history_state_dim = history_state_dim

    def __getitem__(self, idx):
        raw_item = self._dataset[idx]
        history_frames = raw_item.pop("_history_frames", None)
        history_frame_mask = raw_item.pop("_history_frame_mask", None)
        history_time_offsets = raw_item.pop("_history_time_offsets", None)
        history_contrastive_mask = raw_item.pop(
            "_history_contrastive_mask", None
        )
        current = self._transform(raw_item)
        if history_frames is None:
            return current

        prompt = raw_item.get("prompt", raw_item["task"])
        transformed_history = []
        for frame in history_frames[:-1]:
            if frame is None:
                transformed_history.append(None)
                continue
            history_item = {**frame, "prompt": prompt, "task": prompt}
            transformed_history.append(self._transform(history_item))
        transformed_history.append(current)

        current["image"] = {
            key: np.stack(
                [
                    np.zeros_like(np.asarray(current["image"][key]))
                    if frame is None
                    else np.asarray(frame["image"][key])
                    for frame in transformed_history
                ]
            )
            for key in _IMAGE_KEYS
        }
        current["image_mask"] = {
            key: np.asarray(history_frame_mask, dtype=np.bool_)
            & np.asarray(
                [
                    False
                    if frame is None
                    else bool(np.asarray(frame["image_mask"][key]).item())
                    for frame in transformed_history
                ],
                dtype=np.bool_,
            )
            for key in _IMAGE_KEYS
        }
        current["history_states"] = np.stack(
            [
                np.zeros(
                    self._history_state_dim,
                    dtype=np.asarray(current["state"]).dtype,
                )
                if frame is None
                else np.asarray(frame["state"])[..., : self._history_state_dim]
                for frame in transformed_history
            ]
        )
        current["history_frame_mask"] = np.asarray(history_frame_mask, dtype=np.bool_)
        current["history_time_offsets"] = np.asarray(
            history_time_offsets, dtype=np.float32
        )
        if history_contrastive_mask is not None:
            current["history_contrastive_mask"] = np.asarray(
                history_contrastive_mask, dtype=np.bool_
            )
        return current

    def __len__(self) -> int:
        return len(self._dataset.hf_dataset)


def _sft_collate(items) -> tuple[Observation, torch.Tensor]:
    """Collate per-sample transform dicts into ``(Observation, actions)``.

    Plain numpy/torch stacking (no JAX): each per-sample dict is the openpi
    model-input format (nested ``image`` / ``image_mask`` dicts, plus ``state`` /
    ``actions`` / ``tokenized_prompt`` / ``tokenized_prompt_mask``). State and
    actions are cast to float32 (the verified SFT boundary dtype);
    ``Observation.from_dict`` converts the ``uint8`` images to ``float32`` in
    ``[-1, 1]``. Returned actions have shape ``[batch, action_horizon, action_dim]``.
    """
    if not items:
        raise ValueError("Cannot collate an empty BEHAVIOR SFT batch.")

    def _stack(key, dtype=None):
        return torch.from_numpy(
            np.stack([np.asarray(item[key], dtype=dtype) for item in items])
        )

    images = {
        key: torch.from_numpy(
            np.stack([np.asarray(item["image"][key]) for item in items])
        )
        for key in _IMAGE_KEYS
    }
    image_masks = {
        key: torch.from_numpy(
            np.stack(
                [np.asarray(item["image_mask"][key], dtype=np.bool_) for item in items]
            )
        )
        for key in _IMAGE_KEYS
    }
    observation_dict = {
        "image": images,
        "image_mask": image_masks,
        "state": _stack("state", np.float32),
        "tokenized_prompt": _stack("tokenized_prompt", np.int64).long(),
        "tokenized_prompt_mask": _stack("tokenized_prompt_mask", np.bool_),
    }
    if "history_states" in items[0]:
        if not all("history_states" in item for item in items):
            raise ValueError("Short-memory fields must be present on every batch item.")
        observation_dict.update(
            history_states=_stack("history_states", np.float32),
            history_frame_mask=_stack("history_frame_mask", np.bool_),
            history_time_offsets=_stack("history_time_offsets", np.float32),
        )
        if "history_contrastive_mask" in items[0]:
            if not all("history_contrastive_mask" in item for item in items):
                raise ValueError(
                    "history_contrastive_mask must be present on every batch item."
                )
            observation_dict["history_contrastive_mask"] = _stack(
                "history_contrastive_mask", np.bool_
            )
    observation = Observation.from_dict(observation_dict)
    actions = _stack("actions", np.float32)
    return observation, actions


@dataclasses.dataclass(frozen=True)
class BehaviorSftDataConfig:
    """Resolved BEHAVIOR SFT data-pipeline metadata, exposed via
    :meth:`BehaviorSftDataLoader.data_config`."""

    repo_id: str
    action_dim: int
    action_horizon: int
    max_token_len: int


def _worker_init_fn(worker_id: int) -> None:
    """Per-worker init hook (placeholder for worker-local environment setup)."""
    del worker_id


def create_behavior_sft_data_loader(
    *,
    behavior_dataset_root: str,
    assets_dir: str,
    asset_id: str,
    model_path: str,
    config_name: str,
    repo_id: str,
    tasks: list[str],
    modalities: list[str],
    action_dim: int,
    action_horizon: int,
    max_token_len: int,
    batch_size: int,
    num_workers: int,
    fine_grained_level: int,
    tolerance_s: float,
    shuffle: bool,
    seed: int,
    skill_labels: dict[int, str] | None,
    use_skill: bool,
    prompt_source: str,
    primitive_prompt_probability: float,
    mixed_boundary_fallback_to_task: bool,
    history_length: int,
    history_frame_stride: int,
    history_state_dim: int,
    discrete_state_input: bool | None,
    enable_gap: bool,
    allow_left: int,
    allow_right: int,
    dist_rank: int,
    dist_world_size: int,
    data_kwargs: dict | None = None,
    memory_critical_enabled: bool = False,
    memory_critical_primitive_prompts: list[str] | None = None,
    memory_critical_gripper_indices: tuple[int, int] = (14, 22),
    memory_critical_close_threshold: float = 0.0,
    memory_noncritical_keep_probability: float = 1.0,
    memory_critical_require_full_history: bool = True,
) -> "BehaviorSftDataLoader":
    """Build the BEHAVIOR-1K SFT data loader yielding ``(Observation, actions)``.

    Args:
        behavior_dataset_root: Local root of the LeRobot BEHAVIOR dataset.
        assets_dir: Directory holding the norm-stats tree; the openpi
            ``Normalize`` stage reads ``{assets_dir}/{asset_id}/norm_stats.json``
            (the SFT base checkpoint bundles none).
        asset_id: Norm-stats sub-directory (e.g. ``behavior-1k/2025-challenge-demos``).
        model_path: New-format checkpoint dir passed to ``get_openpi_config`` to
            select the shared transform pipeline.
        config_name: openpi TrainConfig key (e.g. ``pi05_behavior``).
        repo_id: LeRobot dataset repo id (used for metadata bookkeeping).
        tasks: BEHAVIOR task names to include.
        modalities: Observation modalities to load (e.g. ``["rgb"]``).
        action_dim: Model action dimension (metadata).
        action_horizon: Number of future action steps per sample.
        max_token_len: Maximum tokenized-prompt length (metadata).
        batch_size: Per-rank batch size.
        num_workers: Number of ``DataLoader`` workers (``> 0`` uses ``spawn``).
        fine_grained_level: Orchestrator level for the prompt task text.
        tolerance_s: Frame-timestamp sync tolerance.
        shuffle: Whether the streaming dataset shuffles its chunk order.
        seed: Base seed for the streaming chunk partition.
        skill_labels: Optional per-skill labels enabling skill mode.
        use_skill: Train on per-frame SKILL text (window-resolved) instead of the
            main-task text; requires explicit ``skill_labels``.
        prompt_source: Prompt supervision source: full ``task``, configured
            ``skill``, annotation-derived canonical ``primitive``, or ``mixed``
            task/primitive supervision.
        primitive_prompt_probability: Fraction of boundary-safe mixed-mode
            samples that use the primitive prompt instead of the full task.
        mixed_boundary_fallback_to_task: In mixed mode, use the task prompt for
            cross-boundary chunks instead of dropping their transition actions.
        history_length: Number of oldest-to-current observations per sample.
        history_frame_stride: Dataset-frame spacing between history observations.
        history_state_dim: Number of normalized state dimensions projected as
            historical state tokens.
        discrete_state_input: Whether the prompt tokenizer also embeds the
            current state as text. Short-memory MEM uses ``False`` because all
            K states are provided as continuous tokens.
        enable_gap: Skill mode — absorb a true gap into both adjacent skills.
        allow_left: Skill mode — frames to extend a contiguous skill start left.
        allow_right: Skill mode — frames to extend a contiguous skill end right.
        dist_rank: This rank's id, threaded into the per-rank chunk partition.
        dist_world_size: Total ranks, threaded into the per-rank chunk partition.
        data_kwargs: Optional ``openpi_data`` overrides forwarded to the pipeline.
        memory_critical_enabled: Whether to bias the stream toward annotated
            gripper-event windows and mark them for paired history loss.
        memory_critical_primitive_prompts: Optional canonical primitive allowlist
            for critical windows.
        memory_critical_gripper_indices: Raw action columns used to detect closure.
        memory_critical_close_threshold: Values below this threshold are closed.
        memory_noncritical_keep_probability: Deterministic keep rate for control
            windows when memory-critical sampling is enabled.
        memory_critical_require_full_history: Only mark windows whose complete
            K-frame context is available.

    Returns:
        A loader whose iteration yields ``(Observation, actions)`` 2-tuples.
    """
    dataset = BehaviorSftDataset(
        repo_id=repo_id,
        root=behavior_dataset_root,
        tolerance_s=tolerance_s,
        tasks=tasks or None,
        modalities=modalities or ["rgb"],
        local_only=True,
        delta_timestamps={"action": [t / 30.0 for t in range(action_horizon)]},
        chunk_streaming_using_keyframe=True,
        shuffle=shuffle,
        seed=seed,
        fine_grained_level=fine_grained_level,
        skill_labels=skill_labels,
        use_skill=use_skill,
        prompt_source=prompt_source,
        primitive_prompt_probability=primitive_prompt_probability,
        mixed_boundary_fallback_to_task=mixed_boundary_fallback_to_task,
        history_length=history_length,
        history_frame_stride=history_frame_stride,
        memory_critical_enabled=memory_critical_enabled,
        memory_critical_primitive_prompts=memory_critical_primitive_prompts,
        memory_critical_gripper_indices=memory_critical_gripper_indices,
        memory_critical_close_threshold=memory_critical_close_threshold,
        memory_noncritical_keep_probability=memory_noncritical_keep_probability,
        memory_critical_require_full_history=memory_critical_require_full_history,
        enable_gap=enable_gap,
        allow_left=allow_left,
        allow_right=allow_right,
        dist_rank=dist_rank,
        dist_world_size=dist_world_size,
    )

    # The shared openpi input pipeline (BehaviorInputs -> Normalize -> ModelTransform),
    # keyed off the checkpoint / config_name, with norm stats from assets_dir/asset_id.
    # Only the transform content comes from openpi; the wrapper + collate are plain
    # numpy/torch. The composed transform is built in the main process and is
    # picklable, so spawn workers receive it directly (no lazy per-worker build).
    input_transforms, _ = build_openpi_transforms(
        model_path,
        config_name,
        data_kwargs=data_kwargs,
        norm_stats_dir=assets_dir,
        norm_stats_asset_id=asset_id,
        discrete_state_input=discrete_state_input,
    )
    source = _TransformedStreamingDataset(
        dataset,
        compose([_Repack(), *input_transforms]),
        history_state_dim=history_state_dim,
    )

    # The streaming dataset partitions chunks (memoryless) or complete episodes
    # (short memory) per (rank, worker), so a DistributedSampler is intentionally
    # omitted (see module docstring).
    mp_context = multiprocessing.get_context("spawn") if num_workers > 0 else None

    generator = torch.Generator()
    generator.manual_seed(seed)

    logger.info(
        "BEHAVIOR SFT data loader: batch_size=%d, num_workers=%d, action_horizon=%d, "
        "prompt_source=%s, primitive_prompt_probability=%.3f, "
        "mixed_boundary_fallback_to_task=%s, history=%dx%d, "
        "norm_stats=%s/%s",
        batch_size,
        num_workers,
        action_horizon,
        prompt_source,
        primitive_prompt_probability,
        mixed_boundary_fallback_to_task,
        history_length,
        history_frame_stride,
        assets_dir,
        asset_id,
    )

    torch_loader = torch.utils.data.DataLoader(
        typing.cast(torch.utils.data.Dataset, source),
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=None,
        num_workers=num_workers,
        multiprocessing_context=mp_context,
        persistent_workers=num_workers > 0,
        collate_fn=_sft_collate,
        worker_init_fn=_worker_init_fn,
        drop_last=True,
        generator=generator,
    )

    data_config = BehaviorSftDataConfig(
        repo_id=repo_id,
        action_dim=action_dim,
        action_horizon=action_horizon,
        max_token_len=max_token_len,
    )
    return BehaviorSftDataLoader(torch_loader, data_config)


class BehaviorSftDataLoader:
    """Infinite ``(Observation, actions)`` loop over the BEHAVIOR SFT dataset.

    Re-iterates the underlying ``torch`` ``DataLoader`` forever. Each batch is
    already collated into an :class:`Observation` plus an actions tensor of shape
    ``[batch, action_horizon, action_dim]`` by :func:`_sft_collate`.
    """

    def __init__(
        self,
        torch_loader: torch.utils.data.DataLoader,
        data_config: BehaviorSftDataConfig,
    ):
        self._torch_loader = torch_loader
        self._data_config = data_config

    def data_config(self) -> BehaviorSftDataConfig:
        """Return the resolved data-pipeline metadata."""
        return self._data_config

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        """Expose the underlying ``torch`` ``DataLoader``."""
        return self._torch_loader

    def __iter__(self):
        while True:
            yield from self._torch_loader

    def __len__(self) -> int:
        return len(self._torch_loader)


def build_behavior_sft_dataloader(
    cfg, world_size, rank, data_paths, eval_dataset=False
):
    """Build the BEHAVIOR SFT data loader for the SFT worker.

    The streaming dataset partitions chunks per ``(rank, worker)``; ``rank`` /
    ``world_size`` are captured here (in the main process) and threaded into the
    dataset so that SPAWNED DataLoader workers -- which cannot read
    ``torch.distributed`` -- still partition by the correct per-rank id (otherwise
    every rank replicates rank 0's chunks, collapsing the effective batch to one
    rank's micro-batch). Every parameter is read directly from YAML (no hidden
    defaults). Returns ``(loader, loader.data_config())``.
    """
    from omegaconf import OmegaConf

    data_path = resolve_lerobot_repo_id(data_paths)
    if data_path is None:
        raise ValueError("openpi_rlinf BEHAVIOR SFT requires data.train_data_paths.")

    model_cfg = cfg.actor.model
    data_cfg = cfg.data

    # Norm stats resolve STRICTLY from YAML (assets_dir / asset_id) for the openpi
    # Normalize stage — the same file the eval / RL paths would resolve.
    assets_dir = model_cfg.openpi.assets_dir
    asset_id = model_cfg.openpi.asset_id
    config_name = str(model_cfg.openpi.config_name)
    data_kwargs = OmegaConf.select(cfg.actor, "openpi_data", default=None)
    if data_kwargs is not None:
        data_kwargs = OmegaConf.to_container(data_kwargs, resolve=True)

    # `cfg.data` is the production source of truth for the BEHAVIOR task set and
    # prompt source. `use_skill` remains a backward-compatible alias while new
    # recipes select task / skill / primitive explicitly with `prompt_source`.
    use_skill = bool(OmegaConf.select(data_cfg, "use_skill", default=False))
    prompt_source = str(
        OmegaConf.select(
            data_cfg,
            "prompt_source",
            default="skill" if use_skill else "task",
        )
    )
    if prompt_source not in {"task", "skill", "primitive", "mixed"}:
        raise ValueError(
            "data.prompt_source must be task, skill, primitive, or mixed; "
            f"got {prompt_source!r}."
        )
    if use_skill and prompt_source != "skill":
        raise ValueError(
            "data.use_skill=true is only compatible with data.prompt_source=skill."
        )
    tasks = list(data_cfg.tasks)
    primitive_prompt_probability = float(
        OmegaConf.select(
            data_cfg,
            "primitive_prompt_probability",
            default=0.5,
        )
    )
    if not 0.0 <= primitive_prompt_probability <= 1.0:
        raise ValueError(
            "data.primitive_prompt_probability must be in [0, 1]; "
            f"got {primitive_prompt_probability}."
        )
    mixed_boundary_fallback_to_task = bool(
        OmegaConf.select(
            data_cfg,
            "mixed_boundary_fallback_to_task",
            default=False,
        )
    )
    history_length = int(OmegaConf.select(data_cfg, "history_length", default=1))
    history_frame_stride = int(
        OmegaConf.select(data_cfg, "history_frame_stride", default=1)
    )
    short_memory_enabled = bool(
        OmegaConf.select(model_cfg, "openpi.short_memory.enabled", default=False)
    )
    if history_length > 1 and not short_memory_enabled:
        raise ValueError(
            "data.history_length > 1 requires actor.model.openpi.short_memory.enabled."
        )
    if short_memory_enabled and history_length <= 1:
        raise ValueError(
            "Short-memory SFT requires data.history_length > 1 so temporal weights "
            "receive a training signal."
        )
    history_state_dim = int(
        OmegaConf.select(model_cfg, "openpi.short_memory.state_dim", default=23)
    )
    memory_critical_enabled = bool(
        OmegaConf.select(data_cfg, "memory_critical.enabled", default=False)
    ) and not eval_dataset
    memory_critical_primitive_prompts = list(
        OmegaConf.select(
            data_cfg, "memory_critical.primitive_prompts", default=[]
        )
    )
    memory_critical_gripper_indices = tuple(
        int(index)
        for index in OmegaConf.select(
            data_cfg, "memory_critical.gripper_indices", default=[14, 22]
        )
    )
    memory_critical_close_threshold = float(
        OmegaConf.select(
            data_cfg, "memory_critical.close_threshold", default=0.0
        )
    )
    memory_noncritical_keep_probability = float(
        OmegaConf.select(
            data_cfg,
            "memory_critical.noncritical_keep_probability",
            default=1.0,
        )
    )
    memory_critical_require_full_history = bool(
        OmegaConf.select(
            data_cfg,
            "memory_critical.require_full_history",
            default=True,
        )
    )
    if memory_critical_enabled and history_length <= 1:
        raise ValueError(
            "data.memory_critical.enabled requires short-memory SFT."
        )
    discrete_state_input = OmegaConf.select(
        model_cfg, "openpi.discrete_state_input", default=None
    )
    skill_labels, enable_gap, allow_left, allow_right = None, True, 0, 0
    if prompt_source == "skill":
        # The skill labels are the REFERENCE per-task subtask list from config (NOT
        # the dataset's collapsed orchestrators, which equal the full task text). The
        # task-0000 local-skill recipe is exactly one task with a configured subtask
        # list and the fixed window recipe below.
        if len(tasks) != 1:
            raise ValueError(
                "openpi_rlinf BEHAVIOR SFT use_skill:true supports exactly one task "
                f"(the task-0000 skill recipe); got data.tasks={tasks}."
            )
        subtask_labels = data_cfg.task_subtasks
        labels = subtask_labels.get(tasks[0]) if subtask_labels else None
        if not labels:
            raise ValueError(
                "openpi_rlinf BEHAVIOR SFT use_skill:true requires the reference "
                f"skill labels at data.task_subtasks.{tasks[0]}; none was configured."
            )
        skill_labels = {i: str(label) for i, label in enumerate(labels)}
        # Fixed reference skill-window recipe (pi05_b1k-task0000_sft_local_skill);
        # intentionally hardcoded so the reference recipe cannot drift via config.
        enable_gap, allow_left, allow_right = True, 100, 100

    loader = create_behavior_sft_data_loader(
        behavior_dataset_root=str(data_cfg.behavior_dataset_root),
        assets_dir=str(assets_dir),
        asset_id=asset_id,
        model_path=str(model_cfg.model_path),
        config_name=config_name,
        repo_id=str(data_cfg.repo_id),
        tasks=tasks,
        modalities=list(data_cfg.modalities),
        action_dim=int(model_cfg.openpi.model_action_dim),
        action_horizon=int(model_cfg.num_action_chunks),
        max_token_len=int(model_cfg.openpi.max_token_len),
        batch_size=int(cfg.actor.eval_batch_size)
        if eval_dataset
        else int(cfg.actor.micro_batch_size),
        num_workers=int(data_cfg.num_workers),
        fine_grained_level=int(data_cfg.fine_grained_level),
        tolerance_s=float(data_cfg.tolerance_s),
        shuffle=not eval_dataset,
        seed=int(cfg.actor.seed),
        skill_labels=skill_labels,
        use_skill=use_skill,
        prompt_source=prompt_source,
        primitive_prompt_probability=primitive_prompt_probability,
        mixed_boundary_fallback_to_task=mixed_boundary_fallback_to_task,
        history_length=history_length,
        history_frame_stride=history_frame_stride,
        history_state_dim=history_state_dim,
        discrete_state_input=(
            None if discrete_state_input is None else bool(discrete_state_input)
        ),
        enable_gap=enable_gap,
        allow_left=allow_left,
        allow_right=allow_right,
        dist_rank=rank,
        dist_world_size=world_size,
        data_kwargs=data_kwargs,
        memory_critical_enabled=memory_critical_enabled,
        memory_critical_primitive_prompts=memory_critical_primitive_prompts,
        memory_critical_gripper_indices=memory_critical_gripper_indices,
        memory_critical_close_threshold=memory_critical_close_threshold,
        memory_noncritical_keep_probability=memory_noncritical_keep_probability,
        memory_critical_require_full_history=memory_critical_require_full_history,
    )
    return loader, loader.data_config()
