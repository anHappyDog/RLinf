#!/usr/bin/env python3
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

"""Train and evaluate the R1 high-level text tail on held-out B1K episodes."""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import re
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Iterator, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from rlinf.data.datasets.openpi_rlinf.behavior.high_level import (
    PaligemmaSubtaskTokenizer,
    R1ManifestEntry,
    read_r1_manifest,
)
from rlinf.data.datasets.openpi_rlinf.behavior.high_level_state import (
    BehaviorStateReader,
    StateNormStats,
    compute_state_norm_stats,
    write_state_norm_stats,
)
from rlinf.models.embodiment.openpi_rlinf.pi0_model.model import Observation
from rlinf.models.embodiment.openpi_rlinf.pi0_model.pi0 import Pi0, TextLossOutput
from toolkits.mem.r0_overfit import (
    R0Sample,
    collate_samples,
    load_pi05_model,
    make_paligemma_tail_trainable,
)

_VIDEO_KEYS = {
    "base_0_rgb": "observation.images.rgb.head",
    "left_wrist_0_rgb": "observation.images.rgb.left_wrist",
    "right_wrist_0_rgb": "observation.images.rgb.right_wrist",
}
_BLANK_IMAGE = np.zeros((224, 224, 3), dtype=np.uint8)
InputMode = Literal["image", "image_state", "state"]
SamplingMode = Literal["proportional", "task_balanced"]


@dataclasses.dataclass(frozen=True)
class ParsedStep:
    """Canonical action components used for diagnostic generation metrics."""

    verb: str
    object_name: str | None
    destination: str | None


class TaskBalancedSampler(Sampler[int]):
    """Sample tasks uniformly and examples within each task uniformly."""

    def __init__(self, entries: list[R1ManifestEntry], *, seed: int):
        indices_by_task: dict[int, list[int]] = {}
        for index, entry in enumerate(entries):
            indices_by_task.setdefault(entry.task_index, []).append(index)
        if not indices_by_task:
            raise ValueError("Task-balanced sampling requires non-empty entries.")
        self._indices_by_task = indices_by_task
        self._seed = seed
        self._epoch = 0
        self._sample_count = len(entries)

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self._seed + self._epoch)
        self._epoch += 1
        task_indices = list(self._indices_by_task)
        output = []
        while len(output) < self._sample_count:
            rng.shuffle(task_indices)
            for task_index in task_indices:
                output.append(rng.choice(self._indices_by_task[task_index]))
                if len(output) == self._sample_count:
                    break
        return iter(output)

    def __len__(self) -> int:
        return self._sample_count


class R1SampleDataset(Dataset[R0Sample]):
    """Lazy exact-frame decoder for an R1 manifest split."""

    def __init__(
        self,
        dataset_root: Path,
        entries: list[R1ManifestEntry],
        tokenizer: PaligemmaSubtaskTokenizer,
        *,
        cache_size: int,
        input_mode: InputMode = "image",
        state_reader: BehaviorStateReader | None = None,
        state_stats: StateNormStats | None = None,
    ):
        if cache_size < 0:
            raise ValueError("cache_size must be non-negative.")
        self.dataset_root = dataset_root
        self.entries = entries
        self.tokenizer = tokenizer
        self.cache_size = cache_size
        self.input_mode = input_mode
        self.state_reader = state_reader
        self.state_stats = state_stats
        if input_mode in {"image_state", "state"} and (
            state_reader is None or state_stats is None
        ):
            raise ValueError(f"Input mode {input_mode!r} requires state data.")
        self._cache: OrderedDict[int, R0Sample] = OrderedDict()

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> R0Sample:
        if index in self._cache:
            sample = self._cache.pop(index)
            self._cache[index] = sample
            return sample

        entry = self.entries[index]
        if self.input_mode == "state":
            images = dict.fromkeys(_VIDEO_KEYS, _BLANK_IMAGE)
        else:
            images = {}
            for model_key, video_key in _VIDEO_KEYS.items():
                video_path = (
                    self.dataset_root
                    / "videos"
                    / f"task-{entry.task_index:04d}"
                    / video_key
                    / f"episode_{entry.episode_index:08d}.mp4"
                )
                images[model_key] = decode_video_frame(video_path, entry.frame_index)

        state = None
        if self.input_mode in {"image_state", "state"}:
            state_reader = self.state_reader
            state_stats = self.state_stats
            if state_reader is None or state_stats is None:
                raise RuntimeError("State input was not initialized.")
            raw_state = state_reader.read(entry.episode_index, entry.frame_index)
            state = state_stats.normalize(raw_state)
        training = self.tokenizer.tokenize(
            entry.task, state=state, subtask=entry.subtask
        )
        prefix = self.tokenizer.tokenize(entry.task, state=state)
        sample = R0Sample(
            manifest=entry,
            images=images,
            train_tokens=training.tokens,
            train_input_mask=training.input_mask,
            train_ar_mask=training.ar_mask,
            train_loss_mask=training.loss_mask,
            prefix_tokens=prefix.tokens,
            prefix_input_mask=prefix.input_mask,
            prefix_ar_mask=prefix.ar_mask,
        )
        if self.cache_size:
            self._cache[index] = sample
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return sample


def decode_video_frame(video_path: Path, frame_index: int) -> np.ndarray:
    """Seek to and decode one exact zero-based MP4 frame."""
    import av

    if frame_index < 0:
        raise ValueError("frame_index must be non-negative.")
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        time_base = float(stream.time_base)
        start_time = stream.start_time or 0
        target_pts = start_time + int(frame_index / fps / time_base)
        # Seeking exactly to the frame before a keyframe can round forward to
        # that keyframe in FFmpeg. Back off one second, then decode by PTS to the
        # exact requested frame.
        seek_margin_pts = round(1 / time_base)
        container.seek(
            max(start_time, target_pts - seek_margin_pts),
            stream=stream,
            backward=True,
            any_frame=False,
        )
        for frame in container.decode(stream):
            decoded_index = round((frame.pts - start_time) * time_base * fps)
            if decoded_index == frame_index:
                return frame.to_ndarray(format="rgb24")
            if decoded_index > frame_index:
                break
    raise ValueError(f"Could not decode frame {frame_index} from {video_path}.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--task-index",
        type=int,
        action="append",
        dest="task_indices",
        help="Restrict the pilot to one or more task indices.",
    )
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-token-len", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--input-mode",
        choices=("image", "image_state", "state"),
        default="image_state",
        help="Use RGB, RGB plus 23D proprio tokens, or proprio tokens only.",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=("proportional", "task_balanced"),
        default="task_balanced",
        help="Training example distribution across selected B1K tasks.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-last-layers", type=int, default=2)
    parser.add_argument("--cache-size", type=int, default=512)
    parser.add_argument(
        "--state-episode-cache-size",
        type=int,
        default=256,
        help="Number of decoded parquet episode-state arrays retained per process.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-initial-eval",
        action="store_true",
        help="Skip redundant base-model evaluation in repeated-seed runs.",
    )
    parser.add_argument(
        "--skip-train-eval",
        action="store_true",
        help="Evaluate validation only when full-train loss is not required.",
    )
    return parser.parse_args()


def main() -> None:
    """Train the language tail and evaluate on disjoint validation episodes."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("R1 training requires a CUDA GPU.")
    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("steps and batch_size must be positive.")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    task_indices = set(args.task_indices) if args.task_indices else None
    sampling_mode: SamplingMode = args.sampling_mode
    train_entries = _select_entries(
        read_r1_manifest(args.manifest_dir / "train.jsonl"),
        task_indices=task_indices,
        max_samples=args.max_train_samples,
        seed=args.seed,
        sampling_mode=sampling_mode,
    )
    val_entries = _select_entries(
        read_r1_manifest(args.manifest_dir / "val.jsonl"),
        task_indices=task_indices,
        max_samples=args.max_val_samples,
        seed=args.seed + 1,
        sampling_mode=sampling_mode,
    )
    if not train_entries or not val_entries:
        raise ValueError("Selected train and validation manifests must be non-empty.")
    train_episodes = {entry.episode_index for entry in train_entries}
    val_episodes = {entry.episode_index for entry in val_entries}
    if train_episodes.intersection(val_episodes):
        raise ValueError("Train and validation entries contain overlapping episodes.")
    _write_selected_manifest(args.output_dir / "train_selection.jsonl", train_entries)
    _write_selected_manifest(args.output_dir / "val_selection.jsonl", val_entries)

    input_mode: InputMode = args.input_mode
    state_reader = None
    state_stats = None
    if input_mode in {"image_state", "state"}:
        state_reader = BehaviorStateReader(
            args.dataset_root,
            episode_cache_size=args.state_episode_cache_size,
        )
        state_stats = compute_state_norm_stats(state_reader, train_entries)
        write_state_norm_stats(
            state_stats, args.output_dir / "state_norm_stats.json"
        )

    tokenizer = PaligemmaSubtaskTokenizer(max_len=args.max_token_len)
    train_dataset = R1SampleDataset(
        args.dataset_root,
        train_entries,
        tokenizer,
        cache_size=args.cache_size,
        input_mode=input_mode,
        state_reader=state_reader,
        state_stats=state_stats,
    )
    val_dataset = R1SampleDataset(
        args.dataset_root,
        val_entries,
        tokenizer,
        cache_size=args.cache_size,
        input_mode=input_mode,
        state_reader=state_reader,
        state_stats=state_stats,
    )
    train_loader = _make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
        sampling_mode=sampling_mode,
    )
    train_eval_loader = _make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
        sampling_mode="proportional",
    )
    val_loader = _make_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.seed,
        sampling_mode="proportional",
    )

    device = torch.device("cuda")
    model = load_pi05_model(args.model_path, device, args.max_token_len)
    trainable_names = make_paligemma_tail_trainable(model, args.train_last_layers)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.0,
    )

    if args.skip_initial_eval:
        initial_train = None
        initial_val = None
        initial_predictions = []
        initial_metrics = None
        print("initial evaluation skipped")
    else:
        initial_train = (
            None
            if args.skip_train_eval
            else evaluate_loss(model, train_eval_loader, device, input_mode)
        )
        initial_val = evaluate_loss(model, val_loader, device, input_mode)
        initial_predictions = evaluate_generation(
            model,
            val_loader,
            tokenizer,
            args.max_new_tokens,
            device,
            input_mode,
        )
        initial_generation_metrics = _generation_metrics(initial_predictions)
        initial_metrics = {
            "train": initial_train,
            "val": {
                **initial_val,
                **initial_generation_metrics,
            },
        }
        train_loss_text = (
            f"train_loss={initial_train['loss']:.6f} "
            if initial_train is not None
            else ""
        )
        print(
            f"initial {train_loss_text}val_loss={initial_val['loss']:.6f} "
            f"val_exact_match={initial_generation_metrics['exact_match']:.4f}"
        )

    model.train()
    train_iterator = iter(train_loader)
    for step in range(1, args.steps + 1):
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        observation = collate_r1_samples(
            batch, training=True, device=device, input_mode=input_mode
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output: TextLossOutput = model.compute_text_loss(observation, train=False)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            max_norm=1.0,
        )
        optimizer.step()
        if step == 1 or step % args.log_interval == 0 or step == args.steps:
            print(
                f"step={step:04d} loss={output.loss.detach().item():.6f} "
                f"token_accuracy={output.token_accuracy.detach().item():.4f}"
            )

    trainable_state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if name in trainable_names
    }
    torch.save(trainable_state, args.output_dir / "trainable_tail.pt")

    final_train = (
        None
        if args.skip_train_eval
        else evaluate_loss(model, train_eval_loader, device, input_mode)
    )
    final_val = evaluate_loss(model, val_loader, device, input_mode)
    final_predictions = evaluate_generation(
        model,
        val_loader,
        tokenizer,
        args.max_new_tokens,
        device,
        input_mode,
    )
    final_generation_metrics = _generation_metrics(final_predictions)
    metrics = {
        "train_sample_count": len(train_dataset),
        "val_sample_count": len(val_dataset),
        "train_episode_count": len(train_episodes),
        "val_episode_count": len(val_episodes),
        "input_mode": input_mode,
        "sampling_mode": sampling_mode,
        "state_norm_sample_count": (
            state_stats.sample_count if state_stats is not None else None
        ),
        "initial": initial_metrics,
        "final": {
            "train": final_train,
            "val": {
                **final_val,
                **final_generation_metrics,
            },
        },
        "steps": args.steps,
        "train_last_layers": args.train_last_layers,
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    _write_predictions(
        args.output_dir / "predictions.jsonl",
        initial_predictions,
        final_predictions,
    )
    print(json.dumps(metrics, indent=2))


def _select_entries(
    entries: list[R1ManifestEntry],
    *,
    task_indices: set[int] | None,
    max_samples: int | None,
    seed: int,
    sampling_mode: SamplingMode,
) -> list[R1ManifestEntry]:
    if task_indices is not None:
        entries = [entry for entry in entries if entry.task_index in task_indices]
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("Sample limits must be positive.")
        sample_count = min(max_samples, len(entries))
        if sampling_mode == "task_balanced":
            entries = _balanced_subset(entries, sample_count, seed)
        else:
            entries = random.Random(seed).sample(entries, k=sample_count)
        entries.sort(
            key=lambda item: (
                item.task_index,
                item.episode_index,
                item.primitive_index,
                item.frame_index,
            )
        )
    return entries


def _balanced_subset(
    entries: list[R1ManifestEntry], sample_count: int, seed: int
) -> list[R1ManifestEntry]:
    entries_by_task: dict[int, list[R1ManifestEntry]] = {}
    for entry in entries:
        entries_by_task.setdefault(entry.task_index, []).append(entry)
    rng = random.Random(seed)
    for task_entries in entries_by_task.values():
        rng.shuffle(task_entries)

    selected = []
    active_tasks = sorted(entries_by_task)
    while len(selected) < sample_count and active_tasks:
        rng.shuffle(active_tasks)
        next_active_tasks = []
        for task_index in active_tasks:
            task_entries = entries_by_task[task_index]
            if task_entries:
                selected.append(task_entries.pop())
                if len(selected) == sample_count:
                    break
            if task_entries:
                next_active_tasks.append(task_index)
        else:
            active_tasks = next_active_tasks
            continue
        break
    return selected


def _make_loader(
    dataset: R1SampleDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    sampling_mode: SamplingMode,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    sampler = (
        TaskBalancedSampler(dataset.entries, seed=seed)
        if shuffle and sampling_mode == "task_balanced"
        else None
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=_list_collate,
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def _list_collate(samples: list[R0Sample]) -> list[R0Sample]:
    return samples


def collate_r1_samples(
    samples: list[R0Sample],
    *,
    training: bool,
    device: torch.device,
    input_mode: InputMode,
) -> Observation:
    """Collate R1 samples and mask images for the state-only ablation."""
    observation = collate_samples(samples, training=training, device=device)
    if input_mode != "state":
        return observation
    return dataclasses.replace(
        observation,
        image_masks={
            key: torch.zeros_like(mask) for key, mask in observation.image_masks.items()
        },
    )


@torch.no_grad()
def evaluate_loss(
    model: Pi0,
    loader: DataLoader,
    device: torch.device,
    input_mode: InputMode,
) -> dict[str, float]:
    """Evaluate token-weighted masked CE over one R1 split."""
    model.eval()
    loss_sum = 0.0
    correct_sum = 0.0
    token_count = 0
    for batch in loader:
        observation = collate_r1_samples(
            batch, training=True, device=device, input_mode=input_mode
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output: TextLossOutput = model.compute_text_loss(observation, train=False)
        count = int(output.token_count.item())
        loss_sum += output.loss.item() * count
        correct_sum += output.token_accuracy.item() * count
        token_count += count
    return {
        "loss": loss_sum / token_count,
        "token_accuracy": correct_sum / token_count,
    }


@torch.no_grad()
def evaluate_generation(
    model: Pi0,
    loader: DataLoader,
    tokenizer: PaligemmaSubtaskTokenizer,
    max_new_tokens: int,
    device: torch.device,
    input_mode: InputMode,
) -> list[dict[str, object]]:
    """Greedily generate subtask text for a validation loader."""
    model.eval()
    predictions = []
    for batch in loader:
        observation = collate_r1_samples(
            batch, training=False, device=device, input_mode=input_mode
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            tokens, token_mask = model.generate_text(
                observation,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
            )
        for sample, sample_tokens, sample_mask in zip(
            batch, tokens.cpu(), token_mask.cpu(), strict=True
        ):
            generated_ids = sample_tokens[sample_mask].tolist()
            prediction = tokenizer.decode(generated_ids)
            target = sample.manifest.subtask
            predictions.append(
                {
                    "task_index": sample.manifest.task_index,
                    "episode_index": sample.manifest.episode_index,
                    "frame_index": sample.manifest.frame_index,
                    "primitive_index": sample.manifest.primitive_index,
                    "target": target,
                    "prediction": prediction,
                    "exact_match": _normalize_text(prediction)
                    == _normalize_text(target),
                    "ended_with_eos": bool(generated_ids)
                    and generated_ids[-1] == tokenizer.eos_token_id,
                }
            )
    return predictions


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _generation_metrics(
    predictions: list[dict[str, object]],
) -> dict[str, object]:
    if not predictions:
        raise ValueError("Generation metrics require at least one prediction.")
    target_counts: dict[str, int] = {}
    target_correct: dict[str, int] = {}
    task_counts: dict[int, int] = {}
    task_correct: dict[int, int] = {}
    for item in predictions:
        target = str(item["target"])
        target_counts[target] = target_counts.get(target, 0) + 1
        target_correct[target] = target_correct.get(target, 0) + int(
            bool(item["exact_match"])
        )
        task_index = int(item["task_index"])
        task_counts[task_index] = task_counts.get(task_index, 0) + 1
        task_correct[task_index] = task_correct.get(task_index, 0) + int(
            bool(item["exact_match"])
        )
    component_metrics = _component_metrics(predictions)
    per_task = {
        str(task_index): task_correct[task_index] / count
        for task_index, count in sorted(task_counts.items())
    }
    return {
        "exact_match": sum(bool(item["exact_match"]) for item in predictions)
        / len(predictions),
        "eos_rate": sum(bool(item["ended_with_eos"]) for item in predictions)
        / len(predictions),
        "per_target_exact_match": {
            target: target_correct[target] / count
            for target, count in sorted(target_counts.items())
        },
        "macro_task_exact_match": sum(per_task.values()) / len(per_task),
        "per_task_exact_match": per_task,
        **component_metrics,
    }


def _component_metrics(
    predictions: list[dict[str, object]],
) -> dict[str, float]:
    correct = {"verb": 0, "object": 0, "destination": 0}
    counts = {"verb": 0, "object": 0, "destination": 0}
    step_count_correct = 0
    for item in predictions:
        target_steps = _parse_subtask(str(item["target"]))
        predicted_steps = _parse_subtask(str(item["prediction"]))
        step_count_correct += len(target_steps) == len(predicted_steps)
        for index, target_step in enumerate(target_steps):
            predicted_step = (
                predicted_steps[index] if index < len(predicted_steps) else None
            )
            for component, value in (
                ("verb", target_step.verb),
                ("object", target_step.object_name),
                ("destination", target_step.destination),
            ):
                if value is None:
                    continue
                counts[component] += 1
                predicted_value = (
                    getattr(
                        predicted_step,
                        "object_name" if component == "object" else component,
                    )
                    if predicted_step is not None
                    else None
                )
                correct[component] += _normalize_text(predicted_value or "") == value
    return {
        "step_count_accuracy": step_count_correct / len(predictions),
        **{
            f"{component}_accuracy": (
                correct[component] / counts[component]
                if counts[component]
                else 0.0
            )
            for component in counts
        },
    }


def _parse_subtask(text: str) -> list[ParsedStep]:
    return [
        _parse_step(step)
        for step in re.split(r"\s+then\s+", _normalize_text(text))
        if step
    ]


def _parse_step(step: str) -> ParsedStep:
    patterns = (
        (r"pick up (.+) from (.+)", "pick up", 1, 2),
        (r"place (.+) (?:in|on|under|next to) (.+)", "place", 1, 2),
        (r"push (.+) to (.+)", "push", 1, 2),
        (r"turn (.+) toward (.+)", "turn", 1, 2),
        (r"pour (.+) into (.+)", "pour", 1, 2),
        (r"insert (.+) into (.+)", "insert", 1, 2),
        (r"attach (.+) to (.+)", "attach", 1, 2),
        (r"hang (.+) on (.+)", "hang", 1, 2),
        (r"spray (.+) with (.+)", "spray", 2, 1),
        (r"(?:sweep|wipe) (.+) with (.+)", None, 2, 1),
        (r"ignite (.+) with (.+)", "ignite", 2, 1),
    )
    for pattern, fixed_verb, object_group, destination_group in patterns:
        match = re.fullmatch(pattern, step)
        if match:
            verb = fixed_verb or step.split(maxsplit=1)[0]
            return ParsedStep(
                verb=verb,
                object_name=match.group(object_group),
                destination=match.group(destination_group),
            )

    if step.startswith("move to "):
        return ParsedStep("move to", None, step.removeprefix("move to "))
    verbs = (
        "pick up",
        "turn on",
        "turn off",
        "hand over",
        "tip over",
        "open",
        "close",
        "press",
        "hold",
        "release",
        "pour",
        "chop",
        "spray",
        "sweep",
        "wipe",
        "ignite",
        "pull",
        "push",
    )
    for verb in verbs:
        prefix = f"{verb} "
        if step.startswith(prefix):
            return ParsedStep(verb, step.removeprefix(prefix), None)
    verb, _, object_name = step.partition(" ")
    return ParsedStep(verb, object_name or None, None)


def _write_predictions(
    output_path: Path,
    initial: Iterable[dict[str, object]],
    final: Iterable[dict[str, object]],
) -> None:
    with output_path.open("w", encoding="utf-8") as output_file:
        for split, predictions in (("initial", initial), ("final", final)):
            for prediction in predictions:
                output_file.write(json.dumps({"split": split, **prediction}) + "\n")


def _write_selected_manifest(
    output_path: Path, entries: Iterable[R1ManifestEntry]
) -> None:
    with output_path.open("w", encoding="utf-8") as output_file:
        for entry in entries:
            output_file.write(json.dumps(dataclasses.asdict(entry)) + "\n")


if __name__ == "__main__":
    main()
