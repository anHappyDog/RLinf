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

"""Run the π0.5 high-level text R0 micro-overfit experiment."""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from rlinf.data.datasets.openpi_rlinf.behavior.high_level import (
    PaligemmaSubtaskTokenizer,
    R0ManifestEntry,
    build_r0_manifest,
    read_r0_manifest,
    write_r0_manifest,
)
from rlinf.models.embodiment.openpi_rlinf.pi0_model.model import Observation
from rlinf.models.embodiment.openpi_rlinf.pi0_model.pi0 import Pi0, TextLossOutput
from rlinf.models.embodiment.openpi_rlinf.pi0_model.pi0_config import Pi0Config
from rlinf.models.embodiment.openpi_rlinf.utils.rlt_utils import (
    load_base_safetensors,
    resolve_model_safetensors,
)

_VIDEO_KEYS = {
    "base_0_rgb": "observation.images.rgb.head",
    "left_wrist_0_rgb": "observation.images.rgb.left_wrist",
    "right_wrist_0_rgb": "observation.images.rgb.right_wrist",
}


@dataclasses.dataclass(frozen=True)
class R0Sample:
    """Decoded images plus training and generation tokens for one R0 item."""

    manifest: R0ManifestEntry
    images: dict[str, np.ndarray]
    train_tokens: np.ndarray
    train_input_mask: np.ndarray
    train_ar_mask: np.ndarray
    train_loss_mask: np.ndarray
    prefix_tokens: np.ndarray
    prefix_input_mask: np.ndarray
    prefix_ar_mask: np.ndarray


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help=(
            "Directory containing a π0.5 model.safetensors in either upstream "
            "OpenPI PyTorch or converted OpenPI_RLinf layout."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Existing R0 JSONL manifest. A task-0000 manifest is built if omitted.",
    )
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--episode-index", type=int)
    parser.add_argument("--samples-per-primitive", type=int, default=4)
    parser.add_argument("--max-token-len", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-last-layers", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fixed-target",
        help="Override every manifest target for the simpler R0-A plumbing test.",
    )
    return parser.parse_args()


def main() -> None:
    """Build the microset, overfit the text tail, and write R0 metrics."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("R0 overfit requires a CUDA GPU.")
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.log_interval <= 0:
        raise ValueError("--log-interval must be positive.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = _load_or_build_manifest(args)
    if args.fixed_target is not None:
        manifest_entries = [
            dataclasses.replace(entry, subtask=args.fixed_target)
            for entry in manifest_entries
        ]

    tokenizer = PaligemmaSubtaskTokenizer(max_len=args.max_token_len)
    samples = load_r0_samples(args.dataset_root, manifest_entries, tokenizer)
    device = torch.device("cuda")
    model = load_pi05_model(args.model_path, device, args.max_token_len)
    trainable_names = make_paligemma_tail_trainable(model, args.train_last_layers)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.0,
    )

    initial_loss = evaluate_loss(model, samples, args.batch_size, device)
    initial_predictions = evaluate_generation(
        model,
        samples,
        tokenizer,
        args.batch_size,
        args.max_new_tokens,
        device,
    )
    print(
        f"initial loss={initial_loss['loss']:.6f} "
        f"token_accuracy={initial_loss['token_accuracy']:.4f} "
        f"exact_match={_exact_match(initial_predictions):.4f} "
        f"eos_rate={_eos_rate(initial_predictions):.4f}"
    )

    model.train()
    rng = random.Random(args.seed)
    for step in range(1, args.steps + 1):
        batch_samples = rng.sample(samples, k=min(args.batch_size, len(samples)))
        observation = collate_samples(batch_samples, training=True, device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model.compute_text_loss(observation, train=False)
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

    final_loss = evaluate_loss(model, samples, args.batch_size, device)
    final_predictions = evaluate_generation(
        model,
        samples,
        tokenizer,
        args.batch_size,
        args.max_new_tokens,
        device,
    )
    shuffled_samples = shuffle_sample_images(samples)
    shuffled_predictions = evaluate_generation(
        model,
        shuffled_samples,
        tokenizer,
        args.batch_size,
        args.max_new_tokens,
        device,
    )

    initial_exact = _exact_match(initial_predictions)
    final_exact = _exact_match(final_predictions)
    shuffled_exact = _exact_match(shuffled_predictions)
    initial_eos_rate = _eos_rate(initial_predictions)
    final_eos_rate = _eos_rate(final_predictions)
    shuffled_eos_rate = _eos_rate(shuffled_predictions)
    loss_drop = 1.0 - final_loss["loss"] / initial_loss["loss"]
    visual_gate = args.fixed_target is not None or shuffled_exact <= 0.5
    r0_pass = (
        loss_drop >= 0.95
        and final_loss["loss"] < 0.1
        and final_loss["token_accuracy"] == 1.0
        and final_exact == 1.0
        and final_eos_rate == 1.0
        and visual_gate
    )
    metrics = {
        "initial": {
            **initial_loss,
            "exact_match": initial_exact,
            "eos_rate": initial_eos_rate,
        },
        "final": {
            **final_loss,
            "exact_match": final_exact,
            "eos_rate": final_eos_rate,
        },
        "shuffled_images": {
            "exact_match": shuffled_exact,
            "eos_rate": shuffled_eos_rate,
        },
        "loss_drop_fraction": loss_drop,
        "r0_pass": r0_pass,
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
        shuffled_predictions,
    )
    trainable_state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if name in trainable_names
    }
    torch.save(trainable_state, args.output_dir / "trainable_tail.pt")
    print(json.dumps(metrics, indent=2))


def _load_or_build_manifest(args: argparse.Namespace) -> list[R0ManifestEntry]:
    if args.manifest is not None:
        return read_r0_manifest(args.manifest)
    entries = build_r0_manifest(
        args.dataset_root,
        task_index=args.task_index,
        episode_index=args.episode_index,
        samples_per_primitive=args.samples_per_primitive,
    )
    write_r0_manifest(entries, args.output_dir / "r0_manifest.jsonl")
    return entries


def load_r0_samples(
    dataset_root: Path,
    manifest_entries: list[R0ManifestEntry],
    tokenizer: PaligemmaSubtaskTokenizer,
) -> list[R0Sample]:
    """Decode all fixed R0 images and build text training tensors."""
    if not manifest_entries:
        raise ValueError("R0 manifest is empty.")
    frames_by_episode: dict[int, set[int]] = defaultdict(set)
    for entry in manifest_entries:
        frames_by_episode[entry.episode_index].add(entry.frame_index)

    decoded_images: dict[tuple[int, str, int], np.ndarray] = {}
    for episode_index, frame_indices in frames_by_episode.items():
        task_index = episode_index // 10_000
        for model_key, video_key in _VIDEO_KEYS.items():
            video_path = (
                dataset_root
                / "videos"
                / f"task-{task_index:04d}"
                / video_key
                / f"episode_{episode_index:08d}.mp4"
            )
            frames = decode_video_frames(video_path, frame_indices)
            for frame_index, frame in frames.items():
                decoded_images[(episode_index, model_key, frame_index)] = frame

    samples = []
    for entry in manifest_entries:
        train_text = tokenizer.tokenize(entry.task, subtask=entry.subtask)
        prefix_text = tokenizer.tokenize(entry.task)
        images = {
            model_key: decoded_images[
                (entry.episode_index, model_key, entry.frame_index)
            ]
            for model_key in _VIDEO_KEYS
        }
        samples.append(
            R0Sample(
                manifest=entry,
                images=images,
                train_tokens=train_text.tokens,
                train_input_mask=train_text.input_mask,
                train_ar_mask=train_text.ar_mask,
                train_loss_mask=train_text.loss_mask,
                prefix_tokens=prefix_text.tokens,
                prefix_input_mask=prefix_text.input_mask,
                prefix_ar_mask=prefix_text.ar_mask,
            )
        )
    return samples


def decode_video_frames(
    video_path: Path, frame_indices: Iterable[int]
) -> dict[int, np.ndarray]:
    """Decode selected zero-based frames from one MP4 in a single pass."""
    import av

    if not video_path.exists():
        raise FileNotFoundError(video_path)
    requested = set(frame_indices)
    frames = {}
    with av.open(str(video_path)) as container:
        for frame_index, frame in enumerate(container.decode(video=0)):
            if frame_index in requested:
                frames[frame_index] = frame.to_ndarray(format="rgb24")
            if frame_index >= max(requested):
                break
    missing = requested.difference(frames)
    if missing:
        raise ValueError(
            f"Video {video_path} ended before frames {sorted(missing)} were decoded."
        )
    return frames


def load_pi05_model(model_path: Path, device: torch.device, max_token_len: int) -> Pi0:
    """Load an upstream or converted base π0.5 checkpoint as a bare Pi0 core."""
    weights_path = resolve_model_safetensors(model_path)
    if weights_path is None:
        raise FileNotFoundError(f"Expected model.safetensors under {model_path}.")
    config = Pi0Config(
        pi05=True,
        action_horizon=32,
        action_dim=32,
        max_token_len=max_token_len,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        dtype="bfloat16",
    )
    model = config.create()
    load_base_safetensors(model, weights_path)
    return model.to(device)


def make_paligemma_tail_trainable(model: Pi0, layer_count: int) -> set[str]:
    """Freeze the model and enable only expert-0 modules in the final LLM layers."""
    if layer_count <= 0 or layer_count > len(model.llm.layers):
        raise ValueError(
            f"layer_count must be in [1, {len(model.llm.layers)}], got {layer_count}."
        )
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    for layer in model.llm.layers[-layer_count:]:
        expert_modules = [
            layer.attn.q_proj[0],
            layer.attn.o_proj[0],
            layer.pre_attention_norms[0],
            layer.pre_ffw_norms[0],
            layer.mlps[0],
        ]
        if layer.attn.k_proj[0] is not None:
            expert_modules.extend([layer.attn.k_proj[0], layer.attn.v_proj[0]])
        for module in expert_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    for parameter in model.llm.final_norms[0].parameters():
        parameter.requires_grad_(True)

    return {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }


def collate_samples(
    samples: list[R0Sample], *, training: bool, device: torch.device
) -> Observation:
    """Collate decoded R0 samples into the local Pi0 Observation type."""
    if not samples:
        raise ValueError("Cannot collate an empty R0 batch.")

    images = {
        key: torch.from_numpy(np.stack([sample.images[key] for sample in samples]))
        for key in _VIDEO_KEYS
    }
    image_masks = {
        key: torch.ones(len(samples), dtype=torch.bool) for key in _VIDEO_KEYS
    }
    if training:
        tokens = np.stack([sample.train_tokens for sample in samples])
        input_mask = np.stack([sample.train_input_mask for sample in samples])
        ar_mask = np.stack([sample.train_ar_mask for sample in samples])
        loss_mask = np.stack([sample.train_loss_mask for sample in samples])
    else:
        tokens = np.stack([sample.prefix_tokens for sample in samples])
        input_mask = np.stack([sample.prefix_input_mask for sample in samples])
        ar_mask = np.stack([sample.prefix_ar_mask for sample in samples])
        loss_mask = None

    observation = Observation.from_dict(
        {
            "image": images,
            "image_mask": image_masks,
            # R0 intentionally omits state from the language prefix so that the
            # same task's primitive label must be distinguished from images.
            "state": torch.zeros(len(samples), 32, dtype=torch.float32),
            "tokenized_prompt": torch.from_numpy(tokens).long(),
            "tokenized_prompt_mask": torch.from_numpy(input_mask).bool(),
            "token_ar_mask": torch.from_numpy(ar_mask).bool(),
            "token_loss_mask": (
                torch.from_numpy(loss_mask).bool() if loss_mask is not None else None
            ),
        }
    )
    return _observation_to_device(observation, device)


def _observation_to_device(
    observation: Observation, device: torch.device
) -> Observation:
    return Observation(
        images={key: value.to(device) for key, value in observation.images.items()},
        image_masks={
            key: value.to(device) for key, value in observation.image_masks.items()
        },
        state=observation.state.to(device),
        tokenized_prompt=observation.tokenized_prompt.to(device),
        tokenized_prompt_mask=observation.tokenized_prompt_mask.to(device),
        token_ar_mask=observation.token_ar_mask.to(device),
        token_loss_mask=(
            observation.token_loss_mask.to(device)
            if observation.token_loss_mask is not None
            else None
        ),
    )


@torch.no_grad()
def evaluate_loss(
    model: Pi0,
    samples: list[R0Sample],
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate masked CE and token accuracy over the fixed microset."""
    model.eval()
    loss_sum = 0.0
    correct_sum = 0.0
    token_count = 0
    for batch in _batches(samples, batch_size):
        observation = collate_samples(batch, training=True, device=device)
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
    samples: list[R0Sample],
    tokenizer: PaligemmaSubtaskTokenizer,
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
) -> list[dict[str, object]]:
    """Greedily decode every microset sample and return exact-match records."""
    model.eval()
    predictions = []
    for batch in _batches(samples, batch_size):
        observation = collate_samples(batch, training=False, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            tokens, token_mask = model.generate_text(
                observation,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
            )
        tokens = tokens.cpu()
        token_mask = token_mask.cpu()
        for sample, sample_tokens, sample_mask in zip(
            batch, tokens, token_mask, strict=True
        ):
            generated_ids = sample_tokens[sample_mask].tolist()
            prediction = tokenizer.decode(generated_ids)
            target = sample.manifest.subtask
            predictions.append(
                {
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


def shuffle_sample_images(samples: list[R0Sample]) -> list[R0Sample]:
    """Rotate images across primitive groups while preserving text targets."""
    unique_targets = {sample.manifest.subtask for sample in samples}
    offset = max(1, len(samples) // max(1, len(unique_targets)))
    return [
        dataclasses.replace(
            sample,
            images=samples[(index + offset) % len(samples)].images,
        )
        for index, sample in enumerate(samples)
    ]


def _batches(samples: list[R0Sample], batch_size: int) -> Iterable[list[R0Sample]]:
    for start in range(0, len(samples), batch_size):
        yield samples[start : start + batch_size]


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _exact_match(predictions: list[dict[str, object]]) -> float:
    return sum(bool(item["exact_match"]) for item in predictions) / len(predictions)


def _eos_rate(predictions: list[dict[str, object]]) -> float:
    return sum(bool(item["ended_with_eos"]) for item in predictions) / len(predictions)


def _write_predictions(
    output_path: Path,
    initial: list[dict[str, object]],
    final: list[dict[str, object]],
    shuffled: list[dict[str, object]],
) -> None:
    with output_path.open("w", encoding="utf-8") as output_file:
        for split, predictions in (
            ("initial", initial),
            ("final", final),
            ("shuffled_images", shuffled),
        ):
            for prediction in predictions:
                output_file.write(json.dumps({"split": split, **prediction}) + "\n")


if __name__ == "__main__":
    main()
