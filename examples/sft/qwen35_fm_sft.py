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

"""Bounded Qwen3.5 FM SFT on existing native-split MolmoBot LeRobot data."""

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from examples.sft.qwen35_fm_data import MolmoBotSubset
from rlinf.models.embodiment.qwen35_fm.action_expert import (
    ActionExpertConfig,
    build_action_expert,
)
from rlinf.models.embodiment.qwen35_fm.policy import FlowMatchingPolicy
from rlinf.models.embodiment.qwen35_fm.vlm import Qwen35VLMWrapper
from rlinf.utils.logging import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--data-path", required=True, help="Parent of molmobot and molmobot_val"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--variants", help="Comma-separated variants; default is config variant"
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--train-episodes", type=int, default=4)
    parser.add_argument("--val-episodes", type=int, default=2)
    parser.add_argument("--chunks-per-episode", type=int, default=2)
    parser.add_argument("--cameras", nargs="+", default=["observation.images.exterior"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--precision", choices=["float32", "bfloat16"], default="bfloat16"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-vlm", action="store_true")
    parser.add_argument("--cache-conditions", action="store_true")
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--intermediate-size", type=int)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--num-kv-heads", type=int)
    parser.add_argument("--gdn-num-key-heads", type=int)
    parser.add_argument("--gdn-num-value-heads", type=int)
    parser.add_argument("--gdn-key-head-dim", type=int)
    parser.add_argument("--gdn-value-head-dim", type=int)
    parser.add_argument("--state-layers", type=int, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("Expected positive steps and batch size")
    if args.train_vlm and args.cache_conditions:
        raise ValueError("Condition caching requires a frozen VLM")
    if args.train_vlm and args.variants and len(args.variants.split(",")) > 1:
        raise ValueError(
            "Trainable VLM comparisons require separate runs from the same checkpoint"
        )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    logger = get_logger()
    config_values = yaml.safe_load(Path(args.config).read_text())
    config_values = config_values.get("expert", config_values)
    for key in (
        "hidden_size",
        "intermediate_size",
        "num_heads",
        "num_kv_heads",
        "gdn_num_key_heads",
        "gdn_num_value_heads",
        "gdn_key_head_dim",
        "gdn_value_head_dim",
    ):
        value = getattr(args, key)
        if value is not None:
            config_values[key] = value
    state_layers = config_values.pop("state_layers", args.state_layers)
    if args.state_layers is not None:
        state_layers = args.state_layers
    config = ActionExpertConfig(**config_values)
    data_root = Path(args.data_path)
    train = MolmoBotSubset(
        data_root / "molmobot",
        config.horizon,
        args.train_episodes,
        args.chunks_per_episode,
        args.cameras,
    )
    val = MolmoBotSubset(
        data_root / "molmobot_val",
        config.horizon,
        args.val_episodes,
        args.chunks_per_episode,
        args.cameras,
    )
    mean, std = train.normalization()
    if mean.numel() != config.action_dim:
        raise ValueError(
            f"Expected action_dim={config.action_dim}; data has {mean.numel()}"
        )
    train_ids = {x["provenance"]["source_episode_id"] for x in train.manifest}
    val_ids = {x["provenance"]["source_episode_id"] for x in val.manifest}
    if train_ids & val_ids:
        raise ValueError("Expected disjoint source episodes in train and validation")
    manifest = {
        "args": vars(args),
        "model_path": str(Path(args.model_path).resolve()),
        "train": train.manifest,
        "validation": val.manifest,
        "normalization": {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "fit": "selected training chunks",
        },
        "cameras": args.cameras,
        "image_resize": [args.image_size, args.image_size],
        "train_metadata_frames": train.info["total_frames"],
        "validation_metadata_frames": val.info["total_frames"],
        "protocol": "Offline FM plumbing diagnostic; no closed-loop task success measurement",
    }
    (output / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info(
        f"SFT data: {len(train)} train chunks, {len(val)} validation chunks; manifest={output}"
    )
    dtype = getattr(torch, args.precision)
    vlm = Qwen35VLMWrapper.from_pretrained(
        args.model_path,
        trainable=args.train_vlm,
        state_layers=state_layers,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(args.device)
    vlm.processor.tokenizer.padding_side = "left"
    manifest["state_layers"] = list(vlm.state_layers)
    manifest["state_specs"] = vlm.state_specs
    (output / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
    cache = {}
    images = {}
    extract_states = "gqa6_vlm_state" in (args.variants or config.variant)

    def batch(dataset, indices, split):
        key = (split, tuple(indices))
        actions = torch.stack(
            [(dataset.samples[i]["actions"] - mean) / std for i in indices]
        ).to(device=args.device, dtype=dtype)
        mask = torch.stack([dataset.samples[i]["mask"] for i in indices]).to(
            args.device
        )
        if key not in cache:
            texts, batch_images = [], []
            for i in indices:
                image_key = (split, i)
                if image_key not in images:
                    images[image_key] = [
                        im.resize((args.image_size, args.image_size))
                        for im in dataset.images(i)
                    ]
                item_images = images[image_key]
                content = [{"type": "image"} for _ in item_images] + [
                    {"type": "text", "text": dataset.samples[i]["episode"]["tasks"][0]}
                ]
                texts.append(
                    vlm.processor.apply_chat_template(
                        [{"role": "user", "content": content}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
                batch_images.extend(item_images)
            inputs = vlm.processor(
                text=texts, images=batch_images, padding=True, return_tensors="pt"
            )
            inputs = {
                k: v.to(
                    device=args.device,
                    dtype=dtype if v.is_floating_point() else v.dtype,
                )
                for k, v in inputs.items()
            }
            condition = vlm.encode(**inputs, extract_states=extract_states)
            if args.cache_conditions:
                cache[key] = condition
        else:
            condition = cache[key]
        return actions, mask, condition

    results = {}
    for variant in (args.variants or config.variant).split(","):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        expert_values = asdict(config)
        expert_values["variant"] = variant
        if args.variants:
            expert_values["gdn_depth"] = None
        expert = build_action_expert(
            ActionExpertConfig(**expert_values), state_specs=vlm.state_specs
        )
        expert_values = asdict(expert.config)
        expert.to(device=args.device, dtype=dtype)
        policy = FlowMatchingPolicy(vlm, expert)
        optimizer = torch.optim.AdamW(
            [p for p in policy.parameters() if p.requires_grad], lr=args.lr
        )
        losses = []
        start_time = time.perf_counter()
        policy.train()
        for step in range(args.steps):
            indices = [
                (step * args.batch_size + j) % len(train)
                for j in range(args.batch_size)
            ]
            actions, mask, condition = batch(train, indices, "train")
            optimizer.zero_grad(set_to_none=True)
            # Matched noise/time draws across architectures, independent of model initialization.
            torch.manual_seed(args.seed + 1000 + step)
            loss = policy.loss(actions, condition, action_mask=mask)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Expected finite FM loss, found {loss.item()}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            logger.info(
                f"{variant} step={step + 1}/{args.steps} fm_loss={loss.item():.6f}"
            )
        policy.eval()
        val_losses = []
        with torch.no_grad():
            for offset in range(0, len(val), args.batch_size):
                indices = list(range(offset, min(offset + args.batch_size, len(val))))
                actions, mask, condition = batch(val, indices, "validation")
                torch.manual_seed(args.seed + 100000 + offset)
                val_losses.append(
                    (
                        policy.loss(actions, condition, action_mask=mask).item(),
                        int(mask.sum().item()),
                    )
                )
        result = {
            "train_loss": losses,
            "validation_fm_loss": sum(v * n for v, n in val_losses)
            / sum(n for _, n in val_losses),
            "updates": args.steps,
            "training_chunk_draws": args.steps * args.batch_size,
            "validation_chunks": len(val),
            "validation_action_scalars": sum(n for _, n in val_losses),
            "elapsed_seconds": time.perf_counter() - start_time,
            "expert_parameters": sum(p.numel() for p in expert.parameters()),
            "expert_config": expert_values,
        }
        state = {
            "expert": expert.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": expert_values,
            "state_layers": list(vlm.state_layers),
            "state_specs": vlm.state_specs,
            "normalization_mean": mean,
            "normalization_std": std,
            "step": args.steps,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "model_path": args.model_path,
            "manifest": "data_manifest.json",
        }
        if args.train_vlm:
            state["vlm"] = vlm.state_dict()
        torch.save(state, output / f"{variant}.pt")
        results[variant] = result
        (output / "results.json").write_text(json.dumps(results, indent=2))
        del optimizer, policy, expert
    logger.info(f"Completed bounded SFT diagnostic: {output / 'results.json'}")


if __name__ == "__main__":
    main()
