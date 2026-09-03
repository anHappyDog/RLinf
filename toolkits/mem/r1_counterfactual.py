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

"""Evaluate whether an R1 tail uses aligned RGB and current-state inputs."""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rlinf.data.datasets.openpi_rlinf.behavior.high_level import (
    PaligemmaSubtaskTokenizer,
    read_r1_manifest,
)
from rlinf.data.datasets.openpi_rlinf.behavior.high_level_state import (
    BehaviorStateReader,
    read_state_norm_stats,
)
from toolkits.mem.action_preservation import load_trainable_tail
from toolkits.mem.r0_overfit import R0Sample, load_pi05_model
from toolkits.mem.r1_train import (
    R1SampleDataset,
    _generation_metrics,
    _list_collate,
    evaluate_generation,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state-norm-stats", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tail-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-token-len", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    """Compare aligned inputs with deliberately mismatched modality controls."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("R1 counterfactual evaluation requires a CUDA GPU.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    entries = read_r1_manifest(args.manifest)
    tokenizer = PaligemmaSubtaskTokenizer(max_len=args.max_token_len)
    state_reader = BehaviorStateReader(args.dataset_root)
    state_stats = read_state_norm_stats(args.state_norm_stats)
    dataset = R1SampleDataset(
        args.dataset_root,
        entries,
        tokenizer,
        cache_size=len(entries),
        input_mode="image_state",
        state_reader=state_reader,
        state_stats=state_stats,
    )
    samples = [dataset[index] for index in range(len(dataset))]
    conditions = {
        "aligned": samples,
        "mismatched_images": mismatch_inputs(samples, modality="images"),
        "mismatched_states": mismatch_inputs(samples, modality="state"),
    }

    device = torch.device("cuda")
    model = load_pi05_model(args.model_path, device, args.max_token_len)
    load_trainable_tail(model, args.tail_checkpoint)
    predictions_by_condition = {}
    metrics = {}
    for condition, condition_samples in conditions.items():
        loader = DataLoader(
            condition_samples,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=_list_collate,
        )
        predictions = evaluate_generation(
            model,
            loader,
            tokenizer,
            args.max_new_tokens,
            device,
            "image_state",
        )
        predictions_by_condition[condition] = predictions
        metrics[condition] = _generation_metrics(predictions)

    aligned_exact = float(metrics["aligned"]["exact_match"])
    metrics["exact_match_drop"] = {
        condition: aligned_exact - float(condition_metrics["exact_match"])
        for condition, condition_metrics in metrics.items()
        if condition != "aligned" and isinstance(condition_metrics, dict)
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as output_file:
        for condition, predictions in predictions_by_condition.items():
            for prediction in predictions:
                output_file.write(
                    json.dumps({"condition": condition, **prediction}) + "\n"
                )
    print(json.dumps(metrics, indent=2))


def mismatch_inputs(
    samples: list[R0Sample], *, modality: str
) -> list[R0Sample]:
    """Replace one modality using samples from another target in the same task."""
    if modality not in {"images", "state"}:
        raise ValueError(f"Unsupported mismatch modality {modality!r}.")
    samples_by_task_target: dict[tuple[int, str], list[R0Sample]] = defaultdict(list)
    targets_by_task: dict[int, set[str]] = defaultdict(set)
    for sample in samples:
        task_index = sample.manifest.task_index
        target = sample.manifest.subtask
        samples_by_task_target[(task_index, target)].append(sample)
        targets_by_task[task_index].add(target)

    donor_offsets: dict[tuple[int, str], int] = defaultdict(int)
    mismatched = []
    for sample in samples:
        task_index = sample.manifest.task_index
        target = sample.manifest.subtask
        alternate_targets = sorted(targets_by_task[task_index].difference({target}))
        if not alternate_targets:
            raise ValueError(
                f"Task {task_index} has no alternate target for a counterfactual."
            )
        donor_target = alternate_targets[0]
        donors = samples_by_task_target[(task_index, donor_target)]
        donor_key = (task_index, target)
        donor = donors[donor_offsets[donor_key] % len(donors)]
        donor_offsets[donor_key] += 1
        if modality == "images":
            mismatched.append(dataclasses.replace(sample, images=donor.images))
        else:
            mismatched.append(
                dataclasses.replace(
                    sample,
                    prefix_tokens=donor.prefix_tokens,
                    prefix_input_mask=donor.prefix_input_mask,
                    prefix_ar_mask=donor.prefix_ar_mask,
                )
            )
    return mismatched


if __name__ == "__main__":
    main()
