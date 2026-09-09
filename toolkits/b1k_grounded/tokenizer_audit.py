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

"""Audit grounded-control token lengths against the real PaliGemma tokenizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import sentencepiece

from rlinf.data.b1k_grounded import (
    ControlProfile,
    ControlSerializer,
    GroundedControlSpec,
    ReservedTokenAllocator,
)

PROFILES = (
    ControlProfile.P0_DIRECT,
    ControlProfile.P1_SIMPLE_SG,
    ControlProfile.P2_GROUND_SG,
)


def extract_behavior_pi05_state(proprio: np.ndarray) -> np.ndarray:
    """Extract the 23 policy-state dimensions used by BehaviorInputs."""
    proprio = np.asarray(proprio, dtype=np.float32)
    if proprio.shape != (256,):
        raise ValueError(f"Expected a 256-dimensional B1K state, got {proprio.shape}.")
    return np.concatenate(
        [
            proprio[253:256],
            proprio[236:240],
            proprio[158:165],
            proprio[197:204],
            [proprio[193:195].sum()],
            [proprio[232:234].sum()],
        ]
    )


def _load_quantile_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    value = json.loads(path.read_text())
    state_stats = value["norm_stats"]["state"]
    q01 = np.asarray(state_stats["q01"], dtype=np.float32)
    q99 = np.asarray(state_stats["q99"], dtype=np.float32)
    if q01.shape != q99.shape or q01.shape[0] < 23:
        raise ValueError("State q01/q99 statistics must contain at least 23 values.")
    return q01[:23], q99[:23]


def _normalize_state(state: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    return (state - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def _pi05_token_ids(
    tokenizer: sentencepiece.SentencePieceProcessor,
    prompt: str,
    normalized_state: np.ndarray,
) -> list[int]:
    cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
    bins = np.linspace(-1, 1, 256 + 1)[:-1]
    discretized_state = np.digitize(normalized_state, bins=bins) - 1
    state_text = " ".join(map(str, discretized_state))
    full_prompt = f"Task: {cleaned_text}, State: {state_text};\nAction: "
    return list(tokenizer.encode(full_prompt, add_bos=True))


def _length_summary(lengths: list[int], max_token_len: int) -> dict[str, Any]:
    values = np.asarray(lengths)
    over_budget = int((values > max_token_len).sum())
    return {
        "min": int(values.min()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": int(values.max()),
        "over_budget": over_budget,
        "over_budget_fraction": over_budget / len(values),
    }


def run_tokenizer_audit(
    sidecar_path: str | Path,
    tokenizer_model: str | Path,
    norm_stats_path: str | Path,
    output_path: str | Path,
    *,
    mapping_output_path: str | Path | None = None,
    max_token_len: int = 200,
    model_vocab_size: int = 257_152,
) -> dict[str, Any]:
    """Audit all P0/P1/P2 prompts in a grounded sidecar and write JSON."""
    sidecar_path = Path(sidecar_path).resolve()
    tokenizer_model = Path(tokenizer_model).resolve()
    norm_stats_path = Path(norm_stats_path).resolve()
    output_path = Path(output_path).resolve()
    if max_token_len <= 0:
        raise ValueError("max_token_len must be positive.")

    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer_model))
    allocator = ReservedTokenAllocator(tokenizer, model_vocab_size=model_vocab_size)
    capabilities = allocator.inspect()
    mapping = allocator.allocate()
    serializer = ControlSerializer(mapping)
    q01, q99 = _load_quantile_stats(norm_stats_path)

    table = pq.read_table(
        sidecar_path,
        columns=[
            "sample_id",
            "skill",
            "control_json",
            "state",
            "fully_grounded",
        ],
    )
    profile_lengths: dict[ControlProfile, list[int]] = {
        profile: [] for profile in PROFILES
    }
    longest: dict[ControlProfile, list[dict[str, Any]]] = {
        profile: [] for profile in PROFILES
    }
    unknown_token_counts = dict.fromkeys(PROFILES, 0)

    for row in table.to_pylist():
        control = GroundedControlSpec.from_json(row["control_json"])
        policy_state = extract_behavior_pi05_state(row["state"])
        normalized_state = _normalize_state(policy_state, q01, q99)
        for profile in PROFILES:
            prompt = serializer.serialize(control, profile)
            token_ids = _pi05_token_ids(tokenizer, prompt, normalized_state)
            length = len(token_ids)
            profile_lengths[profile].append(length)
            unknown_token_counts[profile] += token_ids.count(tokenizer.unk_id())
            longest[profile].append(
                {
                    "sample_id": row["sample_id"],
                    "skill": row["skill"],
                    "arguments": len(control.arguments),
                    "fully_grounded": row["fully_grounded"],
                    "tokens": length,
                }
            )

    profile_reports = {}
    for profile in PROFILES:
        profile_reports[profile.value] = {
            "lengths": _length_summary(profile_lengths[profile], max_token_len),
            "unknown_tokens": unknown_token_counts[profile],
            "longest_samples": sorted(
                longest[profile], key=lambda item: (-item["tokens"], item["sample_id"])
            )[:10],
        }

    if mapping_output_path is None:
        mapping_output_path = output_path.with_name("structural_token_mapping.json")
    mapping_output_path = Path(mapping_output_path).resolve()
    report = {
        "sidecar_path": str(sidecar_path),
        "tokenizer_model": str(tokenizer_model),
        "norm_stats_path": str(norm_stats_path),
        "rows": table.num_rows,
        "max_token_len": max_token_len,
        "model_vocab_size": model_vocab_size,
        "tokenizer_capabilities": {
            "vocab_size": capabilities.vocab_size,
            "location_token_start": capabilities.location_token_start,
            "location_token_end": capabilities.location_token_end,
            "unused_token_count": capabilities.unused_token_count,
        },
        "mapping_output_path": str(mapping_output_path),
        "profiles": profile_reports,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    mapping_output_path.write_text(mapping.to_json(indent=2))
    return report


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mapping-output", type=Path)
    parser.add_argument("--max-token-len", type=int, default=200)
    parser.add_argument("--model-vocab-size", type=int, default=257_152)
    args = parser.parse_args()
    report = run_tokenizer_audit(
        args.sidecar,
        args.tokenizer_model,
        args.norm_stats,
        args.output,
        mapping_output_path=args.mapping_output,
        max_token_len=args.max_token_len,
        model_vocab_size=args.model_vocab_size,
    )
    print(
        json.dumps(
            {key: value["lengths"] for key, value in report["profiles"].items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
