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

"""Count full policies and measure experts under identical synthetic conditions."""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
import yaml

from rlinf.models.embodiment.qwen35_fm.action_expert import (
    ActionExpertConfig,
    build_action_expert,
)
from rlinf.models.embodiment.qwen35_fm.policy import FlowMatchingPolicy
from rlinf.models.embodiment.qwen35_fm.vlm import Qwen35VLMWrapper, VLMCondition
from rlinf.utils.logging import get_logger


def main() -> None:
    """Report actual parameter counts and expert-only latency/memory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config-dir", default="examples/sft/config/qwen35_fm")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--precision", choices=["float32", "bfloat16"], default="bfloat16"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--intermediate-size", type=int)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--counts-only", action="store_true")
    parser.add_argument("--variants", help="Comma-separated variant subset")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if min(args.batch_size, args.horizon, args.sequence_length, args.iterations) < 1:
        raise ValueError("Benchmark sizes and iterations must be positive")
    from transformers import AutoConfig, Qwen3_5ForConditionalGeneration

    dtype = getattr(torch, args.precision)
    native_config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    with torch.device("meta"):
        native = Qwen3_5ForConditionalGeneration(native_config).to(dtype=dtype)
        native.tie_weights()
        vlm = Qwen35VLMWrapper(native)
    results = {
        "protocol": vars(args),
        "scope": "Expert only; synthetic VLM conditioning, no VLM encoding or optimizer latency",
        "models": {},
    }
    device = torch.device(args.device)
    if not args.counts_only and device.type == "cuda":
        torch.cuda.set_device(device)
        results["hardware"] = torch.cuda.get_device_name(device)
    results["torch_version"] = torch.__version__

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for path in sorted(Path(args.config_dir).glob("qwen35_*.yaml")):
        values = yaml.safe_load(path.read_text())["expert"]
        if args.variants and values["variant"] not in args.variants.split(","):
            continue
        for key in ["hidden_size", "intermediate_size"]:
            if getattr(args, key) is not None:
                values[key] = getattr(args, key)
        values["horizon"] = args.horizon
        config = ActionExpertConfig(**values)
        with torch.device("meta"):
            expert = build_action_expert(config, vlm.state_specs)
        policy = FlowMatchingPolicy(vlm, expert)
        record = policy.parameter_report(args.batch_size, args.sequence_length)
        record["expert_config"] = vars(config)
        if not args.counts_only:
            # Construct directly on target device; native GDN parameters have their own initialization.
            del expert, policy
            torch.manual_seed(42)
            expert = build_action_expert(config, vlm.state_specs).to(
                device=device, dtype=dtype
            )
            condition = VLMCondition(
                torch.randn(
                    args.batch_size,
                    args.sequence_length,
                    config.vlm_hidden_size,
                    device=device,
                    dtype=dtype,
                ),
                torch.ones(
                    args.batch_size,
                    args.sequence_length,
                    device=device,
                    dtype=torch.bool,
                ),
                {
                    layer: torch.randn(args.batch_size, *spec, device=device)
                    for layer, spec in vlm.state_specs.items()
                },
            )
            actions = torch.randn(
                args.batch_size,
                args.horizon,
                config.action_dim,
                device=device,
                dtype=dtype,
            )
            timestep = torch.full((args.batch_size,), 0.5, device=device)
            # Both forward and backward kernels warm up before measurement.
            for _ in range(2):
                expert.zero_grad(set_to_none=True)
                expert(actions, timestep, condition).float().square().mean().backward()
            synchronize()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            with torch.no_grad():
                start = time.perf_counter()
                for _ in range(args.iterations):
                    expert(actions, timestep, condition)
                synchronize()
                record["forward_ms"] = (
                    (time.perf_counter() - start) * 1000 / args.iterations
                )
            start = time.perf_counter()
            for _ in range(args.iterations):
                expert.zero_grad(set_to_none=True)
                expert(actions, timestep, condition).float().square().mean().backward()
            synchronize()
            record["forward_backward_ms"] = (
                (time.perf_counter() - start) * 1000 / args.iterations
            )
            record["peak_gpu_memory_bytes"] = (
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None
            )
            del expert, condition, actions, timestep
            gc.collect()
        else:
            del expert, policy
        results["models"][config.variant] = record
        get_logger().info(f"{config.variant}: {record}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
