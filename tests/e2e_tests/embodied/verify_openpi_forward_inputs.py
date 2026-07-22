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

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from rlinf.data.trajectory import ValueRequest
from rlinf.models.embodiment.openpi import get_model
from rlinf.models.embodiment.openpi.forward_inputs import (
    OpenPILiberoForwardInputs,
)
from rlinf.workers.trajectory import infer_value_request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify typed OpenPI LIBERO forward inputs with a real model."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    return parser.parse_args()


def model_config(model_path: Path):
    return OmegaConf.create(
        {
            "model_type": "openpi",
            "model_path": str(model_path),
            "precision": None,
            "num_action_chunks": 5,
            "action_dim": 7,
            "num_steps": 4,
            "add_value_head": True,
            "openpi": {
                "config_name": "pi0_libero",
                "num_images_in_input": 2,
                "noise_level": 0.5,
                "action_chunk": 5,
                "num_steps": 4,
                "train_expert_only": True,
                "action_env_dim": 7,
                "noise_method": "flow_sde",
                "add_value_head": True,
                "detach_critic_input": True,
                "use_dsrl": False,
            },
        }
    )


def environment_observations(
    batch_size: int,
    image_size: int,
) -> dict[str, object]:
    generator = torch.Generator().manual_seed(7)
    image_shape = (batch_size, image_size, image_size, 3)
    return {
        "main_images": torch.randint(
            0,
            256,
            image_shape,
            dtype=torch.uint8,
            generator=generator,
        ),
        "wrist_images": torch.randint(
            0,
            256,
            image_shape,
            dtype=torch.uint8,
            generator=generator,
        ),
        "extra_view_images": None,
        "states": torch.randn(batch_size, 8, generator=generator),
        "task_descriptions": ["pick up the black bowl"] * batch_size,
    }


def assert_same_outputs(
    direct: dict[str, torch.Tensor],
    typed: dict[str, torch.Tensor],
) -> None:
    if direct.keys() != typed.keys():
        raise AssertionError(
            f"Model output keys differ: {direct.keys()} != {typed.keys()}."
        )
    for name in direct:
        if not torch.equal(direct[name], typed[name]):
            raise AssertionError(
                f"Model output {name!r} changed after typed conversion."
            )


def assert_select_round_trip(inputs: OpenPILiberoForwardInputs) -> None:
    partitions = [inputs.select([index]) for index in range(inputs.batch_size)]
    for field_index, (name, original) in enumerate(inputs.tensor_fields()):
        restored = torch.cat(
            [partition.tensor_fields()[field_index][1] for partition in partitions]
        )
        if not torch.equal(restored, original):
            raise AssertionError(f"Selected field {name!r} did not reassemble exactly.")


def main() -> None:
    args = parse_args()
    if not args.model_path.is_dir():
        raise FileNotFoundError(args.model_path)
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive.")

    torch.manual_seed(7)
    device = torch.device(args.device)
    model = get_model(model_config(args.model_path)).to(device).eval()
    observations = environment_observations(args.batch_size, args.image_size)

    with torch.no_grad():
        actions, rollout = model.predict_action_batch(
            observations,
            mode="train",
            compute_values=True,
        )
        raw_inputs = rollout["forward_inputs"]
        typed_inputs = OpenPILiberoForwardInputs.from_model_inputs(raw_inputs)
        direct_output = model.default_forward(
            forward_inputs=raw_inputs,
            compute_values=True,
        )
        typed_output = model.default_forward(
            **typed_inputs.to_model_kwargs(),
            compute_values=True,
        )

    assert_same_outputs(direct_output, typed_output)
    assert_select_round_trip(typed_inputs)
    value_result = infer_value_request(
        model,
        ValueRequest(
            global_step=0,
            rollout_epoch=0,
            chunk_step=0,
            slot_ids=tuple(range(args.batch_size)),
            kind="timeout",
            observations=observations,
        ),
    )
    if value_result.values.shape != (args.batch_size, 1):
        raise AssertionError(
            f"Value result has invalid shape {tuple(value_result.values.shape)}."
        )
    if not torch.isfinite(value_result.values).all():
        raise AssertionError("Value result contains non-finite values.")

    summary = {
        "schema": typed_inputs.schema_name,
        "schema_version": typed_inputs.schema_version,
        "batch_size": typed_inputs.batch_size,
        "field_names": [name for name, _ in typed_inputs.tensor_fields()],
        "field_shapes": {
            name: list(value.shape) for name, value in typed_inputs.tensor_fields()
        },
        "actions_shape": list(actions.shape),
        "logprobs_shape": list(typed_output["logprobs"].shape),
        "values_shape": list(typed_output["values"].shape),
        "value_request_shape": list(value_result.values.shape),
        "device": str(device),
        "status": "passed",
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
