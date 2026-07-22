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

import pytest
import torch

from rlinf.data.forward_inputs import get_forward_inputs_type
from rlinf.models.embodiment.openpi.forward_inputs import (
    OpenPILiberoForwardInputs,
)


def model_inputs(batch_size: int = 4) -> dict[str, torch.Tensor]:
    return {
        "chains": torch.arange(batch_size * 5 * 5 * 32, dtype=torch.float32).reshape(
            batch_size, 5, 5, 32
        ),
        "denoise_inds": torch.arange(4).repeat(batch_size, 1),
        "tokenized_prompt": torch.arange(batch_size * 16).reshape(batch_size, 16),
        "tokenized_prompt_mask": torch.ones(batch_size, 16, dtype=torch.bool),
        "action": torch.arange(batch_size * 35, dtype=torch.float32).reshape(
            batch_size, 35
        ),
        "model_action": torch.arange(batch_size * 50 * 32, dtype=torch.float32).reshape(
            batch_size, 50 * 32
        ),
        "observation/image": torch.arange(batch_size * 8 * 8 * 3, dtype=torch.int32)
        .remainder(256)
        .to(torch.uint8)
        .reshape(batch_size, 8, 8, 3),
        "observation/wrist_image": torch.arange(
            batch_size * 8 * 8 * 3, dtype=torch.int32
        )
        .remainder(256)
        .to(torch.uint8)
        .reshape(batch_size, 8, 8, 3),
        "observation/state": torch.arange(batch_size * 8, dtype=torch.float32).reshape(
            batch_size, 8
        ),
    }


def test_openpi_schema_is_registered() -> None:
    assert get_forward_inputs_type("openpi_libero", 1) is OpenPILiberoForwardInputs


def test_tensor_fields_have_stable_model_key_order() -> None:
    inputs = OpenPILiberoForwardInputs.from_model_inputs(model_inputs())

    assert tuple(name for name, _ in inputs.tensor_fields()) == (
        "chains",
        "denoise_inds",
        "tokenized_prompt",
        "tokenized_prompt_mask",
        "action",
        "model_action",
        "observation/image",
        "observation/wrist_image",
        "observation/state",
    )


def test_to_model_kwargs_preserves_tensor_values() -> None:
    original = model_inputs()
    inputs = OpenPILiberoForwardInputs.from_model_inputs(original)

    model_kwargs = inputs.to_model_kwargs()

    assert set(model_kwargs) == {"forward_inputs"}
    restored = model_kwargs["forward_inputs"]
    assert isinstance(restored, dict)
    for name, value in original.items():
        assert restored[name] is value


def test_select_uses_batch_axis_and_requested_order() -> None:
    original = model_inputs()
    inputs = OpenPILiberoForwardInputs.from_model_inputs(original)

    selected = inputs.select(torch.tensor([3, 1]))

    assert selected.batch_size == 2
    for name, value in selected.tensor_fields():
        assert torch.equal(value, original[name][[3, 1]])


def test_selected_partitions_reassemble_exact_original_batch() -> None:
    inputs = OpenPILiberoForwardInputs.from_model_inputs(model_inputs())
    partitions = (inputs.select([0, 1]), inputs.select([2, 3]))

    for field_index, (name, original) in enumerate(inputs.tensor_fields()):
        reassembled = torch.cat(
            [partition.tensor_fields()[field_index][1] for partition in partitions]
        )
        assert torch.equal(reassembled, original), name


def test_schema_rejects_missing_or_unexpected_fields() -> None:
    missing = model_inputs()
    del missing["chains"]
    with pytest.raises(ValueError, match=r"missing=\['chains'\]"):
        OpenPILiberoForwardInputs.from_model_inputs(missing)

    unexpected = model_inputs()
    unexpected["nft_x0"] = torch.zeros(4, 5, 32)
    with pytest.raises(ValueError, match=r"unexpected=\['nft_x0'\]"):
        OpenPILiberoForwardInputs.from_model_inputs(unexpected)


def test_schema_rejects_misaligned_batch() -> None:
    values = model_inputs()
    values["observation/state"] = torch.zeros(3, 8)

    with pytest.raises(ValueError, match="observation/state"):
        OpenPILiberoForwardInputs.from_model_inputs(values)


def test_schema_rejects_invalid_prompt_mask_dtype() -> None:
    values = model_inputs()
    values["tokenized_prompt_mask"] = torch.ones(4, 16)

    with pytest.raises(TypeError, match="tokenized_prompt_mask"):
        OpenPILiberoForwardInputs.from_model_inputs(values)


def test_schema_rejects_inconsistent_denoising_axis() -> None:
    values = model_inputs()
    values["denoise_inds"] = torch.zeros(4, 5, dtype=torch.int64)

    with pytest.raises(ValueError, match="one more denoising state"):
        OpenPILiberoForwardInputs.from_model_inputs(values)
