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

import pytest
from omegaconf import OmegaConf

from rlinf.config import validate_fsdp_cfg
from rlinf.hybrid_engines.fsdp import utils
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import (
    _get_grad_norm_process_group,
)


@pytest.fixture
def mesh_factory(monkeypatch):
    calls = []

    def fake_init_device_mesh(device_type, *, mesh_shape, mesh_dim_names):
        call = (device_type, mesh_shape, mesh_dim_names)
        calls.append(call)
        return call

    monkeypatch.setattr(utils, "init_device_mesh", fake_init_device_mesh)
    monkeypatch.setattr(utils.Worker, "torch_device_type", "cuda")
    return calls


def test_create_device_mesh_preserves_one_dimensional_full_shard(mesh_factory):
    mesh = utils.create_device_mesh(8)

    assert mesh == ("cuda", (8,), ("fsdp",))
    assert mesh_factory == [mesh]


def test_create_device_mesh_builds_two_dimensional_hybrid_mesh(mesh_factory):
    mesh = utils.create_device_mesh(
        8,
        sharding_strategy="hybrid_shard",
        hybrid_shard_size=4,
        node_local_world_size=4,
    )

    assert mesh == ("cuda", (2, 4), ("replicate", "shard"))
    assert mesh_factory == [mesh]


class _FakeMeshDimension:
    def __init__(self, group):
        self._group = group

    def get_group(self):
        return self._group


class _FakeDeviceMesh:
    def __init__(self, dimensions):
        self.mesh_dim_names = tuple(dimensions)
        self.groups = {name: object() for name in dimensions}

    def __getitem__(self, name):
        return _FakeMeshDimension(self.groups[name])


@pytest.mark.parametrize(
    ("dimensions", "expected_dimension"),
    [
        (("replicate", "shard"), "shard"),
        (("ddp", "fsdp"), "ddp"),
        (("fsdp",), None),
    ],
)
def test_grad_norm_group_does_not_count_hybrid_replicas(dimensions, expected_dimension):
    mesh = _FakeDeviceMesh(dimensions)

    group = _get_grad_norm_process_group(mesh)

    expected_group = (
        mesh.groups[expected_dimension] if expected_dimension is not None else None
    )
    assert group is expected_group


@pytest.mark.parametrize(
    ("world_size", "hybrid_shard_size", "node_local_world_size", "message"),
    [
        (8, None, 4, "must be a positive integer"),
        (10, 4, 4, "is not divisible"),
        (4, 4, 4, "at least two replica groups"),
        (8, 4, 2, "must match the node-local"),
    ],
)
def test_create_device_mesh_rejects_invalid_hybrid_topology(
    mesh_factory,
    world_size,
    hybrid_shard_size,
    node_local_world_size,
    message,
):
    with pytest.raises(ValueError, match=message):
        utils.create_device_mesh(
            world_size,
            sharding_strategy="hybrid_shard",
            hybrid_shard_size=hybrid_shard_size,
            node_local_world_size=node_local_world_size,
        )

    assert mesh_factory == []


def _fsdp_actor_config(*, strategy="fsdp", sharding_strategy="full_shard", **kwargs):
    fsdp_config = {
        "strategy": strategy,
        "sharding_strategy": sharding_strategy,
        "mixed_precision": {
            "param_dtype": "bf16",
            "reduce_dtype": "fp32",
            "buffer_dtype": "fp32",
        },
        **kwargs,
    }
    return OmegaConf.create({"fsdp_config": fsdp_config})


def test_validate_fsdp_cfg_accepts_explicit_hybrid_shard_size():
    config = validate_fsdp_cfg(
        _fsdp_actor_config(sharding_strategy="hybrid_shard", hybrid_shard_size=4),
        world_size=8,
    )

    assert config.fsdp_config.hybrid_shard_size == 4


def test_validate_fsdp_cfg_requires_hybrid_shard_size():
    with pytest.raises(AssertionError, match="must be a positive integer"):
        validate_fsdp_cfg(_fsdp_actor_config(sharding_strategy="hybrid_shard"))


def test_validate_fsdp_cfg_rejects_hybrid_shard_with_fsdp2():
    with pytest.raises(AssertionError, match="currently requires"):
        validate_fsdp_cfg(
            _fsdp_actor_config(
                strategy="fsdp2",
                sharding_strategy="hybrid_shard",
                hybrid_shard_size=4,
            )
        )


def test_validate_fsdp_cfg_rejects_unknown_sharding_strategy():
    with pytest.raises(AssertionError, match="must be one of"):
        validate_fsdp_cfg(_fsdp_actor_config(sharding_strategy="invalid"))


@pytest.mark.parametrize(
    ("world_size", "message"),
    [(10, "must be divisible"), (4, "at least two replica groups")],
)
def test_validate_fsdp_cfg_rejects_invalid_hybrid_world_size(world_size, message):
    with pytest.raises(AssertionError, match=message):
        validate_fsdp_cfg(
            _fsdp_actor_config(sharding_strategy="hybrid_shard", hybrid_shard_size=4),
            world_size=world_size,
        )
