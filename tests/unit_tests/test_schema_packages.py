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

"""Tests for the domain-oriented schema package layout."""

import importlib.util

import rlinf.data.schema as schema
from rlinf.data.schema import agentic, embodied


def test_embodied_schema_exports_canonical_types() -> None:
    """Embodied package and unified schema expose the canonical classes."""
    assert schema.embodied is embodied
    assert schema.Trajectory is embodied.Trajectory
    assert schema.LeRobotChunk is embodied.LeRobotChunk
    assert embodied.Trajectory.__module__ == "rlinf.data.schema.embodied.types"
    assert embodied.TrajectoryCollector.__module__ == (
        "rlinf.data.schema.embodied.trajectory"
    )


def test_agentic_schema_exports_canonical_types() -> None:
    """Agentic package and unified schema expose the canonical classes."""
    assert schema.agentic is agentic
    assert schema.RolloutRequest is agentic.RolloutRequest
    assert schema.RolloutResult is agentic.RolloutResult
    assert agentic.RolloutRequest.__module__ == "rlinf.data.schema.agentic.requests"
    assert agentic.RolloutResult.__module__ == "rlinf.data.schema.agentic.types"


def test_legacy_flat_schema_modules_are_removed() -> None:
    """Only the domain-oriented module paths remain importable."""
    legacy_modules = (
        "rlinf.data.schema.embodied_types",
        "rlinf.data.schema.trajectory_accumulator",
        "rlinf.data.schema.trajectory_collector",
        "rlinf.data.schema.embodied.trajectory_collector",
        "rlinf.data.schema.reasoning_requests",
        "rlinf.data.schema.reasoning_results",
    )
    for module_name in legacy_modules:
        assert importlib.util.find_spec(module_name) is None
