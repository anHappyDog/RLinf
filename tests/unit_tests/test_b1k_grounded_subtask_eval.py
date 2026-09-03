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

from __future__ import annotations

import math

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rlinf.data.b1k_grounded import (
    GroundedControlSpec,
    SidecarControlIndex,
    episode_index_from_annotation_dir,
)
from toolkits.b1k_grounded.kit_cli import split_hydra_and_kit_args
from toolkits.b1k_grounded.subtask_eval_view import (
    prepare_subtask_evaluation_view,
)
from toolkits.b1k_grounded.subtask_predicates import (
    base_planar_pose_from_behavior_state,
    demo_terminal_pose_result,
    is_successful_predicate_termination,
    planar_pose_error,
)


def test_split_hydra_and_kit_args_preserves_both_consumers():
    hydra_args, kit_args = split_hydra_and_kit_args(
        [
            "task.name=preparing_lunch_box",
            "--portable-root",
            "/tmp/kit-local",
            "--/app/tokens/omni_global_cache=/tmp/kit-cache",
            "headless=true",
        ]
    )

    assert hydra_args == ["task.name=preparing_lunch_box", "headless=true"]
    assert kit_args == [
        "--portable-root",
        "/tmp/kit-local",
        "--/app/tokens/omni_global_cache=/tmp/kit-cache",
    ]


def test_episode_index_from_annotation_dir():
    assert (
        episode_index_from_annotation_dir(
            "/dataset/orchestrators/task-0000/episode_00000040"
        )
        == 40
    )

    with pytest.raises(ValueError, match="Cannot parse episode index"):
        episode_index_from_annotation_dir("/dataset/orchestrators/task-0000")


def test_sidecar_control_index_uses_episode_and_segment():
    control = GroundedControlSpec(
        goal="Turn on the radio.",
        subgoal=None,
        skill="press",
        arguments=(),
        timestep=100,
    )
    index = SidecarControlIndex({(40, 2): control})

    assert index.get(40, 2) == control
    with pytest.raises(KeyError, match="episode/segment"):
        index.get(40, 1)


def test_sidecar_control_index_resolves_dense_rows_by_interval_start(tmp_path):
    control = GroundedControlSpec(
        goal="Turn on the radio.",
        subgoal=None,
        skill="press",
        arguments=(),
    )
    sidecar = tmp_path / "sidecar.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "task_name": "turning_on_radio",
                    "episode_index": 40,
                    "segment_index": 2,
                    "interval_start": 100,
                    "interval_end": 150,
                    "control_json": GroundedControlSpec(
                        goal=control.goal,
                        subgoal=control.subgoal,
                        skill=control.skill,
                        arguments=control.arguments,
                        timestep=101,
                    ).to_json(),
                },
                {
                    "task_name": "turning_on_radio",
                    "episode_index": 40,
                    "segment_index": 2,
                    "interval_start": 100,
                    "interval_end": 150,
                    "control_json": control.to_json(),
                },
            ]
        ),
        sidecar,
    )

    index = SidecarControlIndex.from_parquet(sidecar, "turning_on_radio")

    assert index.interval_at_start(40, 100) == (2, 100, 150)
    assert index.intervals_for_episode(40) == ((2, 100, 150),)
    assert index.intervals_for_episode(41) == ()
    assert index.get(40, 2).timestep is None


def test_demo_terminal_navigation_predicate_uses_base_pose_not_eef_distance():
    state = np.zeros(256, dtype=np.float32)
    state[140:142] = [1.0, -2.0]
    state[145] = math.cos(0.5)
    state[148] = math.sin(0.5)
    target = base_planar_pose_from_behavior_state(state)
    current = np.asarray([1.3, -2.1, 0.7])

    position_error, yaw_error = planar_pose_error(current, target)
    result = demo_terminal_pose_result(
        current,
        target,
        position_threshold=0.5,
        yaw_threshold=math.radians(45),
    )

    assert position_error == pytest.approx(math.sqrt(0.1))
    assert yaw_error == pytest.approx(0.2)
    assert result["completed"]


def test_only_successful_predicate_termination_is_safe_during_warmup():
    successful = {
        "done": {
            "termination_conditions": {
                "timeout": {"done": False, "success": False},
                "predicate": {"done": True, "success": True},
            }
        }
    }
    timeout = {
        "done": {
            "termination_conditions": {
                "timeout": {"done": True, "success": False},
                "predicate": {"done": False, "success": False},
            }
        }
    }

    assert is_successful_predicate_termination(True, False, successful)
    assert not is_successful_predicate_termination(False, False, successful)
    assert not is_successful_predicate_termination(True, True, successful)
    assert not is_successful_predicate_termination(True, False, timeout)


def test_prepare_subtask_evaluation_view(tmp_path):
    source = tmp_path / "source"
    (source / "data").mkdir(parents=True)
    control = GroundedControlSpec(
        goal="Turn on the radio.",
        subgoal=None,
        skill="press",
        arguments=(),
    )
    sidecar = tmp_path / "sidecar.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "task_name": "turning_on_radio",
                    "episode_index": 40,
                    "segment_index": 2,
                    "interval_start": 100,
                    "interval_end": 150,
                    "skill": "press",
                    "control_json": control.to_json(),
                }
            ]
        ),
        sidecar,
    )

    view = prepare_subtask_evaluation_view(
        source,
        sidecar,
        tmp_path / "view",
        task_index=0,
        task_name="turning_on_radio",
    )

    assert (view / "data").resolve() == (source / "data").resolve()
    annotation = view / (
        "orchestrators/task-0000/episode_00000040/subtask_2_annotated.json"
    )
    assert annotation.is_file()
    assert '"start_frame": 100' in annotation.read_text()
