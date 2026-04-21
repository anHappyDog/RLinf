# Copyright 2025 The RLinf Authors.
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

import json
import sys
import types

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.envs.behavior.instance_loader import (
    ActivityInstanceFile,
    ActivityInstanceLoader,
    discover_activity_instance_files,
    load_activity_instance_tro_state,
    parse_activity_instance_filename,
)
from rlinf.envs.behavior.utils import (
    infer_done_from_omnigibson_info,
    sync_robot_after_pose_override,
)


def test_parse_activity_instance_filename_for_template():
    parsed = parse_activity_instance_filename(
        "Rs_int_task_turning_on_radio_3_17_template.json",
        activity_name="turning_on_radio",
        instance_file_format="template",
    )

    assert parsed == (3, 17)


def test_parse_activity_instance_filename_for_tro_state():
    parsed = parse_activity_instance_filename(
        "Rs_int_task_turning_on_radio_3_17_template-tro_state.json",
        activity_name="turning_on_radio",
        instance_file_format="tro_state",
    )

    assert parsed == (3, 17)


def test_parse_activity_instance_filename_rejects_mismatched_name():
    parsed = parse_activity_instance_filename(
        "Rs_int_task_washing_dishes_3_17_template.json",
        activity_name="turning_on_radio",
        instance_file_format="template",
    )

    assert parsed is None


def test_discover_activity_instance_files_filters_definition_id(tmp_path):
    (tmp_path / "Rs_int_task_turning_on_radio_0_2_template.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (tmp_path / "Rs_int_task_turning_on_radio_0_7_template.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (tmp_path / "Rs_int_task_turning_on_radio_1_9_template.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (tmp_path / "ignore_me.txt").write_text("", encoding="utf-8")

    instance_files = discover_activity_instance_files(
        activity_instance_dir=tmp_path,
        activity_name="turning_on_radio",
        activity_definition_id=0,
        instance_file_format="template",
    )

    assert instance_files == [
        ActivityInstanceFile(
            instance_id=2,
            path=str(tmp_path / "Rs_int_task_turning_on_radio_0_2_template.json"),
            file_format="template",
        ),
        ActivityInstanceFile(
            instance_id=7,
            path=str(tmp_path / "Rs_int_task_turning_on_radio_0_7_template.json"),
            file_format="template",
        ),
    ]


def test_discover_activity_instance_files_filters_explicit_file_format(tmp_path):
    (tmp_path / "Rs_int_task_turning_on_radio_0_2_template.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (tmp_path / "Rs_int_task_turning_on_radio_0_7_template-tro_state.json").write_text(
        "{}", encoding="utf-8"
    )

    instance_files = discover_activity_instance_files(
        activity_instance_dir=tmp_path,
        activity_name="turning_on_radio",
        activity_definition_id=0,
        instance_file_format="tro_state",
    )

    assert instance_files == [
        ActivityInstanceFile(
            instance_id=7,
            path=str(
                tmp_path / "Rs_int_task_turning_on_radio_0_7_template-tro_state.json"
            ),
            file_format="tro_state",
        ),
    ]


def test_discover_activity_instance_files_raises_when_empty(tmp_path):
    with pytest.raises(ValueError, match="No cached BEHAVIOR task instances"):
        discover_activity_instance_files(
            activity_instance_dir=tmp_path,
            activity_name="turning_on_radio",
            activity_definition_id=0,
            instance_file_format="template",
        )


def test_activity_instance_loader_builds_offline_loader(tmp_path):
    (tmp_path / "Rs_int_task_turning_on_radio_0_2_template.json").write_text(
        "{}", encoding="utf-8"
    )
    (tmp_path / "Rs_int_task_turning_on_radio_0_7_template.json").write_text(
        "{}", encoding="utf-8"
    )

    omni_cfg = OmegaConf.create(
        {
            "task": {
                "activity_name": "turning_on_radio",
                "activity_definition_id": 0,
                "activity_instance_id": 3,
                "activity_instance_dir": str(tmp_path),
                "instance_resample_mode": "offline",
                "instance_file_format": "template",
                "online_object_sampling": False,
                "use_presampled_robot_pose": True,
            },
        }
    )

    loader = ActivityInstanceLoader.from_omni_cfg(omni_cfg)

    assert loader.activity_name == "turning_on_radio"
    assert loader.activity_instance_id == 3
    assert loader.instance_resample_mode == "offline"
    assert loader.activity_instances == (
        ActivityInstanceFile(
            instance_id=2,
            path=str(tmp_path / "Rs_int_task_turning_on_radio_0_2_template.json"),
            file_format="template",
        ),
        ActivityInstanceFile(
            instance_id=7,
            path=str(tmp_path / "Rs_int_task_turning_on_radio_0_7_template.json"),
            file_format="template",
        ),
    )


def test_activity_instance_loader_rejects_auto_format(tmp_path):
    (tmp_path / "Rs_int_task_turning_on_radio_0_2_template.json").write_text(
        "{}", encoding="utf-8"
    )

    omni_cfg = OmegaConf.create(
        {
            "task": {
                "activity_name": "turning_on_radio",
                "activity_definition_id": 0,
                "activity_instance_id": 2,
                "activity_instance_dir": str(tmp_path),
                "instance_resample_mode": "offline",
                "instance_file_format": "auto",
                "online_object_sampling": False,
                "use_presampled_robot_pose": True,
            },
        }
    )

    with pytest.raises(ValueError, match="task.instance_file_format must be one of"):
        ActivityInstanceLoader.from_omni_cfg(omni_cfg)


def test_activity_instance_loader_prepare_reset_uses_fixed_tro_state_instance(
    monkeypatch,
):
    calls = []

    def fake_load_activity_instance_tro_state(
        env,
        instance_id,
        tro_file_path,
        reset_scene=False,
    ):
        calls.append((env, instance_id, tro_file_path, reset_scene))

    monkeypatch.setattr(
        "rlinf.envs.behavior.instance_loader.load_activity_instance_tro_state",
        fake_load_activity_instance_tro_state,
    )

    loader = ActivityInstanceLoader(
        omni_cfg=OmegaConf.create({"task": {}}),
        activity_name="turning_on_radio",
        activity_instance_id=7,
        instance_resample_mode="disabled",
        activity_instances=(
            ActivityInstanceFile(
                instance_id=7,
                path="/tmp/instance_7_template-tro_state.json",
                file_format="tro_state",
            ),
        ),
    )
    env_a = object()
    env_b = object()
    vec_env = types.SimpleNamespace(envs=[env_a, env_b])

    loader.prepare_reset(vec_env)

    assert calls == [
        (env_a, 7, "/tmp/instance_7_template-tro_state.json", False),
        (env_b, 7, "/tmp/instance_7_template-tro_state.json", False),
    ]


def test_infer_done_from_omnigibson_info_reads_termination_payload():
    info = {
        "done": {
            "success": False,
            "termination_conditions": {
                "timeout": {"done": False, "success": False},
                "predicate": {"done": False, "success": False},
            },
        }
    }

    assert infer_done_from_omnigibson_info(info) is False

    info["done"]["termination_conditions"]["predicate"]["done"] = True
    assert infer_done_from_omnigibson_info(info) is True


def test_sync_robot_after_pose_override_resets_to_current_joint_state():
    reset_calls = []

    class FakeRobot:
        n_joints = 3

        def __init__(self):
            self.controllers = {"base": types.SimpleNamespace(reset=lambda: None)}

        def keep_still(self):
            reset_calls.append(("keep_still", None))

        def get_joint_positions(self):
            return torch.tensor([1.0, 2.0, 3.0])

        def set_joint_positions(self, positions, drive=False):
            reset_calls.append(("set_joint_positions", positions.clone(), drive))

        def set_joint_velocities(self, velocities, drive=False):
            reset_calls.append(("set_joint_velocities", velocities.clone(), drive))

    robot = FakeRobot()
    sync_robot_after_pose_override(robot)

    assert reset_calls[0] == ("keep_still", None)
    assert reset_calls[1][0] == "set_joint_positions"
    assert torch.equal(reset_calls[1][1], torch.tensor([1.0, 2.0, 3.0]))
    assert reset_calls[1][2] is False
    assert reset_calls[2][0] == "set_joint_velocities"
    assert torch.equal(reset_calls[2][1], torch.zeros(3))
    assert reset_calls[2][2] is False
    assert reset_calls[3] == ("keep_still", None)


def test_load_activity_instance_tro_state_accepts_missing_robot_poses(
    tmp_path, monkeypatch
):
    tro_state_path = tmp_path / "instance-tro_state.json"
    tro_state_path.write_text(
        json.dumps(
            {
                "obj_a": {
                    "foo": 1,
                }
            }
        ),
        encoding="utf-8",
    )

    fake_og_module = types.ModuleType("omnigibson")
    fake_og_module.sim = types.SimpleNamespace(step_physics=lambda: None)
    monkeypatch.setitem(sys.modules, "omnigibson", fake_og_module)

    fake_python_utils = types.ModuleType("omnigibson.utils.python_utils")
    fake_python_utils.recursively_convert_to_torch = lambda value: value
    monkeypatch.setitem(sys.modules, "omnigibson.utils.python_utils", fake_python_utils)

    loaded_states = []

    class FakeObject:
        exists = True
        is_system = False

        def load_state(self, state, serialized=False):
            loaded_states.append((state, serialized))

        def keep_still(self):
            return None

    class FakeRobot:
        model_name = "robot_r1"

        def set_position_orientation(self, *args, **kwargs):
            raise AssertionError(
                "robot pose should not be loaded when robot_poses is absent"
            )

    class FakeScene:
        def __init__(self):
            self.metadata_calls = []
            self.updated = False
            self.reset_called = False

        def write_task_metadata(self, key, data):
            self.metadata_calls.append((key, data))

        def update_initial_file(self):
            self.updated = True

        def reset(self):
            self.reset_called = True

    class FakeTask:
        def __init__(self):
            self.activity_instance_id = None
            self.object_scope = {"obj_a": FakeObject()}

        def get_agent(self, env):
            return env.robot

    class FakeEnv:
        def __init__(self):
            self.scene = FakeScene()
            self.task = FakeTask()
            self.robot = FakeRobot()

    env = FakeEnv()

    load_activity_instance_tro_state(
        env,
        instance_id=7,
        tro_file_path=str(tro_state_path),
        reset_scene=False,
    )

    assert env.task.activity_instance_id == 7
    assert loaded_states == [({"foo": 1}, False)]
    assert env.scene.metadata_calls == [("robot_poses", None)]
    assert env.scene.updated is True
    assert env.scene.reset_called is False


def test_load_activity_instance_tro_state_rejects_unknown_task_relevant_object(
    tmp_path, monkeypatch
):
    tro_state_path = tmp_path / "instance-tro_state.json"
    tro_state_path.write_text(
        json.dumps(
            {
                "obj_missing": {
                    "foo": 1,
                }
            }
        ),
        encoding="utf-8",
    )

    fake_og_module = types.ModuleType("omnigibson")
    fake_og_module.sim = types.SimpleNamespace(step_physics=lambda: None)
    monkeypatch.setitem(sys.modules, "omnigibson", fake_og_module)

    fake_python_utils = types.ModuleType("omnigibson.utils.python_utils")
    fake_python_utils.recursively_convert_to_torch = lambda value: value
    monkeypatch.setitem(sys.modules, "omnigibson.utils.python_utils", fake_python_utils)

    class FakeRobot:
        model_name = "robot_r1"

    class FakeScene:
        def write_task_metadata(self, key, data):
            return None

        def update_initial_file(self):
            return None

        def reset(self):
            return None

    class FakeTask:
        def __init__(self):
            self.activity_instance_id = None
            self.object_scope = {}

        def get_agent(self, env):
            return env.robot

    class FakeEnv:
        def __init__(self):
            self.scene = FakeScene()
            self.task = FakeTask()
            self.robot = FakeRobot()

    env = FakeEnv()

    with pytest.raises(AssertionError, match="obj_missing"):
        load_activity_instance_tro_state(
            env,
            instance_id=7,
            tro_file_path=str(tro_state_path),
            reset_scene=False,
        )


def test_load_activity_instance_tro_state_ignores_agent_entity(tmp_path, monkeypatch):
    tro_state_path = tmp_path / "instance-tro_state.json"
    tro_state_path.write_text(
        json.dumps(
            {
                "agent.n.01_1": {
                    "joint_pos": list(range(26)),
                },
                "obj_a": {
                    "foo": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    fake_og_module = types.ModuleType("omnigibson")
    fake_og_module.sim = types.SimpleNamespace(step_physics=lambda: None)
    monkeypatch.setitem(sys.modules, "omnigibson", fake_og_module)

    fake_python_utils = types.ModuleType("omnigibson.utils.python_utils")
    fake_python_utils.recursively_convert_to_torch = lambda value: value
    monkeypatch.setitem(sys.modules, "omnigibson.utils.python_utils", fake_python_utils)

    loaded_states = []

    class FakeAgentObject:
        exists = True
        is_system = False
        synset = "agent"

        def load_state(self, state, serialized=False):
            raise AssertionError("agent state should be ignored when loading tro_state")

        def keep_still(self):
            return None

    class FakeObject:
        exists = True
        is_system = False
        synset = "object"

        def load_state(self, state, serialized=False):
            loaded_states.append((state, serialized))

        def keep_still(self):
            return None

    class FakeRobot:
        model_name = "robot_r1"

    class FakeScene:
        def write_task_metadata(self, key, data):
            return None

        def update_initial_file(self):
            return None

        def reset(self):
            return None

    class FakeTask:
        def __init__(self):
            self.activity_instance_id = None
            self.object_scope = {
                "agent.n.01_1": FakeAgentObject(),
                "obj_a": FakeObject(),
            }

        def get_agent(self, env):
            return env.robot

    class FakeEnv:
        def __init__(self):
            self.scene = FakeScene()
            self.task = FakeTask()
            self.robot = FakeRobot()

    env = FakeEnv()

    load_activity_instance_tro_state(
        env,
        instance_id=7,
        tro_file_path=str(tro_state_path),
        reset_scene=False,
    )

    assert loaded_states == [({"foo": 1}, False)]


def test_load_activity_instance_tro_state_syncs_robot_after_pose_override(
    tmp_path, monkeypatch
):
    tro_state_path = tmp_path / "instance-tro_state.json"
    tro_state_path.write_text(
        json.dumps(
            {
                "obj_a": {"foo": 1},
                "robot_poses": {
                    "robot_r1": [
                        {
                            "position": [1.0, 2.0, 3.0],
                            "orientation": [0.0, 0.0, 0.0, 1.0],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    fake_og_module = types.ModuleType("omnigibson")
    fake_og_module.sim = types.SimpleNamespace(step_physics=lambda: None)
    monkeypatch.setitem(sys.modules, "omnigibson", fake_og_module)

    fake_python_utils = types.ModuleType("omnigibson.utils.python_utils")
    fake_python_utils.recursively_convert_to_torch = lambda value: value
    monkeypatch.setitem(sys.modules, "omnigibson.utils.python_utils", fake_python_utils)

    class FakeObject:
        exists = True
        is_system = False
        synset = "object"

        def load_state(self, state, serialized=False):
            return None

        def keep_still(self):
            return None

    class FakeRobot:
        model_name = "robot_r1"
        n_joints = 2

        def __init__(self):
            self.calls = []

        def set_position_orientation(self, position, orientation, frame="world"):
            self.calls.append(
                ("set_position_orientation", position, orientation, frame)
            )

        def keep_still(self):
            self.calls.append(("keep_still",))

        def get_joint_positions(self):
            return torch.tensor([0.25, -0.5])

        def set_joint_positions(self, positions, drive=False):
            self.calls.append(("set_joint_positions", positions.clone(), drive))

        def set_joint_velocities(self, velocities, drive=False):
            self.calls.append(("set_joint_velocities", velocities.clone(), drive))

    class FakeScene:
        def __init__(self):
            self.metadata_calls = []

        def write_task_metadata(self, key, data):
            self.metadata_calls.append((key, data))

        def update_initial_file(self):
            return None

        def reset(self):
            return None

    class FakeTask:
        def __init__(self):
            self.activity_instance_id = None
            self.object_scope = {"obj_a": FakeObject()}

        def get_agent(self, env):
            return env.robot

    class FakeEnv:
        def __init__(self):
            self.scene = FakeScene()
            self.task = FakeTask()
            self.robot = FakeRobot()

    env = FakeEnv()

    load_activity_instance_tro_state(
        env,
        instance_id=7,
        tro_file_path=str(tro_state_path),
        reset_scene=False,
    )

    assert env.robot.calls[0] == (
        "set_position_orientation",
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
        "scene",
    )
    assert env.robot.calls[1] == ("keep_still",)
    assert env.robot.calls[2][0] == "set_joint_positions"
    assert torch.equal(env.robot.calls[2][1], torch.tensor([0.25, -0.5]))
    assert env.robot.calls[2][2] is False
    assert env.robot.calls[3][0] == "set_joint_velocities"
    assert torch.equal(env.robot.calls[3][1], torch.zeros(2))
    assert env.robot.calls[3][2] is False
    assert env.robot.calls[4] == ("keep_still",)


def test_load_activity_instance_tro_state_rebases_object_pose_for_scene_offset(
    tmp_path, monkeypatch
):
    tro_state_path = tmp_path / "instance-tro_state.json"
    tro_state_path.write_text(
        json.dumps(
            {
                "obj_a": {
                    "root_link": {
                        "pos": [1.0, 2.0, 3.0],
                        "ori": [0.0, 0.0, 0.0, 1.0],
                    },
                    "joint_pos": [0.0],
                    "joint_vel": [0.0],
                }
            }
        ),
        encoding="utf-8",
    )

    fake_og_module = types.ModuleType("omnigibson")
    fake_og_module.sim = types.SimpleNamespace(step_physics=lambda: None)
    monkeypatch.setitem(sys.modules, "omnigibson", fake_og_module)

    fake_python_utils = types.ModuleType("omnigibson.utils.python_utils")
    fake_python_utils.recursively_convert_to_torch = lambda value: value
    monkeypatch.setitem(sys.modules, "omnigibson.utils.python_utils", fake_python_utils)

    loaded_states = []

    class FakeObject:
        exists = True
        is_system = False
        synset = "object"

        def load_state(self, state, serialized=False):
            loaded_states.append((state, serialized))

        def keep_still(self):
            return None

    class FakeRobot:
        model_name = "robot_r1"

    class FakeScene:
        idx = 1
        scene_model = "house_double_floor_lower"

        def convert_scene_relative_pose_to_world(self, position, orientation):
            return (
                [position[0] + 100.0, position[1], position[2]],
                orientation,
            )

        def write_task_metadata(self, key, data):
            return None

        def update_initial_file(self):
            return None

        def reset(self):
            return None

    class FakeTask:
        activity_name = "turning_on_radio"

        def __init__(self):
            self.activity_instance_id = None
            self.object_scope = {"obj_a": FakeObject()}

        def get_agent(self, env):
            return env.robot

    class FakeEnv:
        def __init__(self):
            self.scene = FakeScene()
            self.task = FakeTask()
            self.robot = FakeRobot()

    env = FakeEnv()

    load_activity_instance_tro_state(
        env,
        instance_id=7,
        tro_file_path=str(tro_state_path),
        reset_scene=False,
    )

    assert loaded_states == [
        (
            {
                "root_link": {
                    "pos": [101.0, 2.0, 3.0],
                    "ori": [0.0, 0.0, 0.0, 1.0],
                },
                "joint_pos": [0.0],
                "joint_vel": [0.0],
            },
            False,
        )
    ]
