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

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
from omegaconf import OmegaConf

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "rlinf"
    / "envs"
    / "behavior"
    / "instance_generator.py"
)
SPEC = importlib.util.spec_from_file_location(
    "behavior_generate_activity_instances",
    MODULE_PATH,
)
behavior_generate_activity_instances = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(behavior_generate_activity_instances)


def test_load_env_cfg_rejects_config_without_omni_config(tmp_path):
    config_path = tmp_path / "behavior.yaml"
    config_path.write_text("foo: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="omni_config"):
        behavior_generate_activity_instances.load_env_cfg(str(config_path))


def test_build_sampling_omni_cfg_overrides_reset_related_fields(monkeypatch):
    monkeypatch.setattr(
        behavior_generate_activity_instances,
        "setup_omni_cfg",
        lambda env_cfg: OmegaConf.create(
            {
                "env": {"automatic_reset": True},
                "task": {
                    "activity_instance_id": 99,
                    "activity_instance_dir": "/tmp/original",
                    "instance_resample_mode": "offline",
                    "online_object_sampling": False,
                    "use_presampled_robot_pose": True,
                },
                "scene": {
                    "scene_model": "house_double_floor_lower",
                    "scene_instance": "old_instance",
                    "scene_file": "/tmp/old_scene.json",
                },
                "robots": [{"type": "R1Pro"}, {"type": "Fetch"}],
            }
        ),
    )

    omni_cfg = behavior_generate_activity_instances.build_sampling_omni_cfg(
        env_cfg=OmegaConf.create({"omni_config": {}}),
        seed=7,
        robot_staging_position=(1.0, 2.0, 3.0),
    )

    assert OmegaConf.select(omni_cfg, "env.automatic_reset") is False
    assert OmegaConf.select(omni_cfg, "task.activity_instance_id") == 7
    assert OmegaConf.select(omni_cfg, "task.activity_instance_dir") is None
    assert OmegaConf.select(omni_cfg, "task.instance_resample_mode") == "disabled"
    assert OmegaConf.select(omni_cfg, "task.online_object_sampling") is True
    assert OmegaConf.select(omni_cfg, "task.use_presampled_robot_pose") is False
    assert OmegaConf.select(omni_cfg, "scene.scene_instance") is None
    assert OmegaConf.select(omni_cfg, "scene.scene_file") is None
    assert OmegaConf.select(omni_cfg, "robots[0].position") == [1.0, 2.0, 3.0]
    assert OmegaConf.select(omni_cfg, "robots[1].position") == [1.0, 2.0, 3.0]


def test_resolve_output_dir_prefers_explicit_path():
    omni_cfg = OmegaConf.create(
        {
            "scene": {"scene_model": "house_double_floor_lower"},
            "task": {
                "activity_name": "turning_on_radio",
                "activity_instance_dir": "/tmp/from_cfg",
            },
        }
    )

    output_dir = behavior_generate_activity_instances.resolve_output_dir(
        omni_cfg,
        explicit_output_dir="/tmp/from_cli",
    )

    assert output_dir == Path("/tmp/from_cli")


def test_resolve_output_dir_falls_back_to_rlinf_default(monkeypatch):
    fake_asset_utils = types.ModuleType("omnigibson.utils.asset_utils")
    fake_asset_utils.get_task_instance_path = lambda scene_model: (
        f"/tmp/task_instances/{scene_model}"
    )
    monkeypatch.setitem(sys.modules, "omnigibson.utils.asset_utils", fake_asset_utils)

    omni_cfg = OmegaConf.create(
        {
            "scene": {"scene_model": "house_double_floor_lower"},
            "task": {
                "activity_name": "turning_on_radio",
                "activity_instance_dir": None,
            },
        }
    )

    output_dir = behavior_generate_activity_instances.resolve_output_dir(
        omni_cfg,
        explicit_output_dir=None,
    )

    assert output_dir == Path(
        "/tmp/task_instances/house_double_floor_lower/json/"
        "house_double_floor_lower_task_turning_on_radio_instances"
    )


def test_build_output_path_uses_expected_filename():
    class FakeTask:
        scene_name = "house_double_floor_lower"
        activity_name = "turning_on_radio"
        activity_definition_id = 3
        activity_instance_id = 17

        @staticmethod
        def get_cached_activity_scene_filename(
            scene_model,
            activity_name,
            activity_definition_id,
            activity_instance_id,
        ):
            return (
                f"{scene_model}_task_{activity_name}_{activity_definition_id}_"
                f"{activity_instance_id}_template"
            )

    output_dir = Path("/tmp/instances")

    template_path = behavior_generate_activity_instances.build_output_path(
        FakeTask(),
        output_dir,
        "template",
    )
    tro_state_path = behavior_generate_activity_instances.build_output_path(
        FakeTask(),
        output_dir,
        "tro_state",
    )

    assert (
        template_path
        == output_dir
        / "house_double_floor_lower_task_turning_on_radio_3_17_template.json"
    )
    assert (
        tro_state_path
        == output_dir
        / "house_double_floor_lower_task_turning_on_radio_3_17_template-tro_state.json"
    )


def test_resolve_validation_scene_path_prefers_stable(tmp_path, monkeypatch):
    fake_asset_utils = types.ModuleType("omnigibson.utils.asset_utils")
    fake_asset_utils.get_dataset_path = lambda dataset_name: str(tmp_path)
    monkeypatch.setitem(sys.modules, "omnigibson.utils.asset_utils", fake_asset_utils)

    scene_json_dir = tmp_path / "scenes" / "house_double_floor_lower" / "json"
    scene_json_dir.mkdir(parents=True)
    stable_path = scene_json_dir / "house_double_floor_lower_stable.json"
    stable_path.write_text("{}", encoding="utf-8")
    (scene_json_dir / "house_double_floor_lower_best.json").write_text(
        "{}",
        encoding="utf-8",
    )

    resolved_path = behavior_generate_activity_instances.resolve_validation_scene_path(
        "house_double_floor_lower"
    )

    assert resolved_path == stable_path


def test_resolve_validation_scene_path_falls_back_to_best(tmp_path, monkeypatch):
    fake_asset_utils = types.ModuleType("omnigibson.utils.asset_utils")
    fake_asset_utils.get_dataset_path = lambda dataset_name: str(tmp_path)
    monkeypatch.setitem(sys.modules, "omnigibson.utils.asset_utils", fake_asset_utils)

    scene_json_dir = tmp_path / "scenes" / "house_double_floor_lower" / "json"
    scene_json_dir.mkdir(parents=True)
    best_path = scene_json_dir / "house_double_floor_lower_best.json"
    best_path.write_text("{}", encoding="utf-8")

    resolved_path = behavior_generate_activity_instances.resolve_validation_scene_path(
        "house_double_floor_lower"
    )

    assert resolved_path == best_path


def test_dump_tro_state_includes_robot_poses_and_skips_agent(tmp_path, monkeypatch):
    fake_config_utils = types.ModuleType("omnigibson.utils.config_utils")
    fake_config_utils.TorchEncoder = json.JSONEncoder
    monkeypatch.setitem(sys.modules, "omnigibson.utils.config_utils", fake_config_utils)

    class FakeAgent:
        exists = True
        synset = "agent"

        def dump_state(self, serialized=False):
            raise AssertionError("agent state should not be dumped")

    class FakeObject:
        exists = True
        synset = "object"

        def dump_state(self, serialized=False):
            return {"foo": 1}

    class FakeScene:
        def get_task_metadata(self, key):
            assert key == "robot_poses"
            return {"R1Pro": [{"position": [1, 2, 3], "orientation": [0, 0, 0, 1]}]}

    class FakeTask:
        object_scope = {
            "agent.n.01_1": FakeAgent(),
            "obj_a": FakeObject(),
        }

    class FakeEnv:
        scene = FakeScene()
        task = FakeTask()

    output_path = tmp_path / "instance-tro_state.json"

    behavior_generate_activity_instances.dump_tro_state(
        FakeEnv(),
        output_path=output_path,
        overwrite=False,
    )

    with output_path.open("r", encoding="utf-8") as f:
        tro_state = json.load(f)

    assert tro_state == {
        "obj_a": {"foo": 1},
        "robot_poses": {
            "R1Pro": [{"position": [1, 2, 3], "orientation": [0, 0, 0, 1]}]
        },
    }

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        behavior_generate_activity_instances.dump_tro_state(
            FakeEnv(),
            output_path=output_path,
            overwrite=False,
        )


def test_main_uses_configured_activity_instance_dir_when_cli_output_dir_is_absent(
    monkeypatch,
):
    calls = {}

    monkeypatch.setattr(
        behavior_generate_activity_instances,
        "parse_args",
        lambda: types.SimpleNamespace(
            config="/tmp/behavior.yaml",
            output_format="template",
            start_idx=1,
            end_idx=2,
            seed=0,
            num_trials=3,
            output_dir=None,
            robot_staging_position=(-50.0, -50.0, -50.0),
            overwrite=False,
        ),
    )
    monkeypatch.setattr(
        behavior_generate_activity_instances,
        "load_env_cfg",
        lambda config_path: OmegaConf.create(
            {
                "omni_config": {
                    "task": {"activity_instance_dir": "/tmp/from_cfg"},
                }
            }
        ),
    )
    monkeypatch.setattr(
        behavior_generate_activity_instances,
        "install_patch",
        lambda: None,
    )
    monkeypatch.setattr(
        behavior_generate_activity_instances,
        "build_sampling_omni_cfg",
        lambda env_cfg, seed, robot_staging_position: OmegaConf.create(
            {
                "scene": {"scene_model": "house_double_floor_lower"},
                "task": {
                    "activity_name": "turning_on_radio",
                    "activity_instance_dir": None,
                },
            }
        ),
    )

    def fake_resolve_output_dir(omni_cfg, explicit_output_dir):
        calls["explicit_output_dir"] = explicit_output_dir
        return Path("/tmp/resolved")

    monkeypatch.setattr(
        behavior_generate_activity_instances,
        "resolve_output_dir",
        fake_resolve_output_dir,
    )
    monkeypatch.setattr(
        behavior_generate_activity_instances,
        "configure_sampling_macros",
        lambda: None,
    )
    monkeypatch.setattr(
        behavior_generate_activity_instances,
        "generate_activity_instances",
        lambda env, output_dir, output_format, start_idx, end_idx, num_trials, overwrite: (
            calls.update(
                {
                    "output_dir": output_dir,
                    "output_format": output_format,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "num_trials": num_trials,
                    "overwrite": overwrite,
                }
            )
        ),
    )

    fake_og_module = types.ModuleType("omnigibson")
    fake_og_module.Environment = lambda cfg: types.SimpleNamespace(close=lambda: None)
    fake_og_module.shutdown = lambda: None
    monkeypatch.setitem(sys.modules, "omnigibson", fake_og_module)

    behavior_generate_activity_instances.main()

    assert calls["explicit_output_dir"] == "/tmp/from_cfg"
    assert calls["output_dir"] == Path("/tmp/resolved")
    assert calls["output_format"] == "template"
    assert calls["start_idx"] == 1
    assert calls["end_idx"] == 2
    assert calls["num_trials"] == 3
    assert calls["overwrite"] is False
