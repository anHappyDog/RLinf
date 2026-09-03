from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.envs.behavior import demonstration_reset as demonstration_reset_module
from rlinf.envs.behavior.behavior_env import (
    BehaviorEnv,
    apply_short_memory_history_valid_lengths,
)
from rlinf.envs.behavior.demonstration_reset import (
    DemonstrationResetSpec,
    _reset_task_bookkeeping_at_current_state,
    load_demonstration_scene_file,
    load_demonstration_states,
    restore_demonstration_observations,
)


def _write_raw_demo(path, *, instance_id: int = 3, transitions=None) -> None:
    with h5py.File(path, "w") as hdf5_file:
        data_group = hdf5_file.create_group("data")
        data_group.attrs["config"] = json.dumps(
            {"task": {"activity_instance_id": instance_id}}
        )
        data_group.attrs["scene_file"] = json.dumps(
            {"metadata": {"episode": instance_id}}
        )
        demo_group = data_group.create_group("demo_0")
        demo_group.attrs["transitions"] = json.dumps(transitions or {})
        states = np.arange(8 * 6, dtype=np.float32).reshape(8, 6)
        demo_group.create_dataset("state", data=states)
        demo_group.create_dataset("state_size", data=np.array([4, 5, 6, 4, 5, 6, 4, 5]))


def test_demonstration_reset_loads_strided_trimmed_states(tmp_path) -> None:
    path = tmp_path / "episode.hdf5"
    _write_raw_demo(path)
    spec = DemonstrationResetSpec(
        path=str(path),
        frame_index=6,
        history_length=3,
        history_stride=2,
        expected_instance_id=3,
    )

    states = load_demonstration_states(spec)

    assert [frame for frame, _ in states] == [2, 4, 6]
    assert [len(state) for _, state in states] == [6, 5, 4]
    assert states[-1][1].tolist() == [36.0, 37.0, 38.0, 39.0]
    assert load_demonstration_scene_file(path) == {"metadata": {"episode": 3}}


def test_demonstration_reset_rejects_instance_mismatch_and_transitions(
    tmp_path,
) -> None:
    path = tmp_path / "episode.hdf5"
    _write_raw_demo(path, transitions={"5": {}})
    mismatch = DemonstrationResetSpec(
        path=str(path), frame_index=6, expected_instance_id=4
    )
    with pytest.raises(ValueError, match="does not match"):
        load_demonstration_states(mismatch)

    transition = DemonstrationResetSpec(
        path=str(path), frame_index=6, expected_instance_id=3
    )
    with pytest.raises(ValueError, match="dynamic scene transitions"):
        load_demonstration_states(transition)


class _SequentialReward:
    def __init__(self) -> None:
        self._stage_defs = []
        self.active_stage_index = None

    def reset(self, _task, _env) -> None:
        self._stage_defs = [
            {"name": "move_to_radio"},
            {"name": "pickup_from_support"},
            {"name": "press_radio"},
        ]

    def set_active_stage_index(self, stage_index: int) -> None:
        self.active_stage_index = stage_index


class _Resettable:
    def reset(self, *_args) -> None:
        pass


def test_demonstration_reset_primes_sequential_reward_stage() -> None:
    reward = _SequentialReward()
    task = SimpleNamespace(
        _termination_conditions={"done": _Resettable()},
        _reward_functions={"task_specific": reward},
        _reset_variables=lambda _env: None,
    )
    env = SimpleNamespace(task=task, _current_step=99)

    _reset_task_bookkeeping_at_current_state(env, "pickup_from_support")

    assert reward.active_stage_index == 1
    assert env._current_step == 0


def test_demonstration_history_validity_masks_oldest_frames() -> None:
    history = {
        "history_main_images": torch.ones(2, 4, 2),
        "history_wrist_images": torch.ones(2, 4, 2, 2),
        "history_states": torch.ones(2, 4, 3),
        "history_frame_mask": torch.ones(2, 4, dtype=torch.bool),
        "history_time_offsets": torch.tensor(
            [[-3.0, -2.0, -1.0, 0.0], [-3.0, -2.0, -1.0, 0.0]]
        ),
    }

    limited = apply_short_memory_history_valid_lengths(history, [2, 3])

    assert limited["history_frame_mask"].tolist() == [
        [False, False, True, True],
        [False, True, True, True],
    ]
    assert limited["history_states"][0, :2].count_nonzero() == 0
    assert limited["history_states"][1, 1:].count_nonzero() == 9
    assert limited["history_time_offsets"].tolist() == [
        [0.0, 0.0, -1.0, 0.0],
        [0.0, -2.0, -1.0, 0.0],
    ]


def test_demonstration_restore_refreshes_flatcache_before_observation(
    monkeypatch,
) -> None:
    events = []

    class _PoseAPI:
        @classmethod
        def invalidate(cls) -> None:
            events.append("invalidate")

        @classmethod
        def _refresh(cls) -> None:
            events.append("refresh")

    sim = SimpleNamespace(
        scenes=[object()],
        load_state=lambda _state, serialized: events.append(
            f"load:{serialized}"
        ),
        render=lambda: events.append("render"),
    )
    omnigibson = types.ModuleType("omnigibson")
    omnigibson.sim = sim
    omnigibson_utils = types.ModuleType("omnigibson.utils")
    usd_utils = types.ModuleType("omnigibson.utils.usd_utils")
    usd_utils.PoseAPI = _PoseAPI
    monkeypatch.setitem(sys.modules, "omnigibson", omnigibson)
    monkeypatch.setitem(sys.modules, "omnigibson.utils", omnigibson_utils)
    monkeypatch.setitem(sys.modules, "omnigibson.utils.usd_utils", usd_utils)
    monkeypatch.setattr(
        demonstration_reset_module,
        "load_demonstration_states",
        lambda _spec: ((1, torch.zeros(3)), (2, torch.ones(3))),
    )
    monkeypatch.setattr(
        demonstration_reset_module,
        "load_demonstration_scene_file",
        lambda _path: {"scene": "exact"},
    )
    resettable = _Resettable()
    task = SimpleNamespace(
        _termination_conditions={"done": resettable},
        _reward_functions={},
        _reset_variables=lambda _env: events.append("reset_task"),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(
            restore=lambda scene_file, update_initial_file: events.append(
                f"restore:{scene_file['scene']}:{update_initial_file}"
            )
        ),
        task=task,
        _current_step=4,
        reset=lambda: events.append("reset_env"),
        get_obs=lambda: (events.append("get_obs") or {"rgb": 1}, {}),
    )

    restored = restore_demonstration_observations(
        env, DemonstrationResetSpec(path="unused", frame_index=2)
    )

    assert [frame for frame, _ in restored] == [1, 2]
    assert events.index("restore:exact:True") < events.index("load:True")
    assert events.index("reset_env") < events.index("load:True")
    assert events.index("refresh") < events.index("get_obs")
    assert events.count("refresh") == 2


class _PromptController:
    def reset(self) -> None:
        pass


def test_behavior_reset_prewarms_demonstration_history() -> None:
    env = object.__new__(BehaviorEnv)
    env.enable_offload = False
    env.pool = object()
    env.num_envs = 1
    env.history_length = 3
    env.cfg = OmegaConf.create({"demonstration_reset_history_stride": 30})
    env.prompt_controller = _PromptController()
    env._record_metrics = lambda _rewards, infos: infos
    env._reset_metrics = lambda: None
    history = [
        (100, {"frame": 100}),
        (130, {"frame": 130}),
        (160, {"frame": 160}),
    ]
    env.env_reset = lambda: (
        [{"frame": 160}],
        [{"_rlinf_demonstration_history": history}],
    )
    wrapped_frames = []

    def wrap(raw_obs):
        wrapped_frames.append(raw_obs[0]["frame"])
        return {"frame": raw_obs[0]["frame"]}

    env._wrap_obs = wrap
    env.worker_info = SimpleNamespace(rank=0)

    observation, infos = env.reset()

    assert wrapped_frames == [100, 130, 160]
    assert observation == {"frame": 160}
    assert env._history_step == 60
    assert "_rlinf_demonstration_history" not in infos[0]
