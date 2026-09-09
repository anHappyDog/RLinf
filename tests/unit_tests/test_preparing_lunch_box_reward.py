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

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


class _SequentialTaskReward:
    def __init__(self, stage_completion_bonus=1.0):
        self.stage_completion_bonus = stage_completion_bonus

    def reset(self, task, env):
        self._stage_defs = self._build_stages(task, env)

    @staticmethod
    def _progress_reward(previous, current, scale, invert=False):
        if previous is None:
            return 0.0
        return ((previous - current) if invert else (current - previous)) * scale

    @staticmethod
    def _exp_distance_reward(distance, scale):
        return math.exp(-distance) * scale


class _State:
    def __init__(self, value=False):
        self.value = value

    def get_value(self, other=None):
        del other
        return self.value


def _load_reward_module(monkeypatch):
    inside = type("Inside", (), {})
    next_to = type("NextTo", (), {})
    open_state = type("Open", (), {})
    object_states = {
        "omnigibson.object_states.inside": ("Inside", inside),
        "omnigibson.object_states.next_to": ("NextTo", next_to),
        "omnigibson.object_states.open_state": ("Open", open_state),
    }
    for module_name, (attribute, value) in object_states.items():
        module = ModuleType(module_name)
        setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, module_name, module)

    sequential = ModuleType("omnigibson.reward_functions.sequential_task_reward")
    sequential.SequentialTaskReward = _SequentialTaskReward
    monkeypatch.setitem(
        sys.modules,
        "omnigibson.reward_functions.sequential_task_reward",
        sequential,
    )

    support = ModuleType("omnigibson.reward_functions.support_utils")
    support.get_min_eef_distance_to_obj = lambda robot, obj: obj.distance
    support.get_obj_center = lambda obj: obj.center
    support.get_stage_objects_by_name = lambda env, names: [
        env.objects[name] for name in names
    ]
    support.is_supported_by_surface = lambda target, source: target.on_support
    support.is_target_in_hand = lambda robot, target: target.in_hand
    monkeypatch.setitem(
        sys.modules,
        "omnigibson.reward_functions.support_utils",
        support,
    )

    module_name = "test_preparing_lunch_box_reward_impl"
    source = Path(__file__).parents[2] / (
        "toolkits/b1k_grounded/preparing_lunch_box_reward.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, inside


def test_episode_120010_stage_spec_has_all_unique_ordered_stages(monkeypatch):
    module, _ = _load_reward_module(monkeypatch)
    names = {
        object_name
        for stage in module.PREPARING_LUNCH_BOX_EPISODE_120010_STAGES
        for object_name in stage[1:]
    }
    objects = {
        name: SimpleNamespace(
            states={}, distance=0.0, center=0.0, in_hand=False, on_support=False
        )
        for name in names
    }
    reward = module.PreparingLunchBoxReward()
    reward.reset(None, SimpleNamespace(objects=objects))

    assert len(reward._stage_defs) == 26
    assert len({stage["name"] for stage in reward._stage_defs}) == 26
    assert reward._stage_defs[0]["stage_type"] == "move to"
    assert reward._stage_defs[-1]["stage_type"] == "place in"


def test_place_in_requires_inside_and_released(monkeypatch):
    module, inside = _load_reward_module(monkeypatch)
    container = SimpleNamespace(distance=0.2)
    inside_state = _State(True)
    target = SimpleNamespace(
        states={inside: inside_state},
        distance=0.1,
        center=0.0,
        in_hand=True,
        on_support=False,
    )
    reward = module.PreparingLunchBoxReward()
    stage = {
        "stage_type": "place in",
        "objects": [target, container],
        "state": {"previous_distance": None, "was_inside": False},
    }
    env = SimpleNamespace(robots=[object()])

    assert not reward._evaluate_stage(stage, None, env, None)["completed"]
    target.in_hand = False
    assert reward._evaluate_stage(stage, None, env, None)["completed"]


def test_place_on_next_to_uses_annotated_support_surface(monkeypatch):
    module, _ = _load_reward_module(monkeypatch)
    target = SimpleNamespace(
        states={},
        distance=0.1,
        center=module.th.tensor([0.0, 0.0, 0.0]),
        in_hand=False,
        on_support=True,
    )
    support = object()
    reference = SimpleNamespace(center=module.th.tensor([1.0, 0.0, 0.0]))
    reward = module.PreparingLunchBoxReward()
    stage = {
        "stage_type": "place on next to",
        "objects": [target, support, reference],
        "state": {"previous_distance": None},
    }
    env = SimpleNamespace(robots=[object()])

    result = reward._evaluate_stage(stage, None, env, None)

    assert result["completed"]
    assert not result["metrics"]["next_to_reference"]


def test_pickup_from_fridge_requires_target_in_hand(monkeypatch):
    module, inside = _load_reward_module(monkeypatch)
    fridge = object()
    inside_state = _State(True)
    bottle = SimpleNamespace(
        states={inside: inside_state},
        distance=0.1,
        center=0.0,
        in_hand=False,
        on_support=False,
    )
    reward = module.PreparingLunchBoxReward()
    stage = {
        "stage_type": "pick up from",
        "objects": [bottle, fridge],
        "state": {"previous_distance": None},
    }
    env = SimpleNamespace(robots=[object()])

    assert not reward._evaluate_stage(stage, None, env, None)["completed"]
    bottle.in_hand = True
    assert reward._evaluate_stage(stage, None, env, None)["completed"]
