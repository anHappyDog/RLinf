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

"""Direct skill predicates for the B1K ``preparing_lunch_box`` task.

The stage specification matches annotated episode 120010.  It is intentionally
episode-bound: B1K object instance names are part of the recorded scene, and a
different episode must supply its own audited specification.
"""

from __future__ import annotations

import torch as th
from omnigibson.object_states.inside import Inside
from omnigibson.object_states.next_to import NextTo
from omnigibson.object_states.open_state import Open
from omnigibson.reward_functions.sequential_task_reward import SequentialTaskReward
from omnigibson.reward_functions.support_utils import (
    get_min_eef_distance_to_obj,
    get_obj_center,
    get_stage_objects_by_name,
    is_supported_by_surface,
    is_target_in_hand,
)

PREPARING_LUNCH_BOX_EPISODE_120010_STAGES = (
    ("move to", "packing_box_210"),
    ("pick up from", "packing_box_210", "countertop_kelker_0"),
    ("move to", "burner_mjvqii_0"),
    ("place on next to", "packing_box_210", "burner_mjvqii_0", "chopping_board_211"),
    ("move to", "club_sandwich_209"),
    ("pick up from", "club_sandwich_209", "chopping_board_211"),
    ("move to", "packing_box_210"),
    ("place in", "club_sandwich_209", "packing_box_210"),
    ("move to", "chocolate_chip_cookie_207"),
    ("pick up from", "chocolate_chip_cookie_207", "chopping_board_211"),
    ("move to", "packing_box_210"),
    ("place in", "chocolate_chip_cookie_207", "packing_box_210"),
    ("move to", "half_apple_213"),
    ("pick up from", "half_apple_213", "chopping_board_211"),
    ("move to", "packing_box_210"),
    ("place in", "half_apple_213", "packing_box_210"),
    ("move to", "half_apple_212"),
    ("pick up from", "half_apple_212", "chopping_board_211"),
    ("move to", "packing_box_210"),
    ("place in", "half_apple_212", "packing_box_210"),
    ("move to", "fridge_dszchb_0"),
    ("open door", "fridge_dszchb_0"),
    ("pick up from", "bottle_of_tea_208", "fridge_dszchb_0"),
    ("close door", "fridge_dszchb_0"),
    ("move to", "packing_box_210"),
    ("place in", "bottle_of_tea_208", "packing_box_210"),
)


class PreparingLunchBoxReward(SequentialTaskReward):
    """Sequential direct predicates for episode 120010's 26 skill stages."""

    def __init__(
        self,
        move_to_success_threshold: float = 0.5,
        progress_scale: float = 4.0,
        dense_scale: float = 0.3,
        stage_completion_bonus: float = 1.0,
    ) -> None:
        self.move_to_success_threshold = move_to_success_threshold
        self.progress_scale = progress_scale
        self.dense_scale = dense_scale
        self._stage_specs = []
        super().__init__(stage_completion_bonus=stage_completion_bonus)

    def reset(self, task, env) -> None:
        """Resolve the recorded object names against the loaded B1K scene."""
        self._stage_specs = []
        for stage_index, annotation in enumerate(
            PREPARING_LUNCH_BOX_EPISODE_120010_STAGES
        ):
            skill, *object_names = annotation
            objects = get_stage_objects_by_name(env, object_names)
            missing_names = [
                name for name, obj in zip(object_names, objects) if obj is None
            ]
            if missing_names:
                raise RuntimeError(
                    "Preparing-lunch-box reward could not resolve scene objects: "
                    + ", ".join(missing_names)
                )
            self._stage_specs.append(
                {
                    "name": f"stage_{stage_index:02d}_{skill.replace(' ', '_')}",
                    "stage_type": skill,
                    "objects": objects,
                }
            )
        super().reset(task, env)

    def _build_stages(self, task, env):
        del task, env
        stages = []
        for spec in self._stage_specs:
            stage = dict(spec)
            stage["state"] = {
                "previous_distance": None,
                "was_inside": False,
                "was_open": False,
            }
            stages.append(stage)
        return stages

    @staticmethod
    def _is_inside(target, container) -> bool:
        return Inside in target.states and bool(
            target.states[Inside].get_value(container)
        )

    def _distance_reward(self, state: dict, distance: float) -> float:
        progress = self._progress_reward(
            state["previous_distance"],
            distance,
            self.progress_scale,
            invert=True,
        )
        state["previous_distance"] = distance
        return progress + self._exp_distance_reward(distance, self.dense_scale)

    def _evaluate_stage(self, stage, task, env, action):
        del task, action
        robot = env.robots[0]
        skill = stage["stage_type"]
        state = stage["state"]
        target = stage["objects"][0]
        secondary = stage["objects"][1] if len(stage["objects"]) > 1 else None
        tertiary = stage["objects"][2] if len(stage["objects"]) > 2 else None
        target_in_hand = is_target_in_hand(robot, target)

        if skill == "move to":
            distance = get_min_eef_distance_to_obj(robot, target)
            return {
                "reward": self._distance_reward(state, distance),
                "completed": distance <= self.move_to_success_threshold,
                "metrics": {
                    "eef_to_target_distance": distance,
                    "success_threshold": self.move_to_success_threshold,
                },
            }

        if skill == "pick up from":
            distance = get_min_eef_distance_to_obj(robot, target)
            at_source = self._is_inside(target, secondary) or is_supported_by_surface(
                target, secondary
            )
            return {
                "reward": self._distance_reward(state, distance),
                "completed": target_in_hand,
                "metrics": {
                    "eef_to_target_distance": distance,
                    "in_hand": target_in_hand,
                    "at_source": at_source,
                },
            }

        if skill == "place in":
            distance = get_min_eef_distance_to_obj(robot, secondary)
            inside = self._is_inside(target, secondary)
            reward = self._distance_reward(state, distance)
            if inside and not state["was_inside"]:
                reward += self.progress_scale
            state["was_inside"] = inside
            return {
                "reward": reward,
                "completed": inside and not target_in_hand,
                "metrics": {
                    "eef_to_container_distance": distance,
                    "inside_container": inside,
                    "in_hand": target_in_hand,
                },
            }

        if skill == "place on next to":
            target_distance = get_min_eef_distance_to_obj(robot, target)
            reference_distance = th.norm(
                get_obj_center(target) - get_obj_center(tertiary)
            ).item()
            on_support = is_supported_by_surface(target, secondary)
            next_to_reference = NextTo in target.states and bool(
                target.states[NextTo].get_value(tertiary)
            )
            reward = self._distance_reward(state, target_distance)
            reward += self._exp_distance_reward(reference_distance, self.dense_scale)
            return {
                "reward": reward,
                "completed": on_support and not target_in_hand,
                "metrics": {
                    "eef_to_target_distance": target_distance,
                    "target_to_reference_distance": reference_distance,
                    "in_hand": target_in_hand,
                    "on_support": on_support,
                    "next_to_reference": next_to_reference,
                },
            }

        if skill in {"open door", "close door"}:
            distance = get_min_eef_distance_to_obj(robot, target)
            is_open = bool(target.states[Open].get_value())
            state["was_open"] = state["was_open"] or is_open
            completed = is_open if skill == "open door" else not is_open
            return {
                "reward": self._distance_reward(state, distance),
                "completed": completed,
                "metrics": {
                    "eef_to_door_distance": distance,
                    "door_open": is_open,
                    "was_open": state["was_open"],
                },
            }

        raise ValueError(f"Unsupported preparing-lunch-box skill: {skill!r}")
