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

"""Run official B1K skill-reset evaluation with Oracle control conditions.

This entrypoint reuses BEHAVIOR-1K's extended ``eval.py`` subtask loop.  It
only adds the condition transport required by grounded-control SFT:

* P1 sends the current ground-truth skill and typed object arguments.
* P2 additionally recomputes object bboxes from the current simulator masks.

The segmentation images are removed before the observation is sent over the
websocket, so privileged state remains local to the Oracle evaluator.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

import hydra
import numpy as np
import omnigibson as og
import omnigibson.learning.eval as b1k_eval
import omnigibson.utils.transform_utils as transform_utils
import pyarrow.parquet as pq
from omegaconf import OmegaConf, open_dict
from omnigibson.learning.utils.config_utils import register_omegaconf_resolvers
from omnigibson.learning.utils.eval_utils import (
    ROBOT_CAMERA_NAMES,
    get_task_specific_reward,
)
from omnigibson.macros import gm
from omnigibson.sensors.vision_sensor import VisionSensor

from rlinf.data.b1k_grounded import (
    GROUND_CONTROL_JSON_KEY,
    CameraID,
    ControlProfile,
    ControlSerializer,
    EntityResolver,
    GroundedControlSpec,
    ReservedTokenMapping,
    SidecarControlIndex,
    episode_index_from_annotation_dir,
    ground_control_spec,
)
from rlinf.envs.behavior.subpool import (
    SubpoolSnapshot,
    SubpoolStore,
    full_state_sha256,
    validate_subpool_export_request,
)
from rlinf.envs.behavior.subpool_reward import (
    SubtaskRewardSpec,
    validate_demo_horizon,
)
from rlinf.envs.behavior.utils import sync_robot_after_pose_override
from toolkits.b1k_grounded.kit_cli import split_hydra_and_kit_args
from toolkits.b1k_grounded.preparing_lunch_box_reward import (
    PreparingLunchBoxReward,
)
from toolkits.b1k_grounded.subtask_eval_view import (
    prepare_subtask_evaluation_view,
)
from toolkits.b1k_grounded.subtask_predicates import (
    base_planar_pose_from_behavior_state,
    demo_terminal_pose_result,
    is_successful_predicate_termination,
)

_SEGMENTATION_MODALITIES = ("seg_semantic", "seg_instance_id")
_CAMERA_IDS = {
    "head": CameraID.HEAD,
    "left_wrist": CameraID.LEFT_WRIST,
    "right_wrist": CameraID.RIGHT_WRIST,
}


class _CalibrationPolicy:
    """No-op policy placeholder used while replaying ground-truth actions."""

    def reset(self) -> None:
        """Match the evaluator policy lifecycle."""


class GroundedSubtaskEvaluator(b1k_eval.Evaluator):
    """Official evaluator with local Oracle-condition construction."""

    def __init__(self, cfg) -> None:
        self._control_profile = ControlProfile(cfg.grounded_control_profile)
        self._infer_missing_parts = bool(cfg.get("grounded_infer_missing_parts", True))
        self._control_index = SidecarControlIndex.from_parquet(
            cfg.grounded_control_sidecar,
            cfg.task.name,
        )
        self._last_logged_stage = None
        self._target_segment_index = None
        self._target_base_pose = None
        super().__init__(cfg)

    def reset(self) -> None:
        """Clear the selected skill before resetting the underlying evaluator."""
        self._target_segment_index = None
        self._target_base_pose = None
        super().reset()

    def __exit__(self, exc_type, exc_value, exc_tb):
        """Close Kit even when OmniGibson races while deleting temporary USDs.

        Some B1K scenes still have background writers holding a temporary USD
        directory when ``og.shutdown()`` begins.  The upstream evaluator lets
        the resulting ``ENOTEMPTY`` escape before Kit is closed, which turns a
        successful export into a non-zero exit and can subsequently segfault.
        Preserve all upstream teardown behavior, but finish closing the app
        when that known cleanup race occurs.
        """
        try:
            return super().__exit__(exc_type, exc_value, exc_tb)
        except OSError as error:
            if error.errno != 39:  # ENOTEMPTY
                raise
            b1k_eval.logger.warning(
                "Ignoring OmniGibson temporary-directory cleanup race: %s", error
            )
            if og.app is not None:
                og.app.close()
            return None

    def load_env(self, env_wrapper):
        """Enable instance masks only when P2 online grounding needs them."""
        env = super().load_env(env_wrapper)
        if self.cfg.task.name == "preparing_lunch_box":
            env.task._reward_functions["task_specific"] = PreparingLunchBoxReward()  # noqa: SLF001
            b1k_eval.logger.info(
                "Installed episode-120010 preparing-lunch-box direct predicates."
            )
        if self._control_profile is not ControlProfile.P2_GROUND_SG:
            return env

        for camera_name in ROBOT_CAMERA_NAMES["R1Pro"].values():
            sensor = env.robots[0].sensors[camera_name.split("::")[1]]
            for modality in _SEGMENTATION_MODALITIES:
                sensor.add_modality(modality)
        env.load_observation_space()
        b1k_eval.logger.info("Enabled local instance masks for P2 Oracle grounding.")
        return env

    def load_policy(self):
        """Avoid opening a policy connection during demo-only calibration."""
        if self.cfg.get("grounded_demo_calibration", False):
            b1k_eval.logger.info(
                "Using ground-truth demo replay for predicate calibration."
            )
            return _CalibrationPolicy()
        return super().load_policy()

    def _active_segment_index(self) -> int:
        if self._target_segment_index is not None:
            return self._target_segment_index
        reward = self.env.task._reward_functions["task_specific"]  # noqa: SLF001
        if not reward._stage_defs:  # noqa: SLF001
            raise RuntimeError("Task-specific reward has no initialized stages.")
        return min(reward._stage_index, len(reward._stage_defs) - 1)  # noqa: SLF001

    def _load_demo_terminal_base_pose(
        self,
        demo_data_dir: str,
        task_index: int,
        episode_index: int,
        end_frame: int,
    ) -> np.ndarray:
        parquet_path = (
            Path(demo_data_dir)
            / "data"
            / f"task-{task_index:04d}"
            / f"episode_{episode_index:08d}.parquet"
        )
        states = pq.read_table(parquet_path, columns=["observation.state"])[
            "observation.state"
        ]
        if not 0 < end_frame <= len(states):
            raise IndexError(
                f"Interval end {end_frame} is outside demo episode {episode_index} "
                f"with {len(states)} frames."
            )
        return base_planar_pose_from_behavior_state(
            np.asarray(states[end_frame - 1].as_py(), dtype=np.float32)
        )

    def _align_robot_to_demo_frame(
        self,
        *,
        demo_data_dir: str,
        task_index: int,
        episode_index: int,
        frame_index: int,
    ) -> None:
        """Correct accumulated robot drift at a demo subtask boundary.

        Demo replay remains responsible for reconstructing object state. Only
        the robot base and articulation are aligned to the recorded boundary
        observation before replay continues.
        """
        self.load_subtask_init_state(
            demo_data_dir=demo_data_dir,
            task_index=task_index,
            episode_index=episode_index,
            start_frame=frame_index,
        )
        sync_robot_after_pose_override(self.robot)
        b1k_eval.logger.info(
            "Aligned robot to demo boundary: episode=%d frame=%d.",
            episode_index,
            frame_index,
        )

    def _replay_demo_warmup(
        self,
        *,
        demo_data_dir: str,
        task_index: int,
        episode_index: int,
        target_start_frame: int,
        require_successful_terminations: bool,
    ) -> tuple[int, int]:
        """Replay a demo prefix while correcting robot drift at each boundary."""
        parquet_path = (
            Path(demo_data_dir)
            / "data"
            / f"task-{task_index:04d}"
            / f"episode_{episode_index:08d}.parquet"
        )
        intermediate_boundaries = sorted(
            {
                start
                for _, start, _ in self._control_index.intervals_for_episode(
                    episode_index
                )
                if 0 < start < target_start_frame
            }
        )
        chunk_boundaries = [0, *intermediate_boundaries, target_start_frame]
        warmup_steps = 0
        ignored_terminations = 0
        with og.sim.render_on_step(False):
            for chunk_start, chunk_end in zip(
                chunk_boundaries[:-1], chunk_boundaries[1:], strict=True
            ):
                if chunk_start > 0:
                    self._align_robot_to_demo_frame(
                        demo_data_dir=demo_data_dir,
                        task_index=task_index,
                        episode_index=episode_index,
                        frame_index=chunk_start,
                    )
                replay_policy = b1k_eval.ParquetDemoReplayPolicy(
                    parquet_path=parquet_path,
                    start_frame=chunk_start,
                    end_frame=chunk_end,
                    subtask_index=0,
                    subtask_end_index=0,
                )
                replay_policy.reset()
                while not replay_policy.is_done:
                    terminated, truncated, _, info = self._fast_step_with_policy(
                        replay_policy
                    )
                    warmup_steps += 1
                    if not (terminated or truncated):
                        continue
                    ignored_terminations += 1
                    if not require_successful_terminations:
                        continue
                    if is_successful_predicate_termination(terminated, truncated, info):
                        continue
                    raise RuntimeError(
                        "Demo warmup ended for a reason other than successful task "
                        "completion before reaching the preparatory state "
                        f"(episode={episode_index}, "
                        f"target_start_frame={target_start_frame}, "
                        f"warmup_steps={warmup_steps}, terminated={terminated}, "
                        f"truncated={truncated}, info={info})"
                    )

        og.sim.render()
        self.obs = self._preprocess_obs(self.env.get_obs()[0])
        self.last_policy_done = False
        b1k_eval.logger.info(
            "Demo warmup replayed %d actions with %d intermediate robot "
            "alignments before frame %d.",
            warmup_steps,
            len(intermediate_boundaries),
            target_start_frame,
        )
        return warmup_steps, ignored_terminations

    def _activate_target_subtask(
        self,
        *,
        demo_data_dir: str,
        task_index: int,
        episode_index: int,
        start_frame: int,
        refresh_observation: bool,
    ) -> int:
        segment_index, _, end_frame = self._control_index.interval_at_start(
            episode_index, start_frame
        )
        task_reward = get_task_specific_reward(self)
        if not hasattr(task_reward, "set_active_stage_index"):
            raise TypeError(
                "Grounded skill-reset evaluation requires a task reward with "
                "set_active_stage_index()."
            )
        task_reward.set_active_stage_index(segment_index)
        self._target_segment_index = segment_index

        control = self._control_index.get(episode_index, segment_index)
        self._target_base_pose = None
        if control.skill == "move to":
            self._target_base_pose = self._load_demo_terminal_base_pose(
                demo_data_dir,
                task_index,
                episode_index,
                end_frame,
            )

        if refresh_observation:
            self.obs = self._preprocess_obs(self.env.get_obs()[0])
        b1k_eval.logger.info(
            "Activated direct subtask predicate: episode=%d segment=%d skill=%s",
            episode_index,
            segment_index,
            control.skill,
        )
        return segment_index

    def replay_demo_to_preparatory_state(
        self,
        demo_data_dir: str,
        task_index: int,
        episode_index: int,
        target_start_frame: int,
    ) -> dict:
        """Replay GT history, then make the selected skill the direct reward stage."""
        if target_start_frame <= 0:
            result = {
                "used_demo_warmup": False,
                "warmup_steps": 0,
                "warmup_end_frame": 0,
                "ignored_success_terminations": 0,
            }
        else:
            warmup_steps, ignored_success_terminations = self._replay_demo_warmup(
                demo_data_dir=demo_data_dir,
                task_index=task_index,
                episode_index=episode_index,
                target_start_frame=target_start_frame,
                require_successful_terminations=True,
            )
            if ignored_success_terminations:
                b1k_eval.logger.info(
                    "Demo warmup crossed %d successful whole-task termination "
                    "steps before frame %d.",
                    ignored_success_terminations,
                    target_start_frame,
                )
            result = {
                "used_demo_warmup": True,
                "warmup_steps": warmup_steps,
                "warmup_end_frame": target_start_frame,
                "ignored_success_terminations": ignored_success_terminations,
            }
        self._align_robot_to_demo_frame(
            demo_data_dir=demo_data_dir,
            task_index=task_index,
            episode_index=episode_index,
            frame_index=target_start_frame,
        )
        segment_index = self._activate_target_subtask(
            demo_data_dir=demo_data_dir,
            task_index=task_index,
            episode_index=episode_index,
            start_frame=target_start_frame,
            refresh_observation=True,
        )
        return {**result, "target_segment_index": segment_index}

    def _current_base_planar_pose(self) -> np.ndarray:
        position, quaternion = self.robot.get_position_orientation()
        euler = transform_utils.quat2euler(quaternion)
        return np.asarray(
            [float(position[0]), float(position[1]), float(euler[2])],
            dtype=np.float64,
        )

    def _apply_navigation_predicate(self, info: dict) -> None:
        if self._target_segment_index is None:
            return
        episode_index = episode_index_from_annotation_dir(
            self.cfg.orchestrators_annotation_dir
        )
        control = self._control_index.get(episode_index, self._target_segment_index)
        if control.skill != "move to":
            return
        if self._target_base_pose is None:
            raise RuntimeError(
                "The active move-to skill has no demo terminal base pose."
            )

        result = demo_terminal_pose_result(
            self._current_base_planar_pose(),
            self._target_base_pose,
            position_threshold=float(
                self.cfg.get("grounded_move_position_threshold", 0.5)
            ),
            yaw_threshold=math.radians(
                float(self.cfg.get("grounded_move_yaw_threshold_deg", 45.0))
            ),
        )
        reward_info = info["reward"]["task_specific"]
        stage_infos = reward_info["stage_infos"]
        stage_name = list(stage_infos)[self._target_segment_index]
        stage_infos[stage_name].update(result)
        stage_infos[stage_name]["success_source"] = "demo_terminal_base_pose"

    def step(self):
        """Apply the calibrated navigation predicate to the official step info."""
        terminated, truncated, reward, info = super().step()
        self._apply_navigation_predicate(info)
        return terminated, truncated, reward, info

    def calibrate_demo_segment(
        self,
        *,
        demo_data_dir: str,
        task_index: int,
        episode_index: int,
        start_frame: int,
        end_frame: int,
    ) -> tuple[bool, dict, dict]:
        """Replay one GT segment and evaluate only its own completion predicate."""
        parquet_path = (
            Path(demo_data_dir)
            / "data"
            / f"task-{task_index:04d}"
            / f"episode_{episode_index:08d}.parquet"
        )
        if start_frame > 0:
            _, ignored_terminations = self._replay_demo_warmup(
                demo_data_dir=demo_data_dir,
                task_index=task_index,
                episode_index=episode_index,
                target_start_frame=start_frame,
                require_successful_terminations=False,
            )
            if ignored_terminations:
                b1k_eval.logger.info(
                    "Calibration warmup crossed %d whole-task termination steps "
                    "before frame %d.",
                    ignored_terminations,
                    start_frame,
                )
        self._align_robot_to_demo_frame(
            demo_data_dir=demo_data_dir,
            task_index=task_index,
            episode_index=episode_index,
            frame_index=start_frame,
        )
        segment_index = self._activate_target_subtask(
            demo_data_dir=demo_data_dir,
            task_index=task_index,
            episode_index=episode_index,
            start_frame=start_frame,
            refresh_observation=False,
        )
        replay_policy = b1k_eval.ParquetDemoReplayPolicy(
            parquet_path=parquet_path,
            start_frame=start_frame,
            end_frame=end_frame,
            subtask_index=segment_index,
            subtask_end_index=segment_index,
        )
        replay_policy.reset()
        info = None
        ignored_terminations = 0
        while not replay_policy.is_done:
            terminated, truncated, _, info = self._fast_step_with_policy(replay_policy)
            ignored_terminations += int(terminated or truncated)
        if ignored_terminations:
            b1k_eval.logger.info(
                "Calibration target replay crossed %d whole-task termination steps "
                "for episode=%d segment=%d.",
                ignored_terminations,
                episode_index,
                segment_index,
            )
        if info is None:
            raise RuntimeError(
                f"Demo segment {episode_index}/{segment_index} executed no actions."
            )
        self._apply_navigation_predicate(info)
        return b1k_eval.get_reward_stage_result(
            info=info,
            target_stage_idx=segment_index,
        )

    def evaluate_segment_from_serialized_state(
        self,
        *,
        state: dict,
        demo_data_dir: str,
        task_index: int,
        episode_index: int,
        start_frame: int,
        end_frame: int,
    ) -> tuple[bool, dict, dict]:
        """Restore a snapshot and require its GT suffix to pass the predicate."""
        og.sim.load_state(state, serialized=False)
        # Controller goals are not part of the simulator snapshot. Reset them
        # to the restored articulation before advancing physics; otherwise the
        # first step can pull the robot toward the preceding reset pose.
        sync_robot_after_pose_override(self.robot)
        og.sim.step()
        segment_index = self._activate_target_subtask(
            demo_data_dir=demo_data_dir,
            task_index=task_index,
            episode_index=episode_index,
            start_frame=start_frame,
            refresh_observation=True,
        )
        parquet_path = (
            Path(demo_data_dir)
            / "data"
            / f"task-{task_index:04d}"
            / f"episode_{episode_index:08d}.parquet"
        )
        replay_policy = b1k_eval.ParquetDemoReplayPolicy(
            parquet_path=parquet_path,
            start_frame=start_frame,
            end_frame=end_frame,
            subtask_index=segment_index,
            subtask_end_index=segment_index,
        )
        replay_policy.reset()
        info = None
        while not replay_policy.is_done:
            _, _, _, info = self._fast_step_with_policy(replay_policy)
        if info is None:
            raise RuntimeError("Snapshot validation executed no GT actions.")
        self._apply_navigation_predicate(info)
        return b1k_eval.get_reward_stage_result(
            info=info,
            target_stage_idx=segment_index,
        )

    def _current_control(self, observation: dict) -> GroundedControlSpec:
        episode_index = episode_index_from_annotation_dir(
            self.cfg.orchestrators_annotation_dir
        )
        segment_index = self._active_segment_index()
        control = self._control_index.get(episode_index, segment_index)

        if self._control_profile is ControlProfile.P2_GROUND_SG:
            segmentations = {}
            for camera_name, camera_id in _CAMERA_IDS.items():
                sensor_name = ROBOT_CAMERA_NAMES["R1Pro"][camera_name]
                key = f"{sensor_name}::seg_instance_id"
                segmentation = observation.pop(key)
                segmentations[camera_id] = np.asarray(segmentation.cpu())
                observation.pop(f"{sensor_name}::seg_semantic", None)
            resolver = EntityResolver(VisionSensor.INSTANCE_ID_REGISTRY)
            control = ground_control_spec(
                control,
                segmentations,
                resolver,
                infer_missing_parts=self._infer_missing_parts,
            )

        stage_key = (episode_index, segment_index)
        if stage_key != self._last_logged_stage:
            b1k_eval.logger.info(
                "Oracle condition: episode=%d segment=%d profile=%s skill=%s",
                episode_index,
                segment_index,
                self._control_profile.value,
                control.skill,
            )
            self._last_logged_stage = stage_key
        return control

    def _preprocess_obs(self, obs: dict) -> dict:
        observation = super()._preprocess_obs(obs)
        if self._control_profile is not ControlProfile.P0_DIRECT:
            observation[GROUND_CONTROL_JSON_KEY] = self._current_control(
                observation
            ).to_json()
        return observation


def _run_demo_calibration(config) -> None:
    """Require every selected GT segment to satisfy its direct predicate."""
    task_index = b1k_eval.TASK_NAMES_TO_INDICES[config.task.name]
    episode_indices = b1k_eval.resolve_episode_indices(
        config.demo_data_dir,
        task_index,
        run_episode_idx=config.run_episode_idx,
        run_episode_indices=config.get("run_episode_indices"),
    )
    metrics_dir = Path(config.log_path).expanduser() / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    config["orchestrators_annotation_dir"] = (
        Path(config.demo_data_dir)
        / "orchestrators"
        / f"task-{task_index:04d}"
        / f"episode_{episode_indices[0]:08d}"
    )

    results = []
    with GroundedSubtaskEvaluator(config) as evaluator:
        for episode_index in episode_indices:
            annotation_dir = (
                Path(config.demo_data_dir)
                / "orchestrators"
                / f"task-{task_index:04d}"
                / f"episode_{episode_index:08d}"
            )
            evaluator.update_orchestrators_annotation_dir(annotation_dir)
            targets = b1k_eval.build_subtask_eval_targets(
                annotation_dir,
                subtask_skill=config.get("subtask_skill"),
                subtask_index=config.subtask_index,
                subtask_end_index=config.subtask_end_index,
            )
            instance_id = int((episode_index // 10) % 1e3)
            for target in targets:
                if target["subtask_start_idx"] != target["subtask_end_idx"]:
                    raise ValueError("Demo calibration requires one skill per target.")
                segment_index = target["subtask_start_idx"]
                _, subtask_info = target["selected_subtask_infos"][0]
                start_frame = int(subtask_info["start_frame"])
                end_frame = int(subtask_info["end_frame"])

                evaluator.reset()
                evaluator.load_task_instance(instance_id)
                evaluator.reset()
                success, stage_info, reward_info = evaluator.calibrate_demo_segment(
                    demo_data_dir=config.demo_data_dir,
                    task_index=task_index,
                    episode_index=episode_index,
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
                result = {
                    "episode_index": episode_index,
                    "segment_index": segment_index,
                    "skill": subtask_info["skill_description"],
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "success": success,
                    "stage_info": stage_info,
                    "current_stage_name": reward_info.get("current_stage_name"),
                }
                results.append(result)
                b1k_eval.logger.info(
                    "GT predicate calibration: episode=%d segment=%d skill=%s success=%s",
                    episode_index,
                    segment_index,
                    result["skill"],
                    success,
                )

        report = {
            "task": config.task.name,
            "all_passed": all(result["success"] for result in results),
            "results": results,
        }
        report_path = (
            metrics_dir / f"{config.task.name}_demo_predicate_calibration.json"
        )
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if not report["all_passed"]:
            failures = [
                f"ep{result['episode_index']}/st{result['segment_index']}:{result['skill']}"
                for result in results
                if not result["success"]
            ]
            raise RuntimeError(
                "Ground-truth demo failed direct subtask predicates: "
                + ", ".join(failures)
            )


def _run_subpool_export(config) -> None:
    """Replay demos, dump canonical states, and verify every restored suffix."""
    validate_subpool_export_request(
        instance_reward_mode=config.instance_reward_mode,
        run_episode_idx=config.run_episode_idx,
        run_episode_indices=config.get("run_episode_indices"),
    )
    manifest_path = Path(config.subpool_export_manifest).expanduser().resolve()
    if manifest_path.exists() and manifest_path.stat().st_size:
        raise FileExistsError(
            f"Refusing to append canonical snapshots to non-empty {manifest_path}."
        )
    reward_specs = json.loads(
        Path(config.subpool_reward_specs).read_text(encoding="utf-8")
    )
    token_mapping = ReservedTokenMapping.from_dict(
        json.loads(Path(config.subpool_token_mapping).read_text(encoding="utf-8"))
    )
    serializer = ControlSerializer(token_mapping)
    store = SubpoolStore(manifest_path)
    task_index = b1k_eval.TASK_NAMES_TO_INDICES[config.task.name]
    episode_indices = b1k_eval.resolve_episode_indices(
        config.demo_data_dir,
        task_index,
        run_episode_idx=None,
        run_episode_indices=config.get("run_episode_indices"),
    )
    b1k_eval.logger.info("Resolved explicit subpool episodes: %s", episode_indices)
    with open_dict(config):
        config["grounded_demo_calibration"] = True
        config["orchestrators_annotation_dir"] = str(
            Path(config.demo_data_dir)
            / "orchestrators"
            / f"task-{task_index:04d}"
            / f"episode_{episode_indices[0]:08d}"
        )

    with GroundedSubtaskEvaluator(config) as evaluator:
        for episode_index in episode_indices:
            annotation_dir = (
                Path(config.demo_data_dir)
                / "orchestrators"
                / f"task-{task_index:04d}"
                / f"episode_{episode_index:08d}"
            )
            evaluator.update_orchestrators_annotation_dir(annotation_dir)
            targets = b1k_eval.build_subtask_eval_targets(
                annotation_dir,
                subtask_skill=config.get("subtask_skill"),
                subtask_index=config.subtask_index,
                subtask_end_index=config.subtask_end_index,
            )
            instance_id = int((episode_index // 10) % 1e3)
            for target in targets:
                if target["subtask_start_idx"] != target["subtask_end_idx"]:
                    raise ValueError("Subpool export requires one skill per target.")
                segment_index = target["subtask_start_idx"]
                _, subtask_info = target["selected_subtask_infos"][0]
                start_frame = int(subtask_info["start_frame"])
                end_frame = int(subtask_info["end_frame"])
                skill = str(subtask_info["skill_description"])
                if skill not in reward_specs:
                    raise KeyError(
                        f"No reward specification for skill {skill!r} in "
                        f"{config.subpool_reward_specs}."
                    )
                validate_demo_horizon(
                    SubtaskRewardSpec.from_mapping(reward_specs[skill]),
                    start_frame=start_frame,
                    end_frame=end_frame,
                )

                evaluator.reset()
                evaluator.load_task_instance(instance_id)
                evaluator.reset()
                evaluator.replay_demo_to_preparatory_state(
                    demo_data_dir=config.demo_data_dir,
                    task_index=task_index,
                    episode_index=episode_index,
                    target_start_frame=start_frame,
                )
                # The flat OmniGibson serialization omits assisted-grasp
                # constraints.  Press / place snapshots after pickup therefore
                # require the complete nested state.
                state = og.sim.dump_state(serialized=False)
                control = evaluator._control_index.get(  # noqa: SLF001
                    episode_index, segment_index
                )

                evaluator.reset()
                evaluator.load_task_instance(instance_id)
                evaluator.reset()
                success, stage_info, _ = (
                    evaluator.evaluate_segment_from_serialized_state(
                        state=state,
                        demo_data_dir=config.demo_data_dir,
                        task_index=task_index,
                        episode_index=episode_index,
                        start_frame=start_frame,
                        end_frame=end_frame,
                    )
                )
                if not success:
                    raise RuntimeError(
                        "Restored canonical snapshot failed its GT suffix: "
                        f"episode={episode_index}, segment={segment_index}, "
                        f"skill={skill}, stage_info={stage_info}."
                    )

                snapshot_id = (
                    f"canonical-task{task_index:04d}-ep{episode_index:08d}-"
                    f"st{segment_index:02d}"
                )
                base_env = evaluator.env
                while hasattr(base_env, "env"):
                    base_env = base_env.env
                metadata = {
                    "reward": reward_specs[skill],
                    "instance_id": instance_id,
                    "gt_validation": {
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "stage_info": stage_info,
                    },
                }
                if evaluator._target_base_pose is not None:  # noqa: SLF001
                    metadata["target_base_pose"] = evaluator._target_base_pose.tolist()  # noqa: SLF001
                record = SubpoolSnapshot(
                    snapshot_id=snapshot_id,
                    state_path=f"states/{snapshot_id}.pt",
                    state_sha256=full_state_sha256(state),
                    activity_name=config.task.name,
                    scene_model=str(base_env.scene.scene_model),
                    asset_fingerprint=str(config.subpool_asset_fingerprint),
                    subtask_id=segment_index,
                    skill=skill,
                    pool_type="canonical",
                    task_description=serializer.serialize(
                        control, ControlProfile.P2_GROUND_SG
                    ),
                    control_json=control.to_json(),
                    episode_index=episode_index,
                    frame_index=start_frame,
                    metadata=metadata,
                )
                store.append(record, state)
                b1k_eval.logger.info(
                    "Exported validated canonical snapshot %s", snapshot_id
                )


def main() -> None:
    """Compose the official Hydra config and run its subtask loop."""
    register_omegaconf_resolvers()
    hydra_args, kit_args = split_hydra_and_kit_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *kit_args]
    config_dir = Path(b1k_eval.__file__).resolve().parent / "configs"
    with hydra.initialize_config_dir(str(config_dir), version_base="1.1"):
        config = hydra.compose("base_config.yaml", overrides=hydra_args)
    OmegaConf.resolve(config)
    gm.HEADLESS = config.headless
    if config.eval_level != "subtask":
        raise ValueError("Grounded evaluation currently requires eval_level=subtask.")
    if config.keep_running_after_success:
        raise ValueError(
            "Grounded direct-predicate evaluation requires "
            "keep_running_after_success=false so a completed skill terminates "
            "before the policy can undo it."
        )
    if (
        config.subtask_index is not None
        and config.subtask_end_index is not None
        and config.subtask_index != config.subtask_end_index
    ):
        raise ValueError(
            "Grounded direct-predicate evaluation currently requires one skill per "
            "target; subtask_index and subtask_end_index must match."
        )

    source_demo_data_dir = config.demo_data_dir
    config.demo_data_dir = str(
        prepare_subtask_evaluation_view(
            source_demo_data_dir,
            config.grounded_control_sidecar,
            config.grounded_eval_view_dir,
            task_index=b1k_eval.TASK_NAMES_TO_INDICES[config.task.name],
            task_name=config.task.name,
        )
    )

    if config.get("subpool_export_manifest"):
        _run_subpool_export(config)
    elif config.get("grounded_demo_calibration", False):
        _run_demo_calibration(config)
    else:
        b1k_eval.Evaluator = GroundedSubtaskEvaluator
        b1k_eval._run_subtask_eval(config, b1k_eval.logger)  # noqa: SLF001


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
