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

import logging
import os
import queue
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

from omegaconf.dictconfig import DictConfig

from rlinf.algorithms.subtask import outcome_group_is_trainable
from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.checkpoint import parse_global_step_from_checkpoint_path
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics, print_metrics_table
from rlinf.utils.runner_utils import check_progress
from rlinf.utils.timers import Timer

logger = logging.getLogger(__name__)


def _validate_outcome_reset_metadata(
    metadata_shards: list[list[dict]],
    *,
    expected_group_ids: set[int],
    group_size: int,
) -> dict[int, dict]:
    """Verify that every DAPO group was reset to one identical snapshot."""
    grouped_metadata: dict[int, list[dict]] = defaultdict(list)
    for worker_metadata in metadata_shards:
        if not isinstance(worker_metadata, list):
            raise RuntimeError(
                "Outcome reset metadata must be a list from every EnvWorker."
            )
        for metadata in worker_metadata:
            grouped_metadata[int(metadata["outcome_group_id"])].append(metadata)

    if set(grouped_metadata) != expected_group_ids:
        raise RuntimeError(
            "Outcome reset expected group IDs "
            f"{sorted(expected_group_ids)}, received {sorted(grouped_metadata)}."
        )

    synchronized = {}
    signature_keys = ("snapshot_id", "episode_index", "subtask_id", "pool_type")
    for group_id, group_metadata in grouped_metadata.items():
        if len(group_metadata) != group_size:
            raise RuntimeError(
                f"Outcome reset group {group_id} expected {group_size} members, "
                f"received {len(group_metadata)}."
            )
        signatures = {
            tuple(metadata[key] for key in signature_keys)
            for metadata in group_metadata
        }
        if len(signatures) != 1:
            raise RuntimeError(
                f"Outcome reset group {group_id} loaded different snapshots: "
                f"{sorted(signatures)}."
            )
        synchronized[group_id] = group_metadata[0]
    return synchronized


def _record_outcome_snapshot_stats(
    stats: dict[tuple[str, int], dict[str, int]],
    metadata: dict,
    *,
    successes: int,
    failures: int,
    accepted: bool,
) -> None:
    key = (str(metadata["snapshot_id"]), int(metadata["episode_index"]))
    state_stats = stats.setdefault(
        key,
        {
            "candidate_groups": 0,
            "candidate_successes": 0,
            "candidate_failures": 0,
            "accepted_groups": 0,
        },
    )
    state_stats["candidate_groups"] += 1
    state_stats["candidate_successes"] += successes
    state_stats["candidate_failures"] += failures
    state_stats["accepted_groups"] += int(accepted)


def _outcome_snapshot_metrics(
    stats: dict[tuple[str, int], dict[str, int]],
) -> dict[str, float | int]:
    metrics = {}
    for (snapshot_id, episode_index), state_stats in stats.items():
        safe_snapshot_id = snapshot_id.replace("/", "_")
        prefix = f"dynamic_sampling/snapshot/{safe_snapshot_id}"
        candidate_count = (
            state_stats["candidate_successes"] + state_stats["candidate_failures"]
        )
        metrics.update(
            {
                f"{prefix}/episode_index": episode_index,
                f"{prefix}/candidate_groups": state_stats["candidate_groups"],
                f"{prefix}/candidate_successes": state_stats["candidate_successes"],
                f"{prefix}/candidate_failures": state_stats["candidate_failures"],
                f"{prefix}/candidate_success_rate": (
                    state_stats["candidate_successes"] / candidate_count
                ),
                f"{prefix}/accepted_groups": state_stats["accepted_groups"],
            }
        )
    return metrics


if TYPE_CHECKING:
    from rlinf.workers.actor.async_fsdp_sac_policy_worker import (
        AsyncEmbodiedSACFSDPPolicy,
    )
    from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor
    from rlinf.workers.actor.fsdp_nft_policy_worker import EmbodiedNFTFSDPPolicy
    from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy
    from rlinf.workers.env.async_env_worker import AsyncEnvWorker
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker
    from rlinf.workers.rollout.hf.async_huggingface_worker import (
        AsyncMultiStepRolloutWorker,
    )
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


@dataclass(frozen=True)
class _AcceptedOutcomeEnvHandle:
    """Expose metrics only for outcome groups retained from a parallel round."""

    handle: Handle
    group_ids: frozenset[int]
    group_size: int

    def wait(self):
        results = self.handle.wait()
        return [
            result if rank // self.group_size in self.group_ids else None
            for rank, result in enumerate(results)
        ]

    def consume_durations(self, *args, **kwargs):
        return self.handle.consume_durations(*args, **kwargs)


class EmbodiedRunner:
    def __init__(
        self,
        cfg: DictConfig,
        actor: Union[
            "EmbodiedFSDPActor",
            "EmbodiedNFTFSDPPolicy",
            "EmbodiedSACFSDPPolicy",
            "AsyncEmbodiedSACFSDPPolicy",
        ],
        rollout: Union["MultiStepRolloutWorker", "AsyncMultiStepRolloutWorker"],
        env: Union["EnvWorker", "AsyncEnvWorker"],
        reward: Union["EmbodiedRewardWorker"] = None,
        critic=None,
    ):
        self.cfg = cfg
        self.actor = actor
        self.rollout = rollout
        self.env = env
        self.critic = critic
        self.reward = reward
        self.weight_sync_interval = self.cfg.runner.weight_sync_interval
        self.overlap_env_bootstrap = bool(
            self.cfg.runner.get("overlap_env_bootstrap", False)
        )

        # Step-gated profiling: ``cluster.profiling.steps`` lists the global step
        profiling_raw = self.cfg.cluster.get("profiling", None)
        profiling_enabled = profiling_raw is not None and bool(
            profiling_raw.get("enabled", True)
        )
        profile_steps_raw = (
            profiling_raw.get("steps", None) if profiling_enabled else None
        )
        self._profile_all_steps = profiling_enabled and profile_steps_raw is None
        self._profile_steps: set[int] | None = (
            {int(s) for s in profile_steps_raw}
            if profile_steps_raw is not None
            else None
        )

        # Data channels
        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")
        self.actor_channel = Channel.create("Actor")
        if self.reward is not None:
            self.reward_channel = Channel.create("Reward")
        else:
            self.reward_channel = None

        # this timer checks if we should stop training
        self.run_timer = Timer(None)  # Timer that checks if we should stop training

        self.consumed_samples = 0
        # the step here is GRPO step
        self.global_step = 0
        self._outcome_collection_index = 0

        # compute `max_steps`
        self.set_max_steps()

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)

        self.logger = get_logger()
        self.metric_logger = MetricLogger(cfg)
        self.enable_per_worker_metric_log = bool(
            self.cfg.runner.get("per_worker_log", False)
        )

        # Async logging setup
        self.stop_logging = False
        self.log_queue = queue.Queue()
        self.log_thread = threading.Thread(target=self._log_worker, daemon=True)
        self.log_thread.start()

    def _log_worker(self):
        """Background thread for processing log messages."""
        while not self.stop_logging:
            try:
                # Wait for log message with timeout
                log_func, args = self.log_queue.get(timeout=0.1)
                log_func(*args)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Logging error: {e}")
                continue

    def print_metrics_table_async(
        self,
        step: int,
        total_steps: int,
        start_time: float,
        metrics: dict,
        start_step: int = 0,
    ):
        """Async version that puts table printing in queue."""
        self.log_queue.put(
            (
                print_metrics_table,
                (
                    step,
                    total_steps,
                    start_time,
                    metrics,
                    start_step,
                    self.metric_logger.log_path,
                ),
            )
        )

    def init_workers(self):
        # create worker in order to decrease the maximum memory usage
        rollout_handle = self.rollout.init_worker()
        env_handle = self.env.init_worker()

        if self.reward is not None:
            self.reward.init_worker().wait()

        rollout_handle.wait()
        env_handle.wait()
        self.actor.init_worker().wait()

        resume_dir = self.cfg.runner.get("resume_dir", None)
        if resume_dir is None:
            return

        self.logger.info(f"Resuming training from checkpoint directory {resume_dir}.")
        self.global_step = parse_global_step_from_checkpoint_path(resume_dir)
        actor_checkpoint_path = os.path.join(resume_dir, "actor")
        assert os.path.exists(actor_checkpoint_path), (
            f"resume_dir {actor_checkpoint_path} does not exist."
        )
        self.actor.load_checkpoint(actor_checkpoint_path).wait()

    def update_rollout_weights(self):
        rollout_handle: Handle = self.rollout.sync_model_from_actor()
        actor_handle: Handle = self.actor.sync_model_to_rollout()
        actor_handle.wait()
        rollout_handle.wait()

    def _collect_train_rollout(self):
        env_handle: Handle = self.env.interact(
            input_channel=self.env_channel,
            rollout_channel=self.rollout_channel,
            reward_channel=self.reward_channel,
            actor_channel=self.actor_channel,
        )
        rollout_handle: Handle = self.rollout.generate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        reward_handle = None
        if self.reward is not None:
            reward_handle = self.reward.compute_rewards(
                input_channel=self.reward_channel,
                output_channel=self.env_channel,
            )
        outcome_shards = self.actor.recv_rollout_trajectories(
            input_channel=self.actor_channel
        ).wait()
        rollout_handle.wait()
        if reward_handle is not None:
            reward_handle.wait()
        return env_handle, rollout_handle, reward_handle, outcome_shards

    def _reset_outcome_sampling_envs(self, sampling_cfg: DictConfig) -> dict[int, dict]:
        """Reset and verify all physical groups before one candidate rollout."""
        collection_index = getattr(self, "_outcome_collection_index", 0)
        metadata_shards = self.env.reset_train_envs_for_outcome_group(
            collection_index
        ).wait()
        self._outcome_collection_index = collection_index + 1
        groups_per_update = int(sampling_cfg.get("groups_per_update", 1))
        expected_group_ids = (
            set(range(groups_per_update))
            if sampling_cfg.get("parallel_groups", False)
            else {0}
        )
        synchronized = _validate_outcome_reset_metadata(
            metadata_shards,
            expected_group_ids=expected_group_ids,
            group_size=int(sampling_cfg.group_size),
        )
        summary = ", ".join(
            f"group={group_id} snapshot={metadata['snapshot_id']} "
            f"episode={metadata['episode_index']}"
            for group_id, metadata in sorted(synchronized.items())
        )
        self.logger.info(
            "Synchronized outcome reset collection=%d: %s.",
            collection_index,
            summary,
        )
        return synchronized

    def _collect_trainable_rollout(self):
        """Collect independently quota-filtered groups for one on-policy update."""
        sampling_cfg = self.cfg.algorithm.get("outcome_dynamic_sampling", {})
        sampling_enabled = bool(sampling_cfg.get("enabled", False))
        if not sampling_enabled:
            handles = self._collect_train_rollout()
            return ([handles[0]], [handles[1]], [handles[2]], {})
        if sampling_cfg.get("parallel_groups", False):
            return self._collect_parallel_trainable_rollout(sampling_cfg)

        groups_per_update = int(sampling_cfg.get("groups_per_update", 1))
        warning_interval = int(
            sampling_cfg.get(
                "attempt_warning_interval",
                sampling_cfg.get("max_rollout_attempts", 0),
            )
        )
        env_handles = []
        rollout_handles = []
        reward_handles = []
        total_attempts = 0
        accepted_successes = 0
        accepted_failures = 0
        candidate_successes = 0
        candidate_failures = 0
        snapshot_stats: dict[tuple[str, int], dict[str, int]] = {}

        self.actor.begin_rollout_group_collection().wait()
        for group_index in range(groups_per_update):
            group_attempt = 0
            while True:
                group_attempt += 1
                reset_metadata = self._reset_outcome_sampling_envs(sampling_cfg)[0]
                handles = self._collect_train_rollout()
                env_handle, rollout_handle, reward_handle, outcome_shards = handles
                total_attempts += 1
                if any(shard is None for shard in outcome_shards):
                    raise RuntimeError(
                        "Outcome dynamic sampling requires every actor rank to "
                        "receive explicit rollout success outcomes."
                    )
                accepted, successes, failures = outcome_group_is_trainable(
                    outcome_shards,
                    expected_size=int(sampling_cfg.group_size),
                    min_successes=int(sampling_cfg.min_successes),
                    min_failures=int(sampling_cfg.min_failures),
                )
                candidate_successes += successes
                candidate_failures += failures
                _record_outcome_snapshot_stats(
                    snapshot_stats,
                    reset_metadata,
                    successes=successes,
                    failures=failures,
                    accepted=accepted,
                )
                if accepted:
                    self.actor.accept_rollout_group().wait()
                    env_handles.append(env_handle)
                    rollout_handles.append(rollout_handle)
                    reward_handles.append(reward_handle)
                    accepted_successes += successes
                    accepted_failures += failures
                    break

                log_rejection = (
                    self.logger.warning
                    if group_attempt % warning_interval == 0
                    else self.logger.info
                )
                log_rejection(
                    "Rejected outcome-homogeneous rollout group "
                    "(group %d/%d, attempt=%d, successes=%d, failures=%d); "
                    "continuing sampling.",
                    group_index + 1,
                    groups_per_update,
                    group_attempt,
                    successes,
                    failures,
                )
                # The next attempt replaces this handle. Consume its completed
                # result so rejected rollout metrics do not occupy Ray object-store
                # memory for the rest of the update.
                env_handle.wait()

        self.actor.finalize_rollout_group_collection(groups_per_update).wait()
        return (
            env_handles,
            rollout_handles,
            reward_handles,
            {
                "dynamic_sampling/groups_per_update": groups_per_update,
                "dynamic_sampling/attempts": total_attempts,
                "dynamic_sampling/rejected_groups": (
                    total_attempts - groups_per_update
                ),
                "dynamic_sampling/successes": accepted_successes,
                "dynamic_sampling/failures": accepted_failures,
                "dynamic_sampling/candidate_successes": candidate_successes,
                "dynamic_sampling/candidate_failures": candidate_failures,
                **_outcome_snapshot_metrics(snapshot_stats),
            },
        )

    def _collect_parallel_trainable_rollout(self, sampling_cfg: DictConfig):
        """Collect independent quota-filtered groups in concurrent env rounds."""
        groups_per_update = int(sampling_cfg.groups_per_update)
        group_size = int(sampling_cfg.group_size)
        warning_interval = int(
            sampling_cfg.get(
                "attempt_warning_interval",
                sampling_cfg.get("max_rollout_attempts", 0),
            )
        )
        pending_groups = set(range(groups_per_update))
        attempts_by_group = [0] * groups_per_update
        env_handles = []
        rollout_handles = []
        reward_handles = []
        accepted_successes = 0
        accepted_failures = 0
        candidate_successes = 0
        candidate_failures = 0
        sampling_rounds = 0
        snapshot_stats: dict[tuple[str, int], dict[str, int]] = {}

        self.actor.begin_rollout_group_collection().wait()
        while pending_groups:
            sampling_rounds += 1
            reset_metadata = self._reset_outcome_sampling_envs(sampling_cfg)
            env_handle, rollout_handle, reward_handle, outcome_shards = (
                self._collect_train_rollout()
            )
            if any(shard is None for shard in outcome_shards):
                raise RuntimeError(
                    "Parallel outcome sampling requires every actor rank to return "
                    "grouped rollout success outcomes."
                )

            grouped_shards: dict[int, list[list[bool]]] = defaultdict(list)
            for actor_shard in outcome_shards:
                if not isinstance(actor_shard, dict):
                    raise RuntimeError(
                        "Parallel outcome sampling requires actor outcomes keyed by "
                        "group ID."
                    )
                for group_id, outcomes in actor_shard.items():
                    grouped_shards[int(group_id)].append(outcomes)
            expected_group_ids = set(range(groups_per_update))
            if set(grouped_shards) != expected_group_ids:
                raise RuntimeError(
                    "Parallel outcome sampling expected group IDs "
                    f"{sorted(expected_group_ids)}, received "
                    f"{sorted(grouped_shards)}."
                )

            accepted_this_round = []
            for group_id in sorted(pending_groups):
                attempts_by_group[group_id] += 1
                accepted, successes, failures = outcome_group_is_trainable(
                    grouped_shards[group_id],
                    expected_size=group_size,
                    min_successes=int(sampling_cfg.min_successes),
                    min_failures=int(sampling_cfg.min_failures),
                )
                candidate_successes += successes
                candidate_failures += failures
                _record_outcome_snapshot_stats(
                    snapshot_stats,
                    reset_metadata[group_id],
                    successes=successes,
                    failures=failures,
                    accepted=accepted,
                )
                if accepted:
                    accepted_this_round.append(group_id)
                    accepted_successes += successes
                    accepted_failures += failures
                    continue

                group_attempt = attempts_by_group[group_id]
                log_rejection = (
                    self.logger.warning
                    if group_attempt % warning_interval == 0
                    else self.logger.info
                )
                log_rejection(
                    "Rejected outcome-homogeneous parallel rollout group "
                    "(group %d/%d, attempt=%d, successes=%d, failures=%d); "
                    "continuing sampling.",
                    group_id + 1,
                    groups_per_update,
                    group_attempt,
                    successes,
                    failures,
                )

            if accepted_this_round:
                self.actor.accept_rollout_groups(accepted_this_round).wait()
                pending_groups.difference_update(accepted_this_round)
                env_handles.append(
                    _AcceptedOutcomeEnvHandle(
                        handle=env_handle,
                        group_ids=frozenset(accepted_this_round),
                        group_size=group_size,
                    )
                )
                rollout_handles.append(rollout_handle)
                reward_handles.append(reward_handle)
            else:
                env_handle.wait()

        self.actor.finalize_rollout_group_collection(groups_per_update).wait()
        total_attempts = sum(attempts_by_group)
        return (
            env_handles,
            rollout_handles,
            reward_handles,
            {
                "dynamic_sampling/groups_per_update": groups_per_update,
                "dynamic_sampling/sampling_rounds": sampling_rounds,
                "dynamic_sampling/attempts": total_attempts,
                "dynamic_sampling/rejected_groups": (
                    total_attempts - groups_per_update
                ),
                "dynamic_sampling/successes": accepted_successes,
                "dynamic_sampling/failures": accepted_failures,
                "dynamic_sampling/candidate_successes": candidate_successes,
                "dynamic_sampling/candidate_failures": candidate_failures,
                **_outcome_snapshot_metrics(snapshot_stats),
            },
        )

    def evaluate(self):
        env_handle: Handle = self.env.evaluate(
            input_channel=self.env_channel,
            rollout_channel=self.rollout_channel,
        )
        rollout_handle: Handle = self.rollout.evaluate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        env_results = env_handle.wait()
        rollout_handle.wait()
        eval_metrics_list = [results for results in env_results if results is not None]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        return eval_metrics

    def _log_ranked_metrics(
        self,
        metrics_list: list[dict] | None,
        step: int,
        prefix: str,
        worker_group_name: str,
        add_prefix: bool = True,
    ):
        if not self.enable_per_worker_metric_log or not metrics_list:
            return
        for rank, metrics in enumerate(metrics_list):
            if not metrics:
                continue
            metrics_to_log = (
                {f"{prefix}/{k}": v for k, v in metrics.items()}
                if add_prefix
                else metrics
            )
            self.metric_logger.log(
                data=metrics_to_log,
                step=step,
                worker_group_name=worker_group_name,
                rank=rank,
            )

    def _aggregate_numeric_metrics(self, metrics_list: list[dict] | None) -> dict:
        if not metrics_list:
            return {}
        merged_metrics = defaultdict(list)
        for metrics in metrics_list:
            if not metrics:
                continue
            for key, value in metrics.items():
                merged_metrics[key].append(value)
        return {
            key: (sum(values) / len(values))
            for key, values in merged_metrics.items()
            if values
        }

    def _process_ranked_numeric_results(
        self, results: list[dict], metric_field: str
    ) -> tuple[dict, list[dict]]:
        metric_list: list[dict] = []
        per_rank_metrics: dict[int, list[dict]] = defaultdict(list)
        for result in results:
            metrics = result.get(metric_field, None)
            if not metrics:
                continue
            metric_list.append(metrics)
            rank = result.get("rank", None)
            if rank is not None:
                per_rank_metrics[int(rank)].append(metrics)

        aggregated_metrics = self._aggregate_numeric_metrics(metric_list)
        ranked_metrics_list: list[dict] = []
        if per_rank_metrics:
            max_rank = max(per_rank_metrics.keys())
            ranked_metrics_list = [{} for _ in range(max_rank + 1)]
            for rank, metrics_list in per_rank_metrics.items():
                ranked_metrics_list[rank] = self._aggregate_numeric_metrics(
                    metrics_list
                )
        return aggregated_metrics, ranked_metrics_list

    def _process_ranked_eval_results(
        self, results: list[dict], metric_field: str
    ) -> tuple[dict, list[dict]]:
        metric_list: list[dict] = []
        per_rank_metrics: dict[int, list[dict]] = defaultdict(list)
        for result in results:
            metrics = result.get(metric_field, None)
            if not metrics:
                continue
            metric_list.append(metrics)
            rank = result.get("rank", None)
            if rank is not None:
                per_rank_metrics[int(rank)].append(metrics)

        aggregated_metrics = (
            compute_evaluate_metrics(metric_list) if metric_list else {}
        )
        ranked_metrics_list: list[dict] = []
        if per_rank_metrics:
            max_rank = max(per_rank_metrics.keys())
            ranked_metrics_list = [{} for _ in range(max_rank + 1)]
            for rank, metrics_list in per_rank_metrics.items():
                ranked_metrics_list[rank] = compute_evaluate_metrics(metrics_list)
        return aggregated_metrics, ranked_metrics_list

    @staticmethod
    def _split_pipeline_actor_results(
        results: list[dict] | None,
    ) -> tuple[list[dict], list[dict]]:
        if not results:
            return [], []
        rollout_metrics = [result.get("rollout_metrics", {}) for result in results]
        training_metrics = [result.get("training_metrics", {}) for result in results]
        return rollout_metrics, training_metrics

    def _maybe_eval_and_checkpoint(self, step: int) -> dict:
        run_val, save_model, _ = check_progress(
            self.global_step,
            self.max_steps,
            self.cfg.runner.val_check_interval,
            self.cfg.runner.save_interval,
            1.0,
            run_time_exceeded=False,
        )

        eval_metrics = {}
        if run_val:
            with self.timer("eval"):
                self.update_rollout_weights()
                eval_metrics = self.evaluate()
                eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                self.metric_logger.log(data=eval_metrics, step=step)

        if save_model:
            self._save_checkpoint()

        return eval_metrics

    def _log_step_metrics(
        self,
        step: int,
        start_time: float,
        start_step: int,
        env_handles: list[Handle],
        rollout_handles: list[Handle],
        actor_training_handle: Handle,
        reward_handles: list[Handle | None],
        actor_rollout_metrics: list[dict],
        actor_training_metrics: list[dict],
        eval_metrics: dict,
    ) -> None:
        time_metrics = self.timer.consume_durations()
        time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
        env_time_metrics, env_time_metrics_per_rank = env_handles[-1].consume_durations(
            return_per_rank=True
        )
        rollout_time_metrics, rollout_time_metrics_per_rank = rollout_handles[
            -1
        ].consume_durations(return_per_rank=True)
        actor_time_metrics, actor_time_metrics_per_rank = (
            actor_training_handle.consume_durations(return_per_rank=True)
        )
        time_metrics.update({f"time/env/{k}": v for k, v in env_time_metrics.items()})
        time_metrics.update(
            {f"time/rollout/{k}": v for k, v in rollout_time_metrics.items()}
        )
        time_metrics.update(
            {f"time/actor/{k}": v for k, v in actor_time_metrics.items()}
        )
        if self.reward is not None:
            reward_handle = reward_handles[-1]
            assert reward_handle is not None
            reward_time_metrics, reward_time_metrics_per_rank = (
                reward_handle.consume_durations(return_per_rank=True)
            )
            time_metrics.update(
                {f"time/reward/{k}": v for k, v in reward_time_metrics.items()}
            )

        env_results_by_group = [handle.wait() for handle in env_handles]
        env_results_list = [
            result
            for group_results in env_results_by_group
            for result in group_results
            if result is not None
        ]
        env_metrics = compute_evaluate_metrics(env_results_list)
        env_metrics = {f"env/{k}": v for k, v in env_metrics.items()}
        ranked_env_results = [
            {"rank": rank, "env": rank_metrics}
            for group_results in env_results_by_group
            for rank, rank_metrics in enumerate(group_results)
            if rank_metrics is not None
        ]
        _, env_metrics_per_rank = self._process_ranked_eval_results(
            ranked_env_results, metric_field="env"
        )

        rollout_metrics = {
            f"rollout/{k}": v
            for k, v in self._aggregate_numeric_metrics(actor_rollout_metrics).items()
        }
        training_metrics = {
            f"train/{k}": v
            for k, v in self._aggregate_numeric_metrics(actor_training_metrics).items()
        }

        self.metric_logger.log(env_metrics, step)
        self.metric_logger.log(rollout_metrics, step)
        self.metric_logger.log(time_metrics, step)
        self.metric_logger.log(training_metrics, step)
        self._log_ranked_metrics(
            metrics_list=actor_rollout_metrics,
            step=step,
            prefix="rollout",
            worker_group_name=self.actor.worker_group_name,
        )
        self._log_ranked_metrics(
            metrics_list=actor_training_metrics,
            step=step,
            prefix="train",
            worker_group_name=self.actor.worker_group_name,
        )
        self._log_ranked_metrics(
            metrics_list=actor_time_metrics_per_rank,
            step=step,
            prefix="time/actor",
            worker_group_name=self.actor.worker_group_name,
        )
        self._log_ranked_metrics(
            metrics_list=rollout_time_metrics_per_rank,
            step=step,
            prefix="time/rollout",
            worker_group_name=self.rollout.worker_group_name,
        )
        self._log_ranked_metrics(
            metrics_list=env_time_metrics_per_rank,
            step=step,
            prefix="time/env",
            worker_group_name=self.env.worker_group_name,
        )
        self._log_ranked_metrics(
            metrics_list=env_metrics_per_rank,
            step=step,
            prefix="env",
            worker_group_name=self.env.worker_group_name,
        )
        if self.reward is not None:
            self._log_ranked_metrics(
                metrics_list=reward_time_metrics_per_rank,
                step=step,
                prefix="time/reward",
                worker_group_name=self.reward.worker_group_name,
            )

        logging_metrics = time_metrics
        logging_metrics.update(eval_metrics)
        logging_metrics.update(env_metrics)
        logging_metrics.update(rollout_metrics)
        logging_metrics.update(training_metrics)

        self.print_metrics_table_async(
            step, self.max_steps, start_time, logging_metrics, start_step
        )

    def _finish_run(self) -> None:
        self.metric_logger.finish()

        # Stop logging thread
        self.stop_logging = True
        self.log_queue.join()  # Wait for all queued logs to be processed
        self.log_thread.join(timeout=1.0)

    def _should_profile_step(self, step_idx: int) -> bool:
        return self._profile_all_steps or (
            self._profile_steps is not None and step_idx in self._profile_steps
        )

    def _open_profiling_window(self, step_idx: int) -> None:
        """Dispatch ``start_profile`` to all compute worker groups for this step."""
        self.logger.info(f"Opening profiling window at step {step_idx}")
        self.actor.start_profile(step_idx).wait()
        self.rollout.start_profile(step_idx).wait()
        self.env.start_profile(step_idx).wait()

    def _close_profiling_window(self, step_idx: int) -> None:
        """Dispatch ``stop_profile`` to all compute worker groups."""
        self.actor.stop_profile().wait()
        self.rollout.stop_profile().wait()
        self.env.stop_profile().wait()
        self.logger.info(f"Closed profiling window at step {step_idx}")

    def run(self):
        if self.cfg.runner.get("use_training_pipeline", False):
            return self.run_pipeline()

        start_step = self.global_step
        start_time = time.time()
        for _step in range(start_step, self.max_steps):
            # set global step
            self.actor.set_global_step(self.global_step).wait()
            self.rollout.set_global_step(self.global_step).wait()
            self.env.set_global_step(self.global_step).wait()

            profiled_step = (
                self.global_step
                if self._should_profile_step(self.global_step)
                else None
            )
            if profiled_step is not None:
                self._open_profiling_window(profiled_step)

            with self.timer("step", trace_args={"step_idx": _step}):
                with self.timer("sync_weights"):
                    if _step % self.weight_sync_interval == 0:
                        self.update_rollout_weights()
                with self.timer("generate_rollouts"):
                    (
                        env_handles,
                        rollout_handles,
                        reward_handles,
                        dynamic_sampling_metrics,
                    ) = self._collect_trainable_rollout()

                # compute advantages and returns.
                with self.timer("cal_adv_and_returns"):
                    actor_rollout_metrics = (
                        self.actor.compute_advantages_and_returns().wait()
                    )
                    if dynamic_sampling_metrics:
                        for metrics in actor_rollout_metrics:
                            metrics.update(dynamic_sampling_metrics)

                # actor training.
                with self.timer("actor_training"):
                    actor_training_handle: Handle = self.actor.run_training()
                    env_bootstrap_handle: Handle | None = None
                    if self.overlap_env_bootstrap and _step + 1 < self.max_steps:
                        env_bootstrap_handle = self.env.prefetch_train_bootstrap(
                            rollout_channel=self.rollout_channel
                        )

                    actor_training_metrics = actor_training_handle.wait()
                    if env_bootstrap_handle is not None:
                        env_bootstrap_handle.wait()

                self.global_step += 1
                eval_metrics = self._maybe_eval_and_checkpoint(_step)

            if profiled_step is not None:
                self._close_profiling_window(profiled_step)

            self._log_step_metrics(
                step=_step,
                start_time=start_time,
                start_step=start_step,
                env_handles=env_handles,
                rollout_handles=rollout_handles,
                actor_training_handle=actor_training_handle,
                reward_handles=reward_handles,
                actor_rollout_metrics=actor_rollout_metrics,
                actor_training_metrics=actor_training_metrics,
                eval_metrics=eval_metrics,
            )

        self._finish_run()

    def run_pipeline(self):
        start_step = self.global_step
        start_time = time.time()
        for _step in range(start_step, self.max_steps):
            # set global step
            self.actor.set_global_step(self.global_step).wait()
            self.rollout.set_global_step(self.global_step).wait()
            self.env.set_global_step(self.global_step).wait()

            profiled_step = (
                self.global_step
                if self._should_profile_step(self.global_step)
                else None
            )
            if profiled_step is not None:
                self._open_profiling_window(profiled_step)

            with self.timer("step", trace_args={"step_idx": _step}):
                with self.timer("sync_weights"):
                    if _step % self.weight_sync_interval == 0:
                        self.update_rollout_weights()
                env_handle: Handle = self.env.interact(
                    input_channel=self.env_channel,
                    rollout_channel=self.rollout_channel,
                    reward_channel=self.reward_channel,
                    actor_channel=self.actor_channel,
                )
                rollout_handle: Handle = self.rollout.generate(
                    input_channel=self.rollout_channel,
                    output_channel=self.env_channel,
                )
                reward_handle = None
                if self.reward is not None:
                    reward_handle: Handle = self.reward.compute_rewards(
                        input_channel=self.reward_channel,
                        output_channel=self.env_channel,
                    )
                # actor training.
                actor_training_handle: Handle = self.actor.run_training(
                    input_channel=self.actor_channel
                )
                with self.timer("generate_rollouts"):
                    rollout_handle.wait()
                    if self.reward is not None:
                        reward_handle.wait()

                env_bootstrap_handle: Handle | None = None
                if self.overlap_env_bootstrap and _step + 1 < self.max_steps:
                    env_bootstrap_handle = self.env.prefetch_train_bootstrap(
                        rollout_channel=self.rollout_channel
                    )

                actor_results = actor_training_handle.wait()
                actor_rollout_metrics, actor_training_metrics = (
                    self._split_pipeline_actor_results(actor_results)
                )
                if env_bootstrap_handle is not None:
                    env_bootstrap_handle.wait()

                self.global_step += 1
                eval_metrics = self._maybe_eval_and_checkpoint(_step)

            if profiled_step is not None:
                self._close_profiling_window(profiled_step)

            self._log_step_metrics(
                step=_step,
                start_time=start_time,
                start_step=start_step,
                env_handles=[env_handle],
                rollout_handles=[rollout_handle],
                actor_training_handle=actor_training_handle,
                reward_handles=[reward_handle],
                actor_rollout_metrics=actor_rollout_metrics,
                actor_training_metrics=actor_training_metrics,
                eval_metrics=eval_metrics,
            )

        self._finish_run()

    def _save_checkpoint(self):
        self.logger.info(f"Saving checkpoint at step {self.global_step}.")
        base_output_dir = os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
            f"checkpoints/global_step_{self.global_step}",
        )
        actor_save_path = os.path.join(base_output_dir, "actor")
        os.makedirs(actor_save_path, exist_ok=True)
        self.actor.save_checkpoint(actor_save_path, self.global_step).wait()

    def set_max_steps(self):
        self.num_steps_per_epoch = 1
        self.max_steps = self.num_steps_per_epoch * self.cfg.runner.max_epochs

        if (max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    @property
    def epoch(self):
        return self.global_step // self.num_steps_per_epoch
