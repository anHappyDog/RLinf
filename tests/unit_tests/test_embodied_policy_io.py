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

import pickle
from types import SimpleNamespace

import numpy as np
import torch

from rlinf.algorithms.utils import preprocess_embodied_advantages_inputs
from rlinf.data.embodied_io_struct import (
    EnvOutput,
    EnvResult,
    PolicyInput,
    PolicyOutput,
    RolloutResult,
    TrajectoryEpochEnd,
    TrajectorySegment,
    merge_policy_inputs,
    split_policy_input,
)
from rlinf.scheduler.channel.channel import Channel
from rlinf.scheduler.channel.trajectory_channel import (
    TrajectoryChannel,
    TrajectoryWorker,
)
from rlinf.utils.metric_utils import compute_loss_mask
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def test_trajectory_channel_keeps_trajectory_worker_group_alive(monkeypatch):
    class FakeWork:
        def wait(self):
            return None

    class FakeGroup:
        def launch(self, **kwargs):
            self.launch_kwargs = kwargs
            return self

        def init_worker(self):
            self.initialized = True
            return FakeWork()

    channel = TrajectoryChannel()
    ingress_channel = Channel()
    ingress_channel._channel_worker_group = SimpleNamespace(
        _is_ready=lambda: FakeWork()
    )
    group = FakeGroup()
    created_channels = []

    def create_channel(cls, name, *args, **kwargs):
        created_channels.append((cls, name))
        return channel if name == "Actor" else ingress_channel

    monkeypatch.setattr(Channel, "create", classmethod(create_channel))
    monkeypatch.setattr(
        TrajectoryWorker,
        "create_group",
        classmethod(lambda cls, *args, **kwargs: group),
    )
    monkeypatch.setattr(
        "rlinf.scheduler.channel.trajectory_channel.Cluster",
        lambda: SimpleNamespace(num_nodes=1),
    )

    result = TrajectoryChannel.create("Actor", trajectory_cfg=object())

    assert result is channel
    assert channel._trajectory_ingress_channel is ingress_channel
    assert channel._trajectory_worker_group is group
    assert created_channels == [
        (TrajectoryChannel, "Actor"),
        (Channel, "Actor.segments"),
    ]
    assert group.launch_kwargs["name"] == "Actor.trajectory"
    assert group.initialized


def test_split_policy_input_preserves_env_result_fields():
    policy_input = PolicyInput(
        obs={
            "states": torch.arange(8, dtype=torch.float32).reshape(4, 2),
            "task_descriptions": ["a", "b", "c", "d"],
        },
        env_result=EnvResult(
            rewards=torch.arange(4, dtype=torch.float32).reshape(4, 1),
            dones=torch.tensor([[False], [True], [False], [True]]),
            final_obs={"states": torch.arange(8, dtype=torch.float32).reshape(4, 2)},
            intervene_flags=torch.tensor([[False], [True], [False], [False]]),
        ),
        is_last=True,
    )

    first, second = split_policy_input(policy_input, [1, 3])

    assert first.is_last and second.is_last
    assert first.obs["task_descriptions"] == ["a"]
    assert second.obs["task_descriptions"] == ["b", "c", "d"]
    assert torch.equal(first.env_result.rewards, torch.tensor([[0.0]]))
    assert torch.equal(second.env_result.rewards, torch.tensor([[1.0], [2.0], [3.0]]))
    assert first.env_result.final_obs["states"].shape == (1, 2)
    assert second.env_result.intervene_flags.shape == (3, 1)


def test_merge_policy_inputs_preserves_order_and_partial_results():
    first = PolicyInput(
        obs={"states": torch.tensor([[1.0], [2.0]])},
        env_result=EnvResult(
            rewards=torch.tensor([[0.5], [1.5]]),
            dones=torch.tensor([[False], [True]]),
            final_obs={"states": torch.tensor([[10.0], [20.0]])},
            intervene_actions=torch.tensor([[3.0], [4.0]]),
            intervene_flags=torch.tensor([[True], [False]]),
        ),
    )
    second = PolicyInput(
        obs={"states": torch.tensor([[5.0]])},
        env_result=EnvResult(
            rewards=torch.tensor([[2.5]]),
            dones=torch.tensor([[False]]),
        ),
    )

    merged = merge_policy_inputs([first, second])

    assert torch.equal(merged.obs["states"], torch.tensor([[1.0], [2.0], [5.0]]))
    assert torch.equal(
        merged.env_result.final_obs["states"],
        torch.tensor([[10.0], [20.0], [5.0]]),
    )
    assert torch.equal(
        merged.env_result.intervene_actions,
        torch.tensor([[3.0], [4.0], [0.0]]),
    )
    assert torch.equal(
        merged.env_result.intervene_flags,
        torch.tensor([[True], [False], [False]]),
    )


def test_policy_output_preserves_rollout_actions():
    actions = torch.arange(12).reshape(3, 4).transpose(0, 1)

    assert PolicyOutput(actions=actions).actions is actions


def test_policy_output_exposes_its_batch_dimension_to_routing():
    output = PolicyOutput(actions=torch.zeros(3, 2))

    assert EnvWorker._infer_policy_output_batch_size(output) == 3


def test_policy_output_shards_are_contiguous_for_p2p():
    output = PolicyOutput(actions=torch.arange(12).reshape(3, 4).transpose(0, 1))

    shards = MultiStepRolloutWorker._split_policy_output(output, [2, 2])

    assert all(shard.actions.is_contiguous() for shard in shards)


def test_env_result_keeps_chunk_data_only_for_online_lerobot():
    worker = EnvWorker.__new__(EnvWorker)
    env_output = EnvOutput(obs={"states": torch.zeros(1, 2)})
    chunk_step_data = {"chunk_actions": np.zeros((1, 2))}

    worker.enable_online_lerobot = False
    assert (
        worker._build_env_result(
            env_output, chunk_step_data=chunk_step_data
        ).episode_data
        is None
    )

    worker.enable_online_lerobot = True
    assert (
        worker._build_env_result(
            env_output, chunk_step_data=chunk_step_data
        ).episode_data
        is chunk_step_data
    )


def test_split_policy_input_preserves_lerobot_episode_data_and_sources():
    policy_input = PolicyInput(
        obs={"states": torch.tensor([[1.0], [2.0]])},
        env_result=EnvResult(
            episode_data={
                "chunk_actions": torch.tensor([[0.1], [0.2]]),
                "obs_list": [{"states": torch.tensor([[1.0], [2.0]])}],
                "terminations": torch.tensor([[False], [True]]),
                "truncations": torch.tensor([[False], [False]]),
                "infos_list": [{"success": torch.tensor([False, True])}],
            }
        ),
        sources=[(3, 0, 2)],
    )

    first, second = split_policy_input(policy_input, [1, 1])

    assert first.sources == [(3, 0, 1)]
    assert second.sources == [(3, 0, 1)]
    assert torch.equal(
        second.env_result.episode_data["chunk_actions"], torch.tensor([[0.2]])
    )
    assert torch.equal(
        first.env_result.episode_data["infos_list"][0]["success"],
        torch.tensor([False]),
    )


def test_lerobot_episode_data_round_trips_numpy_and_nested_values():
    episode_data = {
        "chunk_actions": np.array([[[1.0]], [[2.0]], [[3.0]]]),
        "obs_list": [
            {
                "states": np.array([[1.0], [2.0], [3.0]]),
                "camera": {"image": np.arange(12).reshape(3, 2, 2)},
            }
        ],
        "terminations": np.array([[False], [True], [False]]),
        "truncations": np.array([[False], [False], [True]]),
        "infos_list": [
            {
                "success": np.array([False, True, False]),
                "nested": {"score": np.array([0.1, 0.2, 0.3])},
            }
        ],
    }
    policy_input = PolicyInput(
        obs={"states": torch.tensor([[1.0], [2.0], [3.0]])},
        env_result=EnvResult(episode_data=episode_data),
        sources=[(0, 0, 1), (1, 0, 2)],
    )

    policy_input = pickle.loads(pickle.dumps(policy_input))
    shards = split_policy_input(policy_input, [1, 2])
    merged = merge_policy_inputs(shards)

    assert shards[0].sources == [(0, 0, 1)]
    assert shards[1].sources == [(1, 0, 2)]
    assert np.array_equal(
        merged.env_result.episode_data["chunk_actions"], episode_data["chunk_actions"]
    )
    assert np.array_equal(
        merged.env_result.episode_data["obs_list"][0]["camera"]["image"],
        episode_data["obs_list"][0]["camera"]["image"],
    )
    assert np.array_equal(
        merged.env_result.episode_data["infos_list"][0]["nested"]["score"],
        episode_data["infos_list"][0]["nested"]["score"],
    )


def test_trajectory_worker_splits_lerobot_numpy_episode_data():
    worker = TrajectoryWorker.__new__(TrajectoryWorker)
    episode_data = {
        "chunk_actions": np.array([[[1.0]], [[2.0]]]),
        "obs_list": [{"states": np.array([[1.0], [2.0]])}],
        "terminations": np.array([[False], [True]]),
        "truncations": np.array([[False], [False]]),
        "infos_list": [{"success": np.array([False, True])}],
    }
    segment = TrajectorySegment(
        step_id=0,
        epoch_id=0,
        sources=[(0, 0, 1), (1, 0, 1)],
        obs={"states": torch.tensor([[1.0], [2.0]])},
        next_obs={"states": torch.tensor([[3.0], [4.0]])},
        rollout_result=RolloutResult(
            actions=torch.tensor([[0.1], [0.2]]),
            forward_inputs={"action": torch.tensor([[0.1], [0.2]])},
        ),
        env_result=EnvResult(episode_data=episode_data),
    )

    shards = list(worker._split_segments(segment))

    assert [source for source, _ in shards] == [(0, 0), (1, 0)]
    assert np.array_equal(
        shards[1][1].env_result.episode_data["chunk_actions"],
        np.array([[[2.0]]]),
    )
    assert np.array_equal(
        shards[0][1].env_result.episode_data["infos_list"][0]["success"],
        np.array([False]),
    )


def test_trajectory_events_make_p2p_tensors_contiguous():
    segment = TrajectorySegment(
        step_id=0,
        epoch_id=0,
        sources=[(0, 0, 2)],
        obs={"states": torch.zeros(2, 3).transpose(0, 1)},
        next_obs={"states": torch.zeros(2, 3).transpose(0, 1)},
        rollout_result=RolloutResult(actions=torch.zeros(2, 1)),
        env_result=EnvResult(),
        next_forward_inputs={"states": torch.zeros(2, 3).transpose(0, 1)},
    )
    event = TrajectoryEpochEnd(
        step_id=0,
        epoch_id=0,
        source=(0, 0),
        sources=[(0, 0, 2)],
        final_prev_values=torch.zeros(2, 3).transpose(0, 1),
    )

    assert segment.obs["states"].is_contiguous()
    assert segment.next_obs["states"].is_contiguous()
    assert segment.next_forward_inputs["states"].is_contiguous()
    assert event.final_prev_values.is_contiguous()


def test_trajectory_worker_collects_rollout_owned_segment():
    worker = TrajectoryWorker.__new__(TrajectoryWorker)
    worker.collectors = {}
    worker.use_training_pipeline = False
    worker.collect_prev_infos = True
    worker.collect_transitions = True
    worker.enable_rlt = False
    worker.enable_online_lerobot = False
    worker.env_reward_weight = 1.0
    worker.reward_weight = 1.0
    worker.cfg = type(
        "Cfg",
        (),
        {
            "env": type(
                "Env",
                (),
                {
                    "train": type(
                        "Train", (), {"auto_reset": True, "max_episode_steps": 1}
                    )
                },
            ),
            "algorithm": {"gamma": 0.9, "bootstrap_type": "standard"},
        },
    )()

    worker._append(
        TrajectorySegment(
            step_id=0,
            epoch_id=0,
            sources=[(0, 0, 1)],
            obs={"states": torch.tensor([[1.0]])},
            next_obs={"states": torch.tensor([[2.0]])},
            rollout_result=RolloutResult(
                actions=torch.tensor([[3.0]]),
                prev_logprobs=torch.tensor([[0.1]]),
                prev_values=torch.tensor([[0.2]]),
                bootstrap_values=torch.tensor([[2.0]]),
                forward_inputs={"action": torch.tensor([[3.0]])},
                versions=torch.tensor([[1.0]]),
            ),
            env_result=EnvResult(
                rewards=torch.tensor([[1.0]]),
                dones=torch.tensor([[True]]),
                truncations=torch.tensor([[True]]),
                terminations=torch.tensor([[False]]),
            ),
            initial_env_result=EnvResult(
                dones=torch.tensor([[False]]),
                truncations=torch.tensor([[False]]),
                terminations=torch.tensor([[False]]),
            ),
        )
    )
    worker._append_final_values(
        TrajectoryEpochEnd(
            step_id=0,
            epoch_id=0,
            source=(0, 0),
            sources=[(0, 0, 1)],
            final_prev_values=torch.tensor([[0.4]]),
        )
    )

    trajectory = worker.collectors[(0, 0)].to_trajectory()

    assert torch.equal(trajectory.actions, torch.tensor([[[3.0]]]))
    assert torch.equal(trajectory.rewards, torch.tensor([[[2.8]]]))
    assert torch.equal(trajectory.dones, torch.tensor([[[False]], [[True]]]))
    assert torch.equal(trajectory.prev_values, torch.tensor([[[0.2]], [[0.4]]]))
    loss_mask, loss_mask_sum = compute_loss_mask(trajectory.dones)
    advantages_input = preprocess_embodied_advantages_inputs(
        rewards=trajectory.rewards,
        dones=trajectory.dones,
        values=trajectory.prev_values,
        loss_mask=loss_mask,
        loss_mask_sum=loss_mask_sum,
        adv_type="gae",
        reward_type="step_level",
    )
    assert advantages_input["rewards"].shape == (1, 1)
    assert advantages_input["dones"].shape == (2, 1)
    assert advantages_input["values"].shape == (2, 1)
    assert torch.equal(trajectory.curr_obs["states"], torch.tensor([[[1.0]]]))
    assert torch.equal(trajectory.next_obs["states"], torch.tensor([[[2.0]]]))


def test_trajectory_worker_collects_online_lerobot_episode():
    worker = TrajectoryWorker.__new__(TrajectoryWorker)
    worker.collectors = {}
    worker.use_training_pipeline = False
    worker.source_count = 1
    worker.enable_online_lerobot = True
    worker.lerobot_only_success = False
    worker.cfg = type(
        "Cfg",
        (),
        {
            "env": type(
                "Env",
                (),
                {
                    "train": type(
                        "Train",
                        (),
                        {"max_episode_steps": 1, "total_num_envs": 1},
                    )
                },
            ),
            "actor": type(
                "Actor",
                (),
                {"model": type("Model", (), {"num_action_chunks": 1, "action_dim": 1})},
            ),
        },
    )()

    worker._append(
        TrajectorySegment(
            step_id=0,
            epoch_id=0,
            sources=[(0, 0, 1)],
            obs={"states": torch.tensor([[1.0, 2.0]])},
            next_obs={"states": torch.tensor([[3.0, 4.0]])},
            rollout_result=RolloutResult(
                actions=torch.tensor([[0.5]]),
                forward_inputs={"action": torch.tensor([[0.5]])},
            ),
            env_result=EnvResult(
                episode_data={
                    "chunk_actions": torch.tensor([[0.5]]),
                    "obs_list": [
                        {
                            "states": torch.tensor([[1.0, 2.0]]),
                            "task_descriptions": ["pick"],
                        }
                    ],
                    "terminations": torch.tensor([[True]]),
                    "truncations": torch.tensor([[False]]),
                    "infos_list": [{}],
                }
            ),
        )
    )

    episodes = worker.collectors[(0, 0)].drain_episodes()

    assert len(episodes) == 1
    assert episodes[0][0]["task"] == "pick"
    assert episodes[0][0]["done"].item()


def test_pipeline_flushes_only_the_completed_step_epoch():
    worker = TrajectoryWorker.__new__(TrajectoryWorker)
    first_epoch = {(0, 0): object(), (1, 0): object()}
    second_epoch = {(0, 0): object(), (1, 0): object()}
    worker.pipeline_collectors = {(7, 0): first_epoch, (7, 1): second_epoch}
    worker.finished_epoch_sources = {
        (7, 0): {(0, 0), (1, 0)},
        (7, 1): {(0, 0)},
    }
    worker.producer_count = 2
    worker.source_count = 2
    published = []
    worker._publish_pipeline = published.append

    worker._flush_pipeline_epoch((7, 0))

    assert published == [first_epoch]
    assert (7, 0) not in worker.pipeline_collectors
    assert (7, 0) not in worker.finished_epoch_sources
    assert worker.pipeline_collectors[(7, 1)] is second_epoch
    assert worker.finished_epoch_sources[(7, 1)] == {(0, 0)}


def test_pipeline_normalizes_all_sources_before_publishing():
    class Collector:
        def __init__(self, value: float):
            self.value = value

        def to_splited_trajectories_by_sizes(self, sizes):
            assert sizes == [1]
            return [SimpleNamespace(value=self.value)]

    class Output:
        def __init__(self):
            self.items = []

        def put(self, item, **kwargs):
            self.items.append((item, kwargs))

    worker = TrajectoryWorker.__new__(TrajectoryWorker)
    worker.cfg = SimpleNamespace(
        rollout=SimpleNamespace(pipeline_stage_num=1),
        env=SimpleNamespace(train=SimpleNamespace(total_num_envs=2)),
        actor=SimpleNamespace(seed=17, micro_batch_size=1),
        algorithm={"normalize_advantages": True},
    )
    worker.shuffle_rollout = False
    worker.source_count = 2
    worker.pipeline_generators = {}
    worker.output = Output()
    worker._component_placement = lambda: SimpleNamespace(
        get_world_size=lambda component: 1 if component == "actor" else None
    )

    def prepare_batch(trajectories):
        value = trajectories[0].value
        return {
            "prev_logprobs": torch.tensor([[[value]]]),
            "advantages": torch.tensor([[[value]]]),
        }

    worker._prepare_pipeline_batch = prepare_batch

    worker._publish_pipeline({(0, 0): Collector(1.0), (1, 0): Collector(3.0)})

    assert [kwargs["key"] for _, kwargs in worker.output.items] == [
        "0_0_pipeline_actor",
        "0_0_pipeline_actor",
    ]
    advantages = [item["advantages"].item() for item, _ in worker.output.items]
    assert torch.allclose(torch.tensor(advantages), torch.tensor([-1.0, 1.0]))
