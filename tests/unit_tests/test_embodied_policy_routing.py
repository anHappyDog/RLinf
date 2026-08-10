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

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import torch

from rlinf.data.schema import (
    EnvOutput,
    PolicyInput,
    PolicyOutput,
    TerminalRequest,
    merge_policy_inputs,
)
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def _policy_input() -> PolicyInput:
    return PolicyInput(
        obs={"states": torch.zeros(16, 4)},
    )


def test_decoupled_policy_route_round_trip():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.env_decoupled_mode = True
    rollout.cfg = SimpleNamespace(env=SimpleNamespace(group_name="EnvGroup"))
    rollout.train_batch_size = 32
    rollout.rollout_queue_size = 0
    rollout.batch_router = {"policy": ["stale"], "terminal": []}
    rollout.recv_from_and_record_batch_routes_with_timeout = AsyncMock(
        return_value=(_policy_input(), [8, 8])
    )
    rollout.send_to_recorded_batch_routes = Mock()

    policy_input, split_sizes = asyncio.run(rollout._receive_policy_input(None, 0))
    output = PolicyOutput(actions=torch.zeros(16, 4))
    rollout._send_policy_output(None, output, 0, split_sizes)

    assert policy_input.obs["states"].shape == (16, 4)
    rollout.recv_from_and_record_batch_routes_with_timeout.assert_awaited_once_with(
        group_name="EnvGroup",
        channel=None,
        tag="policy",
        batch_size=32,
        merge_fn=merge_policy_inputs,
        infer_batch_size_fn=rollout._infer_policy_input_batch_size,
        timeout_time=0.02,
        recv_queue_size=0,
    )
    rollout.send_to_recorded_batch_routes.assert_called_once_with(
        group_name="EnvGroup",
        channel=None,
        data=output,
        tag="policy",
        split_fn=rollout._split_policy_output,
        split_sizes=[8, 8],
    )


def test_decoupled_terminal_request_does_not_record_return_route():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.env_decoupled_mode = True
    rollout.cfg = SimpleNamespace(env=SimpleNamespace(group_name="EnvGroup"))
    rollout.train_batch_size = 32
    rollout.rollout_queue_size = 0
    rollout.batch_router = {"terminal": ["old"]}

    async def receive(**_):
        rollout.batch_router["terminal"].extend(["route-0", "route-1"])
        return TerminalRequest(obs={"states": torch.zeros(16, 4)}, sources=[]), [8, 8]

    rollout.recv_from_and_record_batch_routes_with_timeout = receive

    asyncio.run(rollout._receive_terminal_request(None, 0))

    assert rollout.batch_router["terminal"] == []


def test_env_uses_mode_qualified_decoupled_response_tag():
    env = object.__new__(EnvWorker)
    env.cfg = SimpleNamespace(rollout=SimpleNamespace(group_name="RolloutGroup"))
    env.env_decoupled_mode = True
    env.train_batch_size = 32
    env.recv_from = Mock(return_value=PolicyOutput(actions=torch.zeros(8, 4)))

    env._recv_policy_output(None, stage_id=0)

    env.recv_from.assert_called_once_with(
        group_name="RolloutGroup",
        channel=None,
        tag="train_policy",
        route_key=None,
        batch_size=32,
        infer_batch_size_fn=env._infer_policy_output_batch_size,
        decoupled_mode=True,
    )


def test_env_closes_reward_stream_with_final_chunk():
    env = object.__new__(EnvWorker)
    env.rollout_epoch = 2
    env.stage_num = 1
    env.n_train_chunk_steps = 2
    env._prefetched_train_bootstrap = None
    env._trajectory_step = 0
    env._rank = 0
    env.env_list = [SimpleNamespace()]
    env.use_training_pipeline = False
    env.cfg = SimpleNamespace(
        env=SimpleNamespace(
            train=SimpleNamespace(auto_reset=True, ignore_terminations=False)
        )
    )
    env._bootstrap_and_send_train = Mock(
        return_value=[EnvOutput(obs={"states": torch.zeros(1, 4)})]
    )
    env._maybe_wait_env_delay = AsyncMock()
    env._recv_policy_output = Mock(return_value=PolicyOutput(actions=torch.zeros(1, 4)))
    env.env_interact_step = Mock(
        return_value=(
            EnvOutput(
                obs={"states": torch.zeros(1, 4)},
                rewards=torch.zeros(1, 1),
            ),
            {},
            {},
        )
    )
    env.get_reward_model_output = Mock(return_value=torch.zeros(1, 1))
    env._publish_step = Mock()
    env.record_env_metrics = Mock()
    env.store_last_obs_and_intervened_info = Mock()
    env.finish_rollout = Mock()
    trajectory_channel = Mock()

    asyncio.run(
        EnvWorker._run_interact_once.__wrapped__(
            env,
            input_channel=Mock(),
            rollout_channel=Mock(),
            reward_channel=Mock(),
            trajectory_channel=trajectory_channel,
            cooperative_yield=False,
        )
    )

    assert env.get_reward_model_output.call_count == 4
    assert [
        call.kwargs["last_run"] for call in env.get_reward_model_output.call_args_list
    ] == [False, False, False, True]


def test_rollout_finishes_terminal_task_before_offload():
    rollout = object.__new__(MultiStepRolloutWorker)
    rollout.enable_offload = True
    rollout.offload_model = Mock()

    async def finish_generation():
        rollout._terminal_task = asyncio.create_task(asyncio.Event().wait())
        await rollout.finish_generation()

    asyncio.run(finish_generation())

    assert rollout._terminal_task is None
    rollout.offload_model.assert_called_once_with()
