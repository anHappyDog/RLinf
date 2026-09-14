import asyncio
from types import SimpleNamespace

import torch

from rlinf.data.schema.embodied_types import PolicyOutput
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def _make_worker(receive_sizes: list[list[int]]):
    worker = object.__new__(MultiStepRolloutWorker)
    worker.dynamic_batch_max_size = 2
    worker.dynamic_batch_wait_seconds = 0.005
    worker.total_num_train_envs = 4
    worker.train_batch_size = 4
    worker.collect_prev_infos = True
    worker.enable_opd = False
    worker.rlt_feature_model = None
    worker.version = 3
    worker.cfg = SimpleNamespace(env=SimpleNamespace(group_name="EnvGroup"))
    worker.placement = SimpleNamespace(get_world_size=lambda group: 4)
    worker._world_size = 1
    worker.sent = []
    worker.requested_queue_sizes = []

    async def recv(**kwargs):
        worker.requested_queue_sizes.append(kwargs["recv_queue_size"])
        split_sizes = receive_sizes.pop(0)
        batch_size = sum(split_sizes)
        obs = {"states": torch.zeros(batch_size, 3)}
        return {"obs": obs, "final_obs": None}, split_sizes

    def predict(obs, **kwargs):
        batch_size = obs["states"].shape[0]
        return torch.zeros(batch_size, 2), {
            "prev_logprobs": torch.zeros(batch_size, 2),
            "prev_values": torch.zeros(batch_size, 1),
            "forward_inputs": {"action": torch.zeros(batch_size, 2)},
        }

    def build(actions, result, **kwargs):
        return PolicyOutput(
            actions=actions,
            prev_logprobs=result["prev_logprobs"],
            prev_values=result["prev_values"],
            forward_inputs=result["forward_inputs"],
            versions=torch.full_like(result["prev_logprobs"], worker.version),
        )

    def send(**kwargs):
        worker.sent.append((kwargs["tag"], kwargs["split_sizes"], kwargs["data"]))

    worker.recv_from_and_record_batch_routes_with_timeout = recv
    worker._predict_rollout_actions = predict
    worker._build_policy_output = build
    worker.get_bootstrap_values = lambda final_obs: None
    worker.send_to_recorded_batch_routes = send
    worker._merge_obs_batches = None
    worker._infer_env_batch_size = None
    worker._split_policy_output = None
    return worker


def test_dynamic_phase_processes_exact_target_without_overshoot():
    worker = _make_worker([[1, 1], [1, 1], [1]])
    batch_sizes = asyncio.run(
        worker._generate_dynamic_batch_phase(
            None,
            None,
            tag="rollout_results",
            target_rows=5,
            bootstrap_only=False,
        )
    )

    assert batch_sizes == [2, 2, 1]
    assert worker.requested_queue_sizes == [2, 2, 1]
    assert [tag for tag, _, _ in worker.sent] == ["rollout_results"] * 3


def test_bootstrap_phase_preserves_synchronous_output_contract():
    worker = _make_worker([[1, 1]])
    asyncio.run(
        worker._generate_dynamic_batch_phase(
            None,
            None,
            tag="rollout_bootstrap",
            target_rows=2,
            bootstrap_only=True,
        )
    )

    _, _, output = worker.sent[0]
    assert output.actions is not None
    assert output.prev_values is not None
    assert output.prev_logprobs is None
    assert output.versions is None
    assert output.forward_inputs == {}
