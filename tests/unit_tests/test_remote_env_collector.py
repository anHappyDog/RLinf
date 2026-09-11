# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import socket
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.config import _validate_remote_behavior_collectors
from rlinf.envs.remote_collector import (
    RemoteCollectorClient,
    RemoteCollectorConnectionError,
    RemoteCollectorProtocolError,
    RemoteCollectorServer,
    decode_message,
    encode_message,
    recv_frame,
    send_frame,
)


def _start_server(handler, token="test-token"):
    ready = threading.Event()
    server = RemoteCollectorServer("127.0.0.1", 0, auth_token=token, handler=handler)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"ready": ready}, daemon=True
    )
    thread.start()
    assert ready.wait(timeout=2)
    return server, thread


def _request(session_id: str, request_id: int, method: str, payload=None) -> bytes:
    return encode_message(
        {
            "protocol_version": 1,
            "auth_token": "test-token",
            "session_id": session_id,
            "request_id": request_id,
            "method": method,
            "payload": payload,
        }
    )


def test_safe_tree_codec_round_trips_environment_payloads():
    value = {
        "images": torch.arange(24, dtype=torch.uint8).reshape(2, 3, 4),
        "state": torch.tensor([1.25, -2.5], dtype=torch.bfloat16),
        "empty": torch.empty(2, 0, 3),
        "rewards": np.array([[0.0, 1.0]], dtype=np.float32),
        "metadata": ("snapshot-1", [True, None, 4]),
        "bytes": b"rlinf",
    }

    decoded = decode_message(encode_message(value))

    assert torch.equal(decoded["images"], value["images"])
    assert torch.equal(decoded["state"], value["state"])
    assert decoded["empty"].shape == value["empty"].shape
    assert np.array_equal(decoded["rewards"], value["rewards"])
    assert decoded["metadata"] == value["metadata"]
    assert decoded["bytes"] == value["bytes"]


def test_safe_tree_codec_rejects_object_arrays():
    with pytest.raises(TypeError, match="Object-dtype"):
        encode_message(np.array([object()], dtype=object))


def test_client_calls_server_sequentially():
    calls = []

    def handler(method, payload):
        calls.append((method, payload))
        return {"method": method, "payload": payload}

    server, thread = _start_server(handler)
    client = RemoteCollectorClient(
        "127.0.0.1", server.bound_port, auth_token="test-token"
    )
    try:
        assert client.call("initialize", {"rank": 3})["payload"] == {"rank": 3}
        assert client.call("chunk_step", torch.tensor([2.0]))["method"] == "chunk_step"
        assert [method for method, _ in calls] == ["initialize", "chunk_step"]
        metrics = client.last_call_metrics()
        assert metrics["method"] == "chunk_step"
        assert metrics["rpc_seconds"] >= metrics["handler_seconds"] >= 0.0
        assert metrics["request_mib"] > 0.0
        assert metrics["response_mib"] > 0.0
        assert metrics["reconnects"] == 0.0
    finally:
        client.close()
        server.shutdown()
        thread.join(timeout=2)


def test_client_retries_temporary_tunnel_start_failure(monkeypatch):
    server, thread = _start_server(lambda method, payload: payload)

    class FlakyTunnel:
        def __init__(self):
            self.start_count = 0
            self.close_count = 0

        def start(self):
            self.start_count += 1
            if self.start_count == 1:
                raise RemoteCollectorConnectionError("temporary SSH rejection")

        def close(self):
            self.close_count += 1

    tunnel = FlakyTunnel()
    client = RemoteCollectorClient(
        "127.0.0.1",
        server.bound_port,
        auth_token="test-token",
        reconnect_attempts=1,
        tunnel=tunnel,
    )
    monkeypatch.setattr("rlinf.envs.remote_collector.time.sleep", lambda _: None)
    try:
        assert client.call("initialize", {"rank": 3}) == {"rank": 3}
        assert tunnel.start_count == 2
        assert tunnel.close_count == 1
    finally:
        client.close()
        server.shutdown()
        thread.join(timeout=2)


def test_duplicate_request_returns_cached_response_without_reexecution():
    call_count = 0

    def handler(method, payload):
        nonlocal call_count
        call_count += 1
        return {"call_count": call_count, "method": method, "payload": payload}

    server, thread = _start_server(handler)
    session_id = "stable-session"
    initialize = _request(session_id, 0, "initialize")
    chunk = _request(session_id, 1, "chunk_step", torch.tensor([7.0]))
    try:
        with socket.create_connection(("127.0.0.1", server.bound_port)) as connection:
            send_frame(connection, initialize)
            recv_frame(connection)
            send_frame(connection, chunk)
            first_response = recv_frame(connection)
        with socket.create_connection(("127.0.0.1", server.bound_port)) as connection:
            send_frame(connection, chunk)
            duplicate_response = recv_frame(connection)

        assert duplicate_response == first_response
        assert decode_message(first_response)["result"]["call_count"] == 2
        assert call_count == 2
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_duplicate_request_id_with_different_content_is_rejected():
    server, thread = _start_server(lambda method, payload: payload)
    session_id = "stable-session"
    try:
        with socket.create_connection(("127.0.0.1", server.bound_port)) as connection:
            send_frame(connection, _request(session_id, 0, "initialize"))
            recv_frame(connection)
            send_frame(connection, _request(session_id, 0, "initialize", "changed"))
            response = decode_message(recv_frame(connection))
        assert not response["ok"]
        assert response["error_type"] == "RemoteCollectorProtocolError"
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_closed_session_releases_collector_for_next_session():
    calls = []

    def handler(method, payload):
        calls.append((method, payload))
        return payload

    server, thread = _start_server(handler)
    first_client = RemoteCollectorClient(
        "127.0.0.1", server.bound_port, auth_token="test-token"
    )
    second_client = RemoteCollectorClient(
        "127.0.0.1", server.bound_port, auth_token="test-token"
    )
    try:
        first_client.call("initialize", {"run": 1})
        first_client.call("close")
        first_client.close()

        assert second_client.call("initialize", {"run": 2}) == {"run": 2}
        assert calls == [
            ("initialize", {"run": 1}),
            ("close", None),
            ("initialize", {"run": 2}),
        ]
    finally:
        first_client.close()
        second_client.close()
        server.shutdown()
        thread.join(timeout=2)


def test_decode_rejects_trailing_data():
    with pytest.raises(RemoteCollectorProtocolError, match="trailing bytes"):
        decode_message(encode_message({"ok": True}) + b"unexpected")


def test_distributed_behavior_routes_only_assigned_rank_remotely(monkeypatch):
    from rlinf.envs.behavior import remote_collector as behavior_remote

    captured = {}

    class FakeRemote:
        def __init__(self, cfg, num_envs, seed_offset, total, worker_info, endpoint):
            captured.update(
                rank=seed_offset,
                endpoint=int(endpoint.env_rank),
                total=total,
                group_world_size=worker_info.group_world_size,
            )
            self.is_start = True

        def close(self):
            pass

    monkeypatch.setattr(behavior_remote, "RemoteBehaviorSubpoolEnv", FakeRemote)
    cfg = OmegaConf.create(
        {
            "remote_collector": {
                "enabled": True,
                "endpoints": [{"env_rank": 5, "ssh_host": "collector", "port": 46005}],
            }
        }
    )

    env = behavior_remote.DistributedBehaviorSubpoolEnv(
        cfg,
        num_envs=1,
        seed_offset=5,
        total_num_processes=16,
        worker_info=SimpleNamespace(group_world_size=16),
    )

    assert captured == {
        "rank": 5,
        "endpoint": 5,
        "total": 16,
        "group_world_size": 16,
    }
    env.close()


def test_remote_collector_config_requires_one_env_per_worker():
    cfg = OmegaConf.create(
        {
            "total_num_envs": 8,
            "subpool": {"enabled": True, "dynamic_updates": False},
            "remote_collector": {
                "enabled": True,
                "endpoints": [{"env_rank": 1, "ssh_host": "collector", "port": 46100}],
            },
        }
    )

    _validate_remote_behavior_collectors(cfg, env_world_size=8)
    with pytest.raises(AssertionError, match="one logical environment"):
        _validate_remote_behavior_collectors(cfg, env_world_size=4)


def test_remote_collector_config_rejects_duplicate_assignments():
    cfg = OmegaConf.create(
        {
            "total_num_envs": 8,
            "subpool": {"enabled": True, "dynamic_updates": False},
            "remote_collector": {
                "enabled": True,
                "endpoints": [
                    {"env_rank": 1, "ssh_host": "collector-a", "port": 46100},
                    {"env_rank": 1, "ssh_host": "collector-b", "port": 46100},
                ],
            },
        }
    )

    with pytest.raises(AssertionError, match="unique env_rank"):
        _validate_remote_behavior_collectors(cfg, env_world_size=8)


def test_remote_behavior_closes_transport_when_initialization_fails(monkeypatch):
    from rlinf.envs.behavior import remote_collector as behavior_remote

    clients = []

    class FailingClient:
        def __init__(self, *_args, **_kwargs):
            self.closed = False
            clients.append(self)

        def call(self, _method, _payload):
            raise RuntimeError("initialization failed")

        def close(self):
            self.closed = True

    monkeypatch.setattr(behavior_remote, "RemoteCollectorClient", FailingClient)
    monkeypatch.setenv("RLINF_REMOTE_COLLECTOR_TOKEN", "test-token")
    cfg = OmegaConf.create(
        {
            "seed": 1,
            "remote_collector": {
                "enabled": True,
                "auth_token_env": "RLINF_REMOTE_COLLECTOR_TOKEN",
            },
        }
    )
    endpoint = OmegaConf.create({"env_rank": 0, "host": "127.0.0.1", "port": 1})

    with pytest.raises(RuntimeError, match="initialization failed"):
        behavior_remote.RemoteBehaviorSubpoolEnv(
            cfg,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=SimpleNamespace(group_world_size=1),
            endpoint=endpoint,
        )
    assert len(clients) == 1
    assert clients[0].closed


def test_remote_behavior_matches_noop_reset_state_id_hook():
    from rlinf.envs.behavior.remote_collector import RemoteBehaviorSubpoolEnv

    env = object.__new__(RemoteBehaviorSubpoolEnv)

    assert env.update_reset_state_ids() is None


def test_remote_behavior_forwards_policy_global_step():
    from rlinf.envs.behavior.remote_collector import RemoteBehaviorSubpoolEnv

    class FakeClient:
        def __init__(self):
            self.calls = []

        def call(self, method, payload):
            self.calls.append((method, payload))
            return {"result": None, "attributes": {}}

    env = object.__new__(RemoteBehaviorSubpoolEnv)
    env._client = FakeClient()

    env.set_policy_global_step(34)

    assert env._client.calls == [("set_policy_global_step", 34)]


def test_behavior_service_close_releases_environment():
    from rlinf.envs.behavior.remote_collector import BehaviorCollectorService

    class FakeEnv:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    service = BehaviorCollectorService()
    env = FakeEnv()
    service.env = env

    response = service("close", None)

    assert response == {"result": None, "attributes": {}}
    assert env.closed
    assert service.env is None


def test_env_worker_closes_train_and_eval_environments():
    from rlinf.workers.env.env_worker import EnvWorker

    class FakeEnv:
        def __init__(self):
            self.close_count = 0

        def close(self):
            self.close_count += 1

    train_env = FakeEnv()
    eval_env = FakeEnv()
    worker = object.__new__(EnvWorker)
    worker.env_list = [train_env]
    worker.eval_env_list = [eval_env]

    worker.close_envs()

    assert train_env.close_count == 1
    assert eval_env.close_count == 1


def test_eval_runner_closes_environments_after_success(monkeypatch):
    from rlinf.runners import embodied_eval_runner
    from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner

    class FakeHandle:
        def __init__(self):
            self.wait_count = 0

        def wait(self):
            self.wait_count += 1

    class FakeEnvGroup:
        def __init__(self):
            self.close_count = 0
            self.handle = FakeHandle()

        def close_envs(self):
            self.close_count += 1
            return self.handle

    class FakeMetricLogger:
        log_path = "/tmp/test-eval-runner"

        def log(self, *, step, data):
            assert step == 0
            assert data == {"eval/success": 0.5}

        def finish(self):
            pass

    env = FakeEnvGroup()
    runner = object.__new__(EmbodiedEvalRunner)
    runner.evaluate = lambda: {"success": 0.5}
    runner.logger = SimpleNamespace(info=lambda _metrics: None)
    runner.metric_logger = FakeMetricLogger()
    runner.env = env
    monkeypatch.setattr(embodied_eval_runner, "print_metrics_table", lambda **_: None)

    runner.run()

    assert env.close_count == 1
    assert env.handle.wait_count == 1


def test_eval_runner_closes_environments_after_failure():
    from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner

    class FakeHandle:
        def __init__(self):
            self.wait_count = 0

        def wait(self):
            self.wait_count += 1

    class FakeEnvGroup:
        def __init__(self):
            self.close_count = 0
            self.handle = FakeHandle()

        def close_envs(self):
            self.close_count += 1
            return self.handle

    class FakeMetricLogger:
        def __init__(self):
            self.finish_count = 0

        def finish(self):
            self.finish_count += 1

    env = FakeEnvGroup()
    runner = object.__new__(EmbodiedEvalRunner)

    def fail_evaluation():
        raise RuntimeError("evaluation failed")

    runner.evaluate = fail_evaluation
    runner.metric_logger = FakeMetricLogger()
    runner.env = env

    with pytest.raises(RuntimeError, match="evaluation failed"):
        runner.run()

    assert runner.metric_logger.finish_count == 1
    assert env.close_count == 1
    assert env.handle.wait_count == 1
