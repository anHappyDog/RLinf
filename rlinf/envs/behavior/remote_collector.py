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

"""Cross-datacenter proxy and daemon for ``BehaviorSubpoolEnv``."""

from __future__ import annotations

import argparse
import logging
import os
import signal
from types import SimpleNamespace
from typing import Any

import ray
from omegaconf import DictConfig, OmegaConf

from rlinf.envs.remote_collector import (
    RemoteCollectorClient,
    RemoteCollectorServer,
    SshLocalForward,
    auth_token_from_env,
)

_REMOTE_ATTRIBUTES = (
    "last_executed_action_mask",
    "subtask_ids",
    "subpool_ids",
    "outcome_group_reset_metadata",
    "is_start",
)


def _find_endpoint(remote_cfg: DictConfig, env_rank: int) -> DictConfig | None:
    matches = [
        endpoint
        for endpoint in remote_cfg.get("endpoints", [])
        if int(endpoint.env_rank) == env_rank
    ]
    if len(matches) > 1:
        raise ValueError(
            f"Multiple remote collector endpoints target env rank {env_rank}."
        )
    return matches[0] if matches else None


def _remote_env_config(cfg: DictConfig, endpoint: DictConfig) -> dict[str, Any]:
    """Build a resolved server config without client connection credentials."""
    remote_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    remote_cfg.pop("remote_collector", None)
    overrides = endpoint.get("env_overrides", None)
    if overrides:
        remote_cfg = OmegaConf.merge(remote_cfg, overrides)
    return OmegaConf.to_container(
        remote_cfg,
        resolve=True,
        throw_on_missing=True,
    )


class RemoteBehaviorSubpoolEnv:
    """Expose a remote ``BehaviorSubpoolEnv`` through its local interface."""

    def __init__(
        self,
        cfg: DictConfig,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info,
        endpoint: DictConfig,
    ) -> None:
        if num_envs != 1:
            raise ValueError(
                "A remote BEHAVIOR collector owns exactly one environment."
            )
        self.cfg = cfg
        self.num_envs = num_envs
        self.seed_offset = seed_offset
        self.seed = int(cfg.seed) + seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.last_executed_action_mask = None
        self.subtask_ids = None
        self.subpool_ids = None
        self.outcome_group_reset_metadata = None
        self._is_start = True
        self._closed = False

        port = int(endpoint.port)
        ssh_host = endpoint.get("ssh_host", None)
        tunnel = None
        if ssh_host:
            tunnel = SshLocalForward(
                str(ssh_host),
                port,
                local_host=str(endpoint.get("local_host", "127.0.0.1")),
                local_port=endpoint.get("local_port", None),
                connect_timeout=float(endpoint.get("connect_timeout", 30.0)),
                server_alive_interval=int(endpoint.get("server_alive_interval", 15)),
                server_alive_count_max=int(endpoint.get("server_alive_count_max", 3)),
                compression=bool(endpoint.get("compression", True)),
                extra_options=list(endpoint.get("ssh_options", [])),
            )
            host = tunnel.local_host
            client_port = tunnel.local_port
        else:
            host = str(endpoint.get("host", "127.0.0.1"))
            client_port = port

        token_env = str(
            endpoint.get(
                "auth_token_env",
                cfg.remote_collector.get(
                    "auth_token_env", "RLINF_REMOTE_COLLECTOR_TOKEN"
                ),
            )
        )
        response_compression = cfg.remote_collector.get("response_compression", {})
        endpoint_compression = endpoint.get("response_compression", None)
        if endpoint_compression is not None:
            response_compression = endpoint_compression
        self._client = RemoteCollectorClient(
            host,
            client_port,
            auth_token=auth_token_from_env(token_env),
            request_timeout=float(endpoint.get("request_timeout", 1200.0)),
            connect_timeout=float(endpoint.get("connect_timeout", 30.0)),
            reconnect_attempts=int(endpoint.get("reconnect_attempts", 3)),
            tunnel=tunnel,
            response_compression=str(response_compression.get("codec", "none")),
            response_compression_level=int(response_compression.get("level", 1)),
            response_compression_min_bytes=int(
                response_compression.get("min_bytes", 64 * 1024)
            ),
        )
        try:
            response = self._client.call(
                "initialize",
                {
                    "cfg": _remote_env_config(cfg, endpoint),
                    "num_envs": num_envs,
                    "seed_offset": seed_offset,
                    "total_num_processes": total_num_processes,
                    "group_world_size": int(worker_info.group_world_size),
                },
            )
        except Exception:
            self._client.close()
            raise
        self._consume_response(response)

    @property
    def is_start(self) -> bool:
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool) -> None:
        value = bool(value)
        if not hasattr(self, "_client"):
            self._is_start = value
            return
        response = self._client.call("set_is_start", value)
        self._consume_response(response)

    @property
    def device(self) -> str:
        """Remote results are materialized as CPU tensors in the EnvWorker."""
        return "cpu"

    def prepare_outcome_group_reset(
        self,
        collection_index: int,
        logical_group_index: int | None = None,
        update_index: int | None = None,
    ) -> None:
        response = self._client.call(
            "prepare_outcome_group_reset",
            {
                "collection_index": int(collection_index),
                "logical_group_index": logical_group_index,
                "update_index": update_index,
            },
        )
        self._consume_response(response)

    def set_policy_global_step(self, global_step: int) -> None:
        """Associate remote failure artifacts with the active policy version."""
        response = self._client.call("set_policy_global_step", int(global_step))
        self._consume_response(response)

    def update_reset_state_ids(self) -> None:
        """Match the no-op reset-ID hook exposed by local BEHAVIOR envs."""

    def reset(self):
        response = self._client.call("reset")
        return self._consume_response(response)

    def chunk_step(self, chunk_actions):
        response = self._client.call("chunk_step", chunk_actions)
        return self._consume_response(response)

    def remote_collector_metrics(self) -> dict[str, Any]:
        """Return timing and payload metrics for the latest action chunk RPC."""
        metrics = self._client.last_call_metrics()
        if metrics.get("method") != "chunk_step":
            return {}
        return {
            "time/remote_collector_rpc": metrics["rpc_seconds"],
            "time/remote_collector_handler": metrics["handler_seconds"],
            "time/remote_collector_transport": metrics["transport_seconds"],
            "remote_collector/request_mib": metrics["request_mib"],
            "remote_collector/response_mib": metrics["response_mib"],
            "remote_collector/response_raw_blob_mib": metrics["response_raw_blob_mib"],
            "remote_collector/response_compression_ratio": metrics[
                "response_compression_ratio"
            ],
            "remote_collector/reconnects": metrics["reconnects"],
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._client.call("close")
        finally:
            self._client.close()

    def _consume_response(self, response: Any) -> Any:
        if not isinstance(response, dict):
            raise TypeError("Remote BEHAVIOR response must be a dictionary.")
        attributes = response.get("attributes", {})
        if not isinstance(attributes, dict):
            raise TypeError("Remote BEHAVIOR attributes must be a dictionary.")
        for name, value in attributes.items():
            if name == "is_start":
                self._is_start = bool(value)
            elif name in _REMOTE_ATTRIBUTES:
                setattr(self, name, value)
        return response.get("result")


class DistributedBehaviorSubpoolEnv:
    """Choose a local or remote simulator from the global logical env rank."""

    def __init__(
        self,
        cfg: DictConfig,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info,
        record_metrics: bool = True,
    ) -> None:
        remote_cfg = cfg.get("remote_collector", {})
        endpoint = _find_endpoint(remote_cfg, seed_offset)
        if endpoint is None:
            from rlinf.envs.behavior.behavior_env import BehaviorSubpoolEnv

            delegate = BehaviorSubpoolEnv(
                cfg,
                num_envs,
                seed_offset,
                total_num_processes,
                worker_info,
                record_metrics=record_metrics,
            )
        else:
            if not record_metrics:
                raise ValueError(
                    "Remote BEHAVIOR collectors require record_metrics=true."
                )
            delegate = RemoteBehaviorSubpoolEnv(
                cfg,
                num_envs,
                seed_offset,
                total_num_processes,
                worker_info,
                endpoint,
            )
        object.__setattr__(self, "_delegate", delegate)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    @property
    def is_start(self) -> bool:
        return self._delegate.is_start

    @is_start.setter
    def is_start(self, value: bool) -> None:
        self._delegate.is_start = value

    def reset(self):
        return self._delegate.reset()

    def chunk_step(self, chunk_actions):
        return self._delegate.chunk_step(chunk_actions)

    def prepare_outcome_group_reset(
        self,
        collection_index: int,
        logical_group_index: int | None = None,
        update_index: int | None = None,
    ) -> None:
        self._delegate.prepare_outcome_group_reset(
            collection_index,
            logical_group_index,
            update_index,
        )

    def set_policy_global_step(self, global_step: int) -> None:
        self._delegate.set_policy_global_step(global_step)

    def close(self) -> None:
        self._delegate.close()


class BehaviorCollectorService:
    """RPC method dispatcher owning one real ``BehaviorSubpoolEnv``."""

    def __init__(self) -> None:
        self.env = None

    def __call__(self, method: str, payload: Any) -> dict[str, Any]:
        if method == "initialize":
            result = self._initialize(payload)
        else:
            if self.env is None:
                raise RuntimeError("Remote BEHAVIOR environment is not initialized.")
            if method == "set_is_start":
                self.env.is_start = bool(payload)
                result = None
            elif method == "prepare_outcome_group_reset":
                if not isinstance(payload, dict):
                    raise TypeError(
                        "prepare_outcome_group_reset payload must be a dictionary."
                    )
                self.env.prepare_outcome_group_reset(
                    int(payload["collection_index"]),
                    payload.get("logical_group_index"),
                    payload.get("update_index"),
                )
                result = None
            elif method == "set_policy_global_step":
                self.env.set_policy_global_step(int(payload))
                result = None
            elif method == "reset":
                result = self.env.reset()
            elif method == "chunk_step":
                result = self.env.chunk_step(payload)
            elif method == "close":
                self.env.close()
                self.env = None
                result = None
            else:
                raise KeyError(f"Unknown remote BEHAVIOR method {method!r}.")
        return {"result": result, "attributes": self._attributes()}

    def _initialize(self, payload: Any) -> None:
        if self.env is not None:
            raise RuntimeError("Remote BEHAVIOR environment is already initialized.")
        if not isinstance(payload, dict):
            raise TypeError("Initialize payload must be a dictionary.")
        from rlinf.envs.behavior.behavior_env import BehaviorSubpoolEnv

        cfg = OmegaConf.create(payload["cfg"])
        self.env = BehaviorSubpoolEnv(
            cfg,
            num_envs=int(payload["num_envs"]),
            seed_offset=int(payload["seed_offset"]),
            total_num_processes=int(payload["total_num_processes"]),
            worker_info=SimpleNamespace(
                group_world_size=int(payload["group_world_size"])
            ),
        )
        return None

    def _attributes(self) -> dict[str, Any]:
        if self.env is None:
            return {}
        attributes = {}
        for name in _REMOTE_ATTRIBUTES:
            try:
                value = getattr(self.env, name)
            except (AttributeError, RuntimeError):
                continue
            attributes[name] = value
        return attributes

    def close(self) -> None:
        if self.env is not None:
            self.env.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve one BEHAVIOR subpool environment over authenticated RPC."
    )
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--auth-token-env", default="RLINF_REMOTE_COLLECTOR_TOKEN")
    parser.add_argument("--allow-non-loopback", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run a persistent collector daemon, normally behind an SSH tunnel."""
    args = _parse_args()
    if (
        args.host not in ("127.0.0.1", "::1", "localhost")
        and not args.allow_non_loopback
    ):
        raise ValueError(
            "The collector protocol is authenticated but not encrypted. Bind to "
            "loopback and use SSH, or explicitly pass --allow-non-loopback."
        )
    logging.basicConfig(
        level=os.environ.get("RLINF_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ray.init(address=args.ray_address, log_to_driver=True)
    service = BehaviorCollectorService()
    server = RemoteCollectorServer(
        args.host,
        args.port,
        auth_token=auth_token_from_env(args.auth_token_env),
        handler=service,
    )

    def stop_server(_signum, _frame) -> None:
        server.shutdown()

    signal.signal(signal.SIGTERM, stop_server)
    signal.signal(signal.SIGINT, stop_server)
    logging.getLogger(__name__).info(
        "Remote BEHAVIOR collector listening on %s:%d.", args.host, args.port
    )
    try:
        server.serve_forever()
    finally:
        service.close()
        ray.shutdown()


if __name__ == "__main__":
    main()


__all__ = [
    "BehaviorCollectorService",
    "DistributedBehaviorSubpoolEnv",
    "RemoteBehaviorSubpoolEnv",
]
