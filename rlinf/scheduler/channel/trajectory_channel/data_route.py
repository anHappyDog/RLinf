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

from dataclasses import dataclass
from typing import Callable, Literal, TypeAlias

from rlinf.data.embodied_io_struct import (
    EnvResult,
    LeRobotStepResult,
    PolicyInput,
    PolicyOutput,
    RewardRequest,
    RewardResult,
    RolloutResult,
    TrajectoryData,
    ValueRequest,
    ValueResult,
)
from rlinf.scheduler.channel.trajectory_channel.owner_key import (
    OwnerKeyBuilder,
    lerobot_actor_owner_key,
    pipeline_batch_owner_key,
    trajectory_batch_owner_key,
)
from rlinf.scheduler.channel.trajectory_channel.storage import (
    LeRobotEpisodeBatch,
    PipelineMicroBatch,
    TrajectoryBatch,
)

Participant: TypeAlias = Literal[
    "actor",
    "env",
    "rollout",
    "reward",
]
RouteVia: TypeAlias = Literal["channel_worker", "storage_worker"]


@dataclass(frozen=True)
class DataRoute:
    """Declare the participants and transport path for one record type."""

    src: Participant | None
    dst: Participant | None
    via: RouteVia
    owner_key: OwnerKeyBuilder | None = None
    extra_key: str | None = None


DataRouteDict: TypeAlias = dict[type[TrajectoryData], DataRoute]
_DataRouteProvider: TypeAlias = Callable[[bool], DataRouteDict]

_DATA_ROUTE_PROVIDERS: dict[str, _DataRouteProvider] = {}


def register_data_routes(
    *algorithm_types: str,
) -> Callable[[_DataRouteProvider], _DataRouteProvider]:
    """Register a route provider for one or more algorithm types."""
    if not algorithm_types:
        raise ValueError("At least one algorithm type is required.")

    def decorator(provider: _DataRouteProvider) -> _DataRouteProvider:
        for algorithm_type in algorithm_types:
            key = algorithm_type.lower()
            if key in _DATA_ROUTE_PROVIDERS:
                raise ValueError(
                    "Data route provider for algorithm type "
                    f"'{algorithm_type}' is already registered."
                )
            _DATA_ROUTE_PROVIDERS[key] = provider
        return provider

    return decorator


def get_data_routes(
    algorithm_type: str, use_training_pipeline: bool = False
) -> DataRouteDict:
    """Build the routes registered for an algorithm type."""
    key = algorithm_type.lower()
    if key not in _DATA_ROUTE_PROVIDERS:
        raise ValueError(
            f"No data route provider registered for algorithm type '{algorithm_type}'"
        )
    return _DATA_ROUTE_PROVIDERS[key](use_training_pipeline)


def basic_policy_routes(use_training_pipeline: bool = False) -> DataRouteDict:
    """Return policy request, response, and training-batch routes."""
    routes = {
        PolicyInput: DataRoute(src="env", dst="rollout", via="channel_worker"),
        PolicyOutput: DataRoute(
            src="rollout",
            dst="env",
            via="channel_worker",
            extra_key="route_key",
        ),
    }
    if use_training_pipeline:
        routes[PipelineMicroBatch] = DataRoute(
            src=None,
            dst="actor",
            via="storage_worker",
            extra_key="actor_rank",
        )
    else:
        routes[TrajectoryBatch] = DataRoute(
            src=None,
            dst="actor",
            via="storage_worker",
        )
    return routes


def rollout_owner_key(use_training_pipeline: bool) -> OwnerKeyBuilder:
    """Select the storage ownership boundary for a rollout."""
    return (
        pipeline_batch_owner_key
        if use_training_pipeline
        else trajectory_batch_owner_key
    )


@register_data_routes("ppo", "nft", "opd")
def ppo_data_routes(use_training_pipeline: bool = False) -> DataRouteDict:
    """Return PPO routes including value and reward records."""
    owner_key = rollout_owner_key(use_training_pipeline)
    return {
        **basic_policy_routes(use_training_pipeline),
        ValueRequest: DataRoute(src="env", dst="rollout", via="storage_worker"),
        RewardRequest: DataRoute(src="env", dst="reward", via="storage_worker"),
        EnvResult: DataRoute(
            src="env",
            dst=None,
            via="storage_worker",
            owner_key=owner_key,
        ),
        RolloutResult: DataRoute(
            src="rollout",
            dst=None,
            via="storage_worker",
            owner_key=owner_key,
        ),
        ValueResult: DataRoute(
            src="rollout",
            dst=None,
            via="storage_worker",
            owner_key=owner_key,
        ),
        RewardResult: DataRoute(
            src="reward",
            dst=None,
            via="storage_worker",
            owner_key=owner_key,
        ),
    }


@register_data_routes("sac", "rlt_ac")
def sac_data_routes(use_training_pipeline: bool = False) -> DataRouteDict:
    """Return SAC routes for environment and rollout records."""
    owner_key = rollout_owner_key(use_training_pipeline)
    return {
        **basic_policy_routes(use_training_pipeline),
        EnvResult: DataRoute(
            src="env",
            dst=None,
            via="storage_worker",
            owner_key=owner_key,
        ),
        RolloutResult: DataRoute(
            src="rollout",
            dst=None,
            via="storage_worker",
            owner_key=owner_key,
        ),
    }


@register_data_routes("grpo")
def grpo_data_routes(use_training_pipeline: bool = False) -> DataRouteDict:
    """Return GRPO routes including reward records."""
    owner_key = rollout_owner_key(use_training_pipeline)
    return {
        **basic_policy_routes(use_training_pipeline),
        EnvResult: DataRoute(
            src="env",
            dst=None,
            via="storage_worker",
            owner_key=owner_key,
        ),
        RolloutResult: DataRoute(
            src="rollout",
            dst=None,
            via="storage_worker",
            owner_key=owner_key,
        ),
        RewardRequest: DataRoute(src="env", dst="reward", via="storage_worker"),
        RewardResult: DataRoute(
            src="reward",
            dst=None,
            via="storage_worker",
            owner_key=owner_key,
        ),
    }


@register_data_routes("dsrl")
def dsrl_data_routes(use_training_pipeline: bool = False) -> DataRouteDict:
    """Return DSRL routes for environment and rollout records."""
    owner_key = rollout_owner_key(use_training_pipeline)
    return {
        **basic_policy_routes(use_training_pipeline),
        EnvResult: DataRoute(
            src="env",
            dst=None,
            via="storage_worker",
            owner_key=owner_key,
        ),
        RolloutResult: DataRoute(
            src="rollout",
            dst=None,
            via="storage_worker",
            owner_key=owner_key,
        ),
    }


@register_data_routes("dagger")
def dagger_data_routes(use_training_pipeline: bool = False) -> DataRouteDict:
    """Return DAgger routes including episode collection records."""
    return {
        **basic_policy_routes(use_training_pipeline),
        LeRobotEpisodeBatch: DataRoute(
            src=None,
            dst="actor",
            via="storage_worker",
        ),
        LeRobotStepResult: DataRoute(
            src="env",
            dst=None,
            via="storage_worker",
            owner_key=lerobot_actor_owner_key,
        ),
        EnvResult: DataRoute(
            src="env",
            dst=None,
            via="storage_worker",
            owner_key=trajectory_batch_owner_key,
        ),
        RolloutResult: DataRoute(
            src="rollout",
            dst=None,
            via="storage_worker",
            owner_key=trajectory_batch_owner_key,
        ),
    }
