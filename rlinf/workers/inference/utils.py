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
from typing import TYPE_CHECKING, Optional, Union

import torch
from omegaconf import DictConfig
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed.tensor import DTensor

if TYPE_CHECKING:
    from rlinf.workers.inference.fsdp_inference_worker import FSDPInference
    from rlinf.workers.inference.megatron_inference_worker import MegatronInference


def get_inference_backend_worker(
    cfg: DictConfig,
) -> Union["FSDPInference", "MegatronInference"]:
    """Get the inference backend worker class based on the training backend.

    Args:
        cfg (DictConfig): Configuration for the inference task.

    Returns:
        Inference worker class.
    """
    training_backend = cfg.actor.training_backend
    if training_backend == "megatron":
        from rlinf.workers.inference.megatron_inference_worker import (
            MegatronInference,
        )

        return MegatronInference
    elif training_backend == "fsdp":
        from rlinf.workers.inference.fsdp_inference_worker import FSDPInference

        return FSDPInference
    else:
        raise ValueError(
            f"Unsupported training backend for inference: {training_backend}"
        )


def _ceil_div(a: int, b: int) -> int:
    """
    Ceiling division of a by b.
    """
    return (a + b - 1) // b


def _get_local_tensor(
    obj: Union[torch.Tensor, DTensor, ShardedTensor],
) -> Optional[torch.Tensor]:
    """
    Get the local tensor from a ShardedTensor or DTensor.
    If the input is a regular tensor, return it directly.
    Returns None if there is no local shard.

    Args:
        obj: The input object which can be a torch.Tensor, ShardedTensor, or DTensor.

    Returns:
        The local tensor if available, otherwise None.
    """
    try:
        if isinstance(obj, ShardedTensor):
            shards = obj.local_shards()
            if not shards:
                return None
            return shards[0].tensor
    except Exception:
        pass
    try:
        if isinstance(obj, DTensor):
            return obj.to_local()
    except Exception:
        pass
    if isinstance(obj, torch.Tensor):
        return obj
    raise TypeError(f"Unsupported state_dict value type: {type(obj)}")


def _get_full_numel(obj: Union[torch.Tensor, ShardedTensor, DTensor]) -> int:
    """
    Get the total number of elements in the full tensor represented by the object.

    Args:
        obj: The input object which can be a torch.Tensor, ShardedTensor, or DTensor.

    Returns:
        The total number of elements in the full tensor.
    """
    if hasattr(obj, "size"):
        shape = obj.size()
    else:
        shape = obj.shape
    n = 1
    for d in shape:
        n *= int(d)
    return int(n)


def _shard_range_1d(full_numel: int, ws: int, rank: int) -> tuple[int, int, int]:
    """
    Compute the start and end indices for a 1D sharded tensor.

    Args:
        full_numel: Total number of elements in the full tensor.
        ws: World size (number of shards).
        rank: Rank of the current shard.

    Returns:
        A tuple of (start_index, end_index, shard_size).
    """
    shard_size = _ceil_div(full_numel, ws)
    start = rank * shard_size
    end = min(start + shard_size, full_numel)
    return start, end, shard_size
