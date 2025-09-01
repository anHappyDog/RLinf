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

from typing import Optional

import torch
from torch.nn import Parameter
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

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


original_vocab_loader = VocabParallelEmbedding.weight_loader


def vocab_loader(
    self: VocabParallelEmbedding, param: Parameter, loaded_weight: torch.Tensor
):
    output_dim = getattr(param, "output_dim", 0)

    full_vocab_size = getattr(self, "org_vocab_size", self.num_embeddings)
    partition_size = self.num_embeddings_per_partition
    loaded_vocab_size = loaded_weight.shape[output_dim]
    print(
        f"loaded vocab size {loaded_vocab_size}, partition_size:{partition_size}, full_vocab_size:{full_vocab_size}, tp_rank:{get_tensor_model_parallel_rank()}",
        flush=True,
    )
    if loaded_vocab_size == partition_size:
        assert param.data.shape[output_dim] == partition_size, (
            f"Parameter shard size mismatch. Expected {partition_size}, got {param.data.shape[output_dim]}"
        )
        param.data.copy_(loaded_weight)
        return

    elif loaded_vocab_size == full_vocab_size:
        original_vocab_loader(self, param, loaded_weight)
        return

    elif loaded_vocab_size < partition_size:
        param.data.zero_()
        param.data[:loaded_vocab_size].copy_(loaded_weight)
        return

    else:
        raise ValueError(
            f"Shape mismatch in VocabParallelEmbedding loader for TP rank {get_tensor_model_parallel_rank()}. "
            f"Loaded weight vocab size ({loaded_vocab_size}) does not match "
            f"the expected partition size ({partition_size}) or the full vocab size ({full_vocab_size})."
        )


VocabParallelEmbedding.weight_loader = vocab_loader

original_column_loader = ColumnParallelLinear.weight_loader


def surgical_column_loader(
    self: ColumnParallelLinear, param: Parameter, loaded_weight: torch.Tensor
):
    tp_rank = get_tensor_model_parallel_rank()
    output_dim = getattr(param, "output_dim", 0)
    param_data = param.data
    shard_size = param_data.shape[output_dim]
    print(
        f"tp_rank:{tp_rank}, output_dim:{output_dim}, param_data.shape:{param_data.shape}, shard_size:{shard_size}, loaded_weight.shape:{loaded_weight.shape}"
    )
    if loaded_weight.shape[output_dim] == shard_size:
        param_data.copy_(loaded_weight)
    elif loaded_weight.shape[output_dim] > shard_size:
        start_idx = tp_rank * shard_size
        print(
            f"loaded_weight shape {loaded_weight.shape[output_dim]} is greater than shard_size:{shard_size},start_idx:{start_idx},shard_size:{shard_size},output_dim:{output_dim}"
        )
        loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        assert param_data.shape == loaded_weight.shape, (
            f"Shape mismatch in ColumnParallel: {param_data.shape} vs {loaded_weight.shape}"
        )
        param_data.copy_(loaded_weight)
    else:
        original_column_loader(self, param, loaded_weight)


ColumnParallelLinear.weight_loader = surgical_column_loader

original_row_loader = RowParallelLinear.weight_loader


def surgical_row_loader(
    self: RowParallelLinear, param: Parameter, loaded_weight: torch.Tensor
):
    tp_rank = get_tensor_model_parallel_rank()
    input_dim = getattr(param, "input_dim", 1)
    param_data = param.data
    shard_size = param_data.shape[input_dim]

    if loaded_weight.shape[input_dim] == shard_size:
        param_data.copy_(loaded_weight)
    elif loaded_weight.shape[input_dim] > shard_size:
        start_idx = tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(input_dim, start_idx, shard_size)
        assert param_data.shape == loaded_weight.shape, (
            f"Shape mismatch in RowParallel: {param_data.shape} vs {loaded_weight.shape}"
        )
        param_data.copy_(loaded_weight)
    else:
        original_row_loader(self, param, loaded_weight)


RowParallelLinear.weight_loader = surgical_row_loader

original_merged_column_loader = MergedColumnParallelLinear.weight_loader


def surgical_merged_column_loader(
    self: MergedColumnParallelLinear,
    param: Parameter,
    loaded_weight: torch.Tensor,
    loaded_shard_id: Optional[int] = None,
):
    if loaded_shard_id is None:
        return original_merged_column_loader(
            self, param, loaded_weight, loaded_shard_id
        )

    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()
    output_dim = getattr(param, "output_dim", 0)

    shard_offset = sum(self.output_sizes[:loaded_shard_id]) // tp_size
    shard_size = self.output_sizes[loaded_shard_id] // tp_size
    param_data = param.data.narrow(output_dim, shard_offset, shard_size)

    if loaded_weight.shape[output_dim] == shard_size:
        param_data.copy_(loaded_weight)
    elif loaded_weight.shape[output_dim] > shard_size:
        start_idx = tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        assert param_data.shape == loaded_weight.shape, (
            f"Shape mismatch in MergedColumn: {param_data.shape} vs {loaded_weight.shape}"
        )
        param_data.copy_(loaded_weight)
    else:
        original_merged_column_loader(self, param, loaded_weight, loaded_shard_id)


MergedColumnParallelLinear.weight_loader = surgical_merged_column_loader

original_qkv_loader = QKVParallelLinear.weight_loader


def surgical_qkv_loader(
    self: QKVParallelLinear,
    param: Parameter,
    loaded_weight: torch.Tensor,
    loaded_shard_id: Optional[str] = None,
):
    if loaded_shard_id is None:
        return original_qkv_loader(self, param, loaded_weight, loaded_shard_id)

    tp_rank = get_tensor_model_parallel_rank()

    output_dim = getattr(param, "output_dim", 0)

    if loaded_shard_id == "q":
        shard_offset = 0
        shard_size = self.num_heads * self.head_size
    elif loaded_shard_id == "k":
        shard_offset = self.num_heads * self.head_size
        shard_size = self.num_kv_heads * self.head_size
    elif loaded_shard_id == "v":  # "v"
        shard_offset = (self.num_heads + self.num_kv_heads) * self.head_size
        shard_size = self.num_kv_heads * self.head_size
    else:
        raise ValueError(
            f"Invalid shard_id {loaded_shard_id} in QKVParallelLinear loader."
        )

    param_data = param.data.narrow(output_dim, shard_offset, shard_size)

    if loaded_weight.shape[output_dim] == shard_size:
        param_data.copy_(loaded_weight)
    elif loaded_weight.shape[output_dim] > shard_size:
        if loaded_shard_id == "q":
            shard_id = tp_rank
        else:
            num_kv_head_replicas = getattr(self, "num_kv_head_replicas", 1)
            shard_id = tp_rank // num_kv_head_replicas

        start_idx = shard_id * shard_size
        loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        assert param_data.shape == loaded_weight.shape, (
            f"Shape mismatch in QKV: {param_data.shape} vs {loaded_weight.shape}"
        )
        param_data.copy_(loaded_weight)
    else:
        original_qkv_loader(self, param, loaded_weight, loaded_shard_id)


QKVParallelLinear.weight_loader = surgical_qkv_loader
