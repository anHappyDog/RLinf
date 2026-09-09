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

import math
from collections.abc import Iterable
from contextlib import nullcontext
from typing import ContextManager, Union

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard
from torch.optim import Optimizer

from rlinf.config import torch_dtype_from_precision
from rlinf.hybrid_engines.fsdp import (
    CPUOffloadPolicy,
    FSDPModule,
    MixedPrecisionPolicy,
    OffloadPolicy,
)
from rlinf.hybrid_engines.fsdp.strategy.base import FSDPStrategyBase
from rlinf.hybrid_engines.fsdp.utils import (
    FSDPVersion,
    apply_fsdp2_to_model,
    clip_grad_by_total_norm_,
)
from rlinf.hybrid_engines.fsdp.utils import (
    get_grad_norm as get_fsdp_grad_norm,
)
from rlinf.utils.utils import clear_memory


class FSDP2Strategy(FSDPStrategyBase):
    def wrap_model(self, model: nn.Module, device_mesh: DeviceMesh) -> FSDPModule:
        """
        Wrap the model with FSDP2's fully_shard.

        Args:
            - model (nn.Module): The model to be wrapped.
            - device_mesh (DeviceMesh): The device mesh for FSDP2.

        Returns:
            - FSDPModule: The FSDP2 wrapped model.
        """
        mixed_precision_config = self.cfg.fsdp_config.mixed_precision
        param_dtype = torch_dtype_from_precision(mixed_precision_config.param_dtype)
        reduce_dtype = torch_dtype_from_precision(mixed_precision_config.reduce_dtype)
        cast_forward_inputs = mixed_precision_config.get("cast_forward_inputs", True)

        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            cast_forward_inputs=cast_forward_inputs,
        )

        offload_policy = (
            CPUOffloadPolicy(pin_memory=self.cfg.fsdp_config.offload_pin_memory)
            if self.cfg.fsdp_config.cpu_offload
            else OffloadPolicy()
        )

        fsdp2_model: FSDPModule = apply_fsdp2_to_model(
            module=model,
            config=self.cfg.fsdp_config,
            device_mesh=device_mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            reshard_after_forward=self.cfg.fsdp_config.reshard_after_forward,
        )

        return fsdp2_model

    @classmethod
    def get_fsdp_version(cls) -> FSDPVersion:
        return FSDPVersion.FSDP2

    @torch.no_grad()
    def onload_param_and_grad(
        self, model: FSDPModule, device: torch.device, onload_grad: bool
    ) -> None:
        """
        Load model parameters and gradients to the specified device.

        Args:
            - model (FSDPModule): The FSDP2 wrapped model.
            - device (torch.device): The target device.
            - onload_grad (bool): Whether to load gradients or not.
        """
        model.to(device=device)
        if onload_grad:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = param.grad.to(device)
        clear_memory()

    @torch.no_grad()
    def offload_param_and_grad(self, model: FSDPModule, offload_grad: bool) -> None:
        """
        Offload model parameters and gradients to CPU.

        Args:
            - model (FSDPModule): The FSDP2 wrapped model.
            - offload_grad (bool): Whether to offload gradients or not.
        """
        model.to(device="cpu")

        if offload_grad:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = param.grad.cpu()
        clear_memory()

    @torch.no_grad()
    def offload_optimizer(self, optimizer: Optimizer) -> None:
        """
        Offload optimizer states to CPU.

        Args:
            - optimizer (Optimizer): The optimizer.
        """
        for st in optimizer.state.values():
            if not isinstance(st, dict):
                continue
            for k, v in list(st.items()):
                if torch.is_tensor(v):
                    if v.device.type != "cpu":
                        st[k] = v.detach().to("cpu", non_blocking=True)
                        del v
        clear_memory()

    @torch.no_grad()
    def onload_optimizer(self, optimizer: Optimizer, device: torch.device) -> None:
        """
        Load optimizer states to the specified device.

        Args:
            - optimizer (Optimizer): The optimizer.
            - device (torch.device): The target device.
        """
        for st in optimizer.state.values():
            if not isinstance(st, dict):
                continue
            for k, v in list(st.items()):
                if torch.is_tensor(v):
                    if v.device != device:
                        st[k] = v.detach().to(device, non_blocking=True)
                        del v
        clear_memory()

    def get_grad_norm(
        self,
        model: FSDPModule,
        parameters: Iterable[torch.nn.Parameter],
        norm_type: Union[float, int] = 2.0,
    ) -> float:
        """Return the global norm of a parameter subset's gradients."""
        del model
        parameters = list(parameters)
        sharded_parameters = [
            parameter
            for parameter in parameters
            if isinstance(parameter, DTensor)
            and any(isinstance(placement, Shard) for placement in parameter.placements)
        ]
        sharded_parameter_ids = {id(parameter) for parameter in sharded_parameters}
        replicated_parameters = [
            parameter
            for parameter in parameters
            if id(parameter) not in sharded_parameter_ids
        ]
        sharded_norm = get_fsdp_grad_norm(
            sharded_parameters,
            dp_group=self._dp_group,
            norm_type=norm_type,
        )
        replicated_norm = get_fsdp_grad_norm(
            replicated_parameters,
            dp_group=None,
            norm_type=norm_type,
        )
        norm_type = float(norm_type)
        if norm_type == torch.inf:
            return max(sharded_norm, replicated_norm)
        return math.pow(
            math.pow(sharded_norm, norm_type) + math.pow(replicated_norm, norm_type),
            1.0 / norm_type,
        )

    def clip_grad_norm_(
        self,
        model: FSDPModule,
        parameters: Iterable[torch.nn.Parameter] | None = None,
        max_norm: float | None = None,
        norm_type: Union[float, int] = 2.0,
    ) -> float:
        """
        Clip the gradients of the model parameters by total norm.

        Args:
            - model (FSDPModule): The FSDP2 wrapped model.
            - parameters: Parameters to include. Defaults to all model parameters.
            - max_norm: Maximum gradient norm. Defaults to ``optim.clip_grad``.
            - norm_type (float): The type of the used p-norm.

        Returns:
            - float: The total norm of the gradients before clipping.
        """
        parameters = list(model.parameters() if parameters is None else parameters)
        max_norm = self.cfg.optim.clip_grad if max_norm is None else max_norm
        grad_norm = self.get_grad_norm(model, parameters, norm_type)
        clip_grad_by_total_norm_(
            parameters,
            max_grad_norm=max_norm,
            total_norm=grad_norm,
        )
        return grad_norm

    def before_micro_batch(
        self, model: FSDPModule, is_last_micro_batch: bool
    ) -> ContextManager:
        """
        Context manager to control gradient synchronization for FSDP2.
        FSDP2 does not provide model.no_sync, but provides set_requires_gradient_sync.

        Args:
            - model (FSDPModule): The FSDP2 wrapped model.
            - is_last_micro_batch (bool): Whether this is the last micro batch.

        Returns:
            - ContextManager: nullcontext, just for interface consistency.
        """
        if not self.cfg.fsdp_config.enable_gradient_accumulation:
            return nullcontext()
        if is_last_micro_batch:
            model.set_requires_gradient_sync(True)
        else:
            model.set_requires_gradient_sync(False)
        return nullcontext()
