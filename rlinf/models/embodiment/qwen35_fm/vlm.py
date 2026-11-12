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

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


@dataclass
class VLMCondition:
    """Native multimodal features and optional final Key-to-Value memories.

    Shapes are hidden_states [B,L,C], attention_mask [B,L], and each
    gdn_states[layer] [B,value_heads,key_dim,value_dim].
    """

    hidden_states: torch.Tensor
    attention_mask: torch.Tensor
    gdn_states: dict[int, torch.Tensor] = field(default_factory=dict)
    intermediate_hidden_states: tuple[torch.Tensor, ...] | None = None


class Qwen35VLMWrapper(nn.Module):
    """Encode using the unmodified native Qwen3.5 multimodal backbone.

    Each encode call owns a fresh cache. Only recurrent matrices are returned;
    neither attention KV nor convolution buffers escape the call.
    """

    def __init__(
        self,
        model: nn.Module,
        processor: Any = None,
        trainable: bool = False,
        state_layers: list[int] | None = None,
        state_scope: str = "current_vlm_forward",
        lora_config: Any = None,
    ) -> None:
        super().__init__()
        if state_scope != "current_vlm_forward":
            raise ValueError(
                "Only current_vlm_forward is supported; persistent_stream requires "
                "an incremental observation and episode reset protocol."
            )
        self.model = model
        self.processor = processor
        if processor is not None:
            processor.tokenizer.padding_side = "left"
        self.trainable = trainable or lora_config is not None
        text_config = model.config.text_config
        self.gdn_layers = tuple(
            i
            for i, kind in enumerate(text_config.layer_types)
            if kind == "linear_attention"
        )
        self.state_layers = (
            self.gdn_layers if state_layers is None else tuple(state_layers)
        )
        if len(set(self.state_layers)) != len(self.state_layers) or any(
            i not in self.gdn_layers for i in self.state_layers
        ):
            raise ValueError(
                f"state_layers must be distinct GDN indices from {self.gdn_layers}, "
                f"got {self.state_layers}"
            )
        self.state_specs = dict.fromkeys(
            self.state_layers,
            (
                text_config.linear_num_value_heads,
                text_config.linear_key_head_dim,
                text_config.linear_value_head_dim,
            ),
        )
        model.requires_grad_(trainable)
        if lora_config is not None:
            from peft import get_peft_model

            # Adapter injection acts on the backbone we actually call below.
            self.model.model = get_peft_model(self.model.model, lora_config)
        if not self.trainable:
            self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        trainable: bool = False,
        state_layers: list[int] | None = None,
        state_scope: str = "current_vlm_forward",
        lora_config: Any = None,
        load_processor: bool = True,
        **model_kwargs: Any,
    ) -> "Qwen35VLMWrapper":
        """Load existing local assets without downloading or modifying weights."""
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        if model_kwargs.pop("local_files_only", True) is not True:
            raise ValueError("Qwen35VLMWrapper requires local_files_only=True")
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, local_files_only=True, **model_kwargs
        )
        processor = (
            AutoProcessor.from_pretrained(model_path, local_files_only=True)
            if load_processor
            else None
        )
        return cls(model, processor, trainable, state_layers, state_scope, lora_config)

    def train(self, mode: bool = True) -> "Qwen35VLMWrapper":
        """Keep a frozen pretrained backbone in evaluation mode."""
        super().train(mode)
        if not self.trainable:
            self.model.eval()
        return self

    def preprocess(
        self,
        observation: Any,
        instruction: str | list[str],
        **processor_kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Build native chat inputs; observations may be one or several images.

        For batched instructions, observation is a parallel list whose entries
        are images or image lists. Native video/other inputs can instead be
        supplied directly to encode as processed model keyword arguments.
        """
        if self.processor is None:
            raise ValueError("preprocess requires a native Qwen3.5 processor")
        instructions = [instruction] if isinstance(instruction, str) else instruction
        observations = [observation] if isinstance(instruction, str) else observation
        if len(observations) != len(instructions):
            raise ValueError("Expected one observation entry per instruction")
        conversations = []
        for images, text in zip(observations, instructions, strict=True):
            images = images if isinstance(images, (list, tuple)) else [images]
            content = [{"type": "image", "image": image} for image in images]
            content.append({"type": "text", "text": text})
            conversations.append([{"role": "user", "content": content}])
        return self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            **processor_kwargs,
        )

    def encode(
        self,
        *,
        observation: Any = None,
        instruction: str | list[str] | None = None,
        extract_states: bool = False,
        output_hidden_states: bool = False,
        validate_padding: bool = True,
        **model_inputs: Any,
    ) -> VLMCondition:
        """Return native features and, when requested, differentiable memories.

        Processed native keyword inputs pass through unchanged. External caches
        are rejected so complete observations cannot accumulate into old state.
        Trainable state extraction requires gradient checkpointing disabled.
        """
        if observation is not None or instruction is not None:
            if observation is None or instruction is None or model_inputs:
                raise ValueError(
                    "Supply both observation and instruction, or processed native inputs"
                )
            model_inputs = self.preprocess(observation, instruction)
            device = self.model.get_input_embeddings().weight.device
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        if model_inputs.pop("past_key_values", None) is not None:
            raise ValueError(
                "encode owns a fresh cache; past_key_values is not accepted"
            )
        if model_inputs.pop("use_cache", extract_states) != extract_states:
            raise ValueError("use_cache is controlled by extract_states")
        mask = model_inputs.get("attention_mask")
        if extract_states and mask is not None and validate_padding:
            if not isinstance(mask, torch.Tensor) or mask.ndim != 2:
                raise ValueError("GDN state extraction requires a 2-D token mask")
            if bool((mask[:, :-1].bool() & ~mask[:, 1:].bool()).any()):
                raise ValueError(
                    "GDN state extraction requires left padding: trailing padding "
                    "would update the final recurrent state after the last real token"
                )
        if (
            extract_states
            and self.training
            and self.trainable
            and any(
                getattr(module, "gradient_checkpointing", False)
                for module in self.model.modules()
            )
        ):
            raise ValueError(
                "Disable gradient checkpointing for differentiable GDN state extraction"
            )
        with nullcontext() if self.trainable else torch.no_grad():
            output = self.model.model(
                **model_inputs,
                use_cache=extract_states,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
        hidden = output.last_hidden_state
        mask = model_inputs.get("attention_mask")
        if mask is None:
            mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
        if not isinstance(mask, torch.Tensor) or mask.ndim != 2:
            raise ValueError("Action conditioning requires a 2-D token attention_mask")
        states = {}
        if extract_states:
            cache = output.past_key_values
            for index in self.state_layers:
                if hasattr(cache, "recurrent_states"):
                    state = cache.recurrent_states[index]
                else:
                    state = cache.layers[index].recurrent_states[0]
                if state is None or tuple(state.shape[1:]) != self.state_specs[index]:
                    raise RuntimeError(
                        f"Unexpected native recurrent state at layer {index}: "
                        f"expected [B,{self.state_specs[index]}], "
                        f"got {None if state is None else tuple(state.shape)}"
                    )
                # The cache is local and never reused; preserving this tensor
                # keeps gradients to the native GDN projections intact.
                states[index] = state
        return VLMCondition(
            hidden, mask, states, getattr(output, "hidden_states", None)
        )

    def encode_with_states(self, **model_inputs: Any) -> VLMCondition:
        """Encode one independent multimodal input and return final GDN states."""
        return self.encode(extract_states=True, **model_inputs)

    def forward(self, **model_inputs: Any) -> VLMCondition:
        """Alias encode for ordinary module composition."""
        return self.encode(**model_inputs)
