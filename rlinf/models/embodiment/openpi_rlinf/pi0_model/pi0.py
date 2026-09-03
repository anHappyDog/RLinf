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

"""Pi0 model for PyTorch, aligned with JAX models/pi0.py.

Flow matching model for continuous action generation.
"""

from __future__ import annotations

import dataclasses
import logging

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import gemma, model, pointnet, short_memory, siglip
from .pi0_config import Pi0Config
from .utils import _str_to_dtype

logger = logging.getLogger("openpi")


@dataclasses.dataclass(frozen=True)
class TextLossOutput:
    """Token-level metrics for a PaliGemma text training batch."""

    loss: torch.Tensor
    token_accuracy: torch.Tensor
    token_count: torch.Tensor


def make_attn_mask(input_mask: torch.Tensor, mask_ar: torch.Tensor) -> torch.Tensor:
    """Create attention mask from input mask and autoregressive mask.

    Tokens can attend to valid input tokens which have a cumulative mask_ar
    smaller or equal to theirs.

    Args:
        input_mask: bool[B, N] - true if token is valid
        mask_ar: bool[N] - true where next token starts a new autoregressive block
    """
    mask_ar = mask_ar.expand(input_mask.shape[0], -1)
    cumsum = torch.cumsum(mask_ar.int(), dim=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return torch.logical_and(attn_mask, valid_mask)


def posemb_sincos(
    pos: torch.Tensor,
    embedding_dim: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
) -> torch.Tensor:
    """Sine-cosine positional embedding for scalar positions.

    Args:
        pos: (B,) float positions
        embedding_dim: output dimension (must be even)

    Returns:
        (B, embedding_dim) positional embedding
    """
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = torch.linspace(
        0.0, 1.0, embedding_dim // 2, device=pos.device, dtype=torch.float32
    )
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = torch.einsum("i,j->ij", pos.float(), 1.0 / period * 2 * torch.pi)
    # Match JAX which keeps posemb in float32. However, PT Linear does not support
    # mixed float32/bf16 matmul, so cast back to the model's embed_dtype.
    # The caller should upcast to float32 if needed for high-precision ops.
    return torch.cat([torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=-1).to(
        pos.dtype
    )


class Pi0(model.BaseModel):
    """Pi0 flow matching model for continuous action generation."""

    def __init__(self, config: Pi0Config):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.pcd = config.pcd
        self.embed_dtype = _str_to_dtype(config.dtype)
        self._config = config

        paligemma_config = gemma.get_config(config.paligemma_variant)
        action_expert_config = gemma.get_config(config.action_expert_variant)

        # Gemma LLM with dual experts
        # Expert 0 (PaliGemma) uses regular RMSNorm; Expert 1 (Action Expert) may use adaRMS
        adarms = [False, config.pi05]
        self.llm = gemma.Module(
            configs=[paligemma_config, action_expert_config],
            embed_dtype=config.dtype,
            adarms=adarms,
            use_gradient_checkpointing=False,
        )

        # SigLIP vision encoder
        self.img = siglip.SigLIPViT(
            variant="So400m/14",
            pool_type="none",
            num_classes=paligemma_config.width,
            use_gradient_checkpointing=False,
            dtype_mm=config.dtype,
        )
        self.short_memory_encoder = None
        self.history_state_encoder = None
        if config.short_memory:
            self.short_memory_encoder = short_memory.ShortMemoryVisionEncoder(
                temporal_layers=config.short_memory_temporal_layers,
                drop_history_layer=config.short_memory_drop_history_layer,
            )
            self.history_state_encoder = short_memory.HistoricalStateEncoder(
                state_dim=config.short_memory_state_dim,
                output_dim=paligemma_config.width,
            )

        action_expert_width = action_expert_config.width
        self.action_dim = config.action_dim

        # Action input projection
        self.action_in_proj = nn.Linear(config.action_dim, action_expert_width)

        if config.pi05:
            self.time_mlp_in = nn.Linear(action_expert_width, action_expert_width)
            self.time_mlp_out = nn.Linear(action_expert_width, action_expert_width)
        else:
            self.state_proj = nn.Linear(config.action_dim, action_expert_width)
            self.action_time_mlp_in = nn.Linear(
                2 * action_expert_width, action_expert_width
            )
            self.action_time_mlp_out = nn.Linear(
                action_expert_width, action_expert_width
            )

        # Action output projection
        self.action_out_proj = nn.Linear(action_expert_width, config.action_dim)

        # Optional PointNet
        if config.pcd:
            pointnet_config = pointnet.get_config(config.pointnet_variant)
            self.pointnet = pointnet.UncoloredPointNet(
                n_coordinates=pointnet_config.n_coordinates,
                output_dim=pointnet_config.output_dim,
                hidden_dim=pointnet_config.hidden_dim,
                hidden_depth=pointnet_config.hidden_depth,
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize projection weights."""
        nn.init.normal_(self.action_in_proj.weight, std=0.02)
        nn.init.zeros_(self.action_in_proj.bias)
        nn.init.normal_(self.action_out_proj.weight, std=0.02)
        nn.init.zeros_(self.action_out_proj.bias)

        if self.pi05:
            nn.init.normal_(self.time_mlp_in.weight, std=0.02)
            nn.init.zeros_(self.time_mlp_in.bias)
            nn.init.normal_(self.time_mlp_out.weight, std=0.02)
            nn.init.zeros_(self.time_mlp_out.bias)
        else:
            nn.init.normal_(self.state_proj.weight, std=0.02)
            nn.init.zeros_(self.state_proj.bias)
            nn.init.normal_(self.action_time_mlp_in.weight, std=0.02)
            nn.init.zeros_(self.action_time_mlp_in.bias)
            nn.init.normal_(self.action_time_mlp_out.weight, std=0.02)
            nn.init.zeros_(self.action_time_mlp_out.bias)

    def embed_prefix(
        self, obs: model.Observation
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed the prefix (images + language + optional point cloud).

        Returns:
            tokens: (B, S, emb_dim) embedded tokens
            input_mask: (B, S) mask of valid tokens
            ar_mask: (S,) autoregressive mask (all False for prefix)
        """
        tokens = []
        input_mask = []
        ar_mask = []

        # Embed images through SigLIP
        for name in obs.images:
            images = obs.images[name]
            if images.ndim == 5:
                if self.short_memory_encoder is None:
                    raise ValueError(
                        "Video observations require Pi0Config(short_memory=True)."
                    )
                image_tokens = self.short_memory_encoder(
                    self.img,
                    images,
                    frame_mask=obs.image_masks[name],
                    time_offsets=obs.history_time_offsets,
                )
                image_mask = obs.image_masks[name][:, -1]
            else:
                image_tokens, _ = self.img(images)
                image_mask = obs.image_masks[name]
            tokens.append(image_tokens)

            # Image tokens use bidirectional attention
            input_mask.append(
                einops.repeat(image_mask, "b -> b s", s=image_tokens.shape[1])
            )
            ar_mask += [False] * image_tokens.shape[1]

        if obs.history_states is not None:
            if self.history_state_encoder is None:
                raise ValueError("history_states require Pi0Config(short_memory=True).")
            if obs.history_frame_mask is None or obs.history_time_offsets is None:
                raise ValueError(
                    "history_states require history_frame_mask and "
                    "history_time_offsets."
                )
            state_tokens, state_mask = self.history_state_encoder(
                obs.history_states,
                frame_mask=obs.history_frame_mask,
                time_offsets=obs.history_time_offsets,
            )
            tokens.append(state_tokens)
            input_mask.append(state_mask)
            ar_mask += [False] * state_tokens.shape[1]

        # Add language tokens
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.llm.embed(obs.tokenized_prompt)
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]

        # Add point cloud tokens
        if self.pcd and obs.pcd_xyz is not None:
            # pcd_xyz: (B, 16, 2025, 3)
            # PointNet expects (B, num_points, 3)
            B = obs.pcd_xyz.shape[0]
            pcd_flat = obs.pcd_xyz.reshape(B, -1, 3)  # (B, 16*2025, 3)
            pcd_tokens = self.pointnet(pcd_flat)  # (B, 16, 2048)
            # Reshape to match expected dimensions
            if pcd_tokens.dim() == 2:
                pcd_tokens = pcd_tokens.unsqueeze(1)  # (B, 1, 2048)

            tokens.append(pcd_tokens)
            input_mask.append(
                torch.ones(
                    pcd_tokens.shape[:2], dtype=torch.bool, device=pcd_tokens.device
                )
            )
            ar_mask += [False] * pcd_tokens.shape[1]

        tokens = torch.cat(tokens, dim=1)
        input_mask = torch.cat(input_mask, dim=1)
        ar_mask = torch.tensor(ar_mask, device=tokens.device)
        return tokens, input_mask, ar_mask

    def embed_text_inputs(
        self, obs: model.Observation
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Embed image and text tokens with a per-token autoregressive mask.

        The ordinary action path treats the entire language prompt as a
        bidirectional prefix. Text training additionally marks every response
        token as a new autoregressive block through ``token_ar_mask``. Keeping
        this as a separate path preserves the existing action-prefix behavior.

        Returns:
            Embedded tokens, input mask, autoregressive mask, and the number of
            image tokens prepended to the text sequence.
        """
        if self.pcd:
            raise NotImplementedError(
                "Text training is not implemented for point-cloud Pi0 models."
            )
        if obs.tokenized_prompt is None or obs.tokenized_prompt_mask is None:
            raise ValueError("Text inputs require tokenized_prompt and its mask.")
        if obs.token_ar_mask is None:
            raise ValueError("Text inputs require token_ar_mask.")
        if obs.tokenized_prompt.shape != obs.tokenized_prompt_mask.shape:
            raise ValueError(
                "tokenized_prompt and tokenized_prompt_mask must have the same shape."
            )
        if obs.tokenized_prompt.shape != obs.token_ar_mask.shape:
            raise ValueError(
                "tokenized_prompt and token_ar_mask must have the same shape."
            )

        tokens, input_mask, _ = self.embed_prefix(obs)
        text_length = obs.tokenized_prompt.shape[1]
        image_token_count = tokens.shape[1] - text_length
        image_ar_mask = torch.zeros(
            tokens.shape[0],
            image_token_count,
            dtype=obs.token_ar_mask.dtype,
            device=tokens.device,
        )
        ar_mask = torch.cat(
            [image_ar_mask, obs.token_ar_mask.to(device=tokens.device)], dim=1
        )
        return tokens, input_mask, ar_mask, image_token_count

    def compute_text_loss(
        self,
        observation: model.Observation,
        *,
        train: bool = False,
        rng: torch.Generator | None = None,
    ) -> TextLossOutput:
        """Compute causal next-token loss on the masked text response.

        ``tokenized_prompt`` contains the full ``prefix + response + EOS``
        sequence. Each token predicts the following token, so model inputs and
        labels are shifted by one. ``token_loss_mask`` selects response and EOS
        labels only. Vocabulary logits are materialized only for selected
        positions because the PaliGemma vocabulary projection is large.
        """
        if observation.tokenized_prompt is None:
            raise ValueError("Text loss requires tokenized_prompt.")
        if observation.token_loss_mask is None:
            raise ValueError("Text loss requires token_loss_mask.")
        if observation.tokenized_prompt.shape != observation.token_loss_mask.shape:
            raise ValueError(
                "tokenized_prompt and token_loss_mask must have the same shape."
            )
        if observation.tokenized_prompt.shape[1] < 2:
            raise ValueError("Text loss requires at least two token positions.")

        observation = model.preprocess_observation(observation, train=train, rng=rng)
        observation = model._observation_to_dtype(observation, self.embed_dtype)
        tokens, input_mask, ar_mask, image_token_count = self.embed_text_inputs(
            observation
        )

        # The last text token has no next-token label, matching the official
        # Pi0-FAST training convention.
        input_tokens = tokens[:, :-1]
        input_mask = input_mask[:, :-1]
        ar_mask = ar_mask[:, :-1]
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = torch.cumsum(input_mask.int(), dim=1) - 1

        text_output = self.llm(
            [input_tokens, None],
            positions=positions,
            mask=attn_mask,
        )[0][0][:, image_token_count:]

        labels = observation.tokenized_prompt[:, 1:]
        loss_mask = observation.token_loss_mask[:, 1:].bool()
        loss_mask = torch.logical_and(
            loss_mask, observation.tokenized_prompt_mask[:, 1:].bool()
        )
        token_count = loss_mask.sum()
        if token_count.item() == 0:
            raise ValueError("Text loss mask selects no response tokens.")

        selected_output = text_output[loss_mask]
        selected_labels = labels[loss_mask]
        decoder_dtype = self.llm.embedder.embedding.weight.dtype
        logits = self.llm.embedder.decode(selected_output.to(decoder_dtype))
        loss = F.cross_entropy(logits.float(), selected_labels)
        token_accuracy = (logits.argmax(dim=-1) == selected_labels).float().mean()
        return TextLossOutput(
            loss=loss,
            token_accuracy=token_accuracy,
            token_count=token_count,
        )

    @torch.no_grad()
    def generate_text(
        self,
        observation: model.Observation,
        *,
        eos_token_id: int,
        max_new_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Greedily generate response tokens with a cached image-text prefix.

        Returns:
            A pair ``(generated_tokens, generated_mask)`` with shape
            ``[batch, max_new_tokens]``. The mask is true through each sample's
            EOS token and false for unused capacity.
        """
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")
        if observation.tokenized_prompt is None:
            raise ValueError("Text generation requires tokenized_prompt.")
        if observation.tokenized_prompt_mask is None:
            raise ValueError("Text generation requires tokenized_prompt_mask.")
        if observation.token_ar_mask is None:
            raise ValueError("Text generation requires token_ar_mask.")

        observation = model.preprocess_observation(observation, train=False)
        observation = model._observation_to_dtype(observation, self.embed_dtype)
        prompt_mask = observation.tokenized_prompt_mask.bool()
        lengths = prompt_mask.sum(dim=1).long()
        if torch.any(lengths == 0):
            raise ValueError("Text generation requires a non-empty prefix.")

        batch_size, sequence_capacity = observation.tokenized_prompt.shape
        generated = torch.zeros(
            batch_size,
            max_new_tokens,
            dtype=observation.tokenized_prompt.dtype,
            device=observation.tokenized_prompt.device,
        )
        generated_mask = torch.zeros(
            batch_size,
            max_new_tokens,
            dtype=torch.bool,
            device=generated.device,
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=generated.device)

        active_length = int(lengths.max().item())
        prefix_observation = dataclasses.replace(
            observation,
            tokenized_prompt=observation.tokenized_prompt[:, :active_length],
            tokenized_prompt_mask=prompt_mask[:, :active_length],
            token_ar_mask=observation.token_ar_mask[:, :active_length],
            token_loss_mask=None,
        )
        tokens, cache_mask, token_ar_mask, image_token_count = self.embed_text_inputs(
            prefix_observation
        )
        if cache_mask.ndim != 2:
            raise ValueError(
                f"Expected a [batch, sequence] prefix mask, got {cache_mask.shape}."
            )
        attn_mask = make_attn_mask(cache_mask, token_ar_mask)
        positions = torch.cumsum(cache_mask.int(), dim=1) - 1
        outputs, kv_cache = self.llm(
            [tokens, None],
            positions=positions,
            mask=attn_mask,
        )
        batch_indices = torch.arange(batch_size, device=generated.device)
        last_hidden = outputs[0][batch_indices, image_token_count + lengths - 1]
        decoder_dtype = self.llm.embedder.embedding.weight.dtype

        for step in range(max_new_tokens):
            can_append = lengths < sequence_capacity
            active = torch.logical_and(~finished, can_append).reshape(batch_size)
            if not torch.any(active):
                break

            next_tokens = self.llm.embedder.decode(
                last_hidden.to(decoder_dtype)
            ).argmax(dim=-1)
            next_tokens = torch.where(active, next_tokens, 0)

            active_indices = torch.nonzero(active, as_tuple=False).squeeze(1)
            active_tokens = next_tokens[active_indices]
            lengths[active_indices] += 1
            generated[active_indices, step] = active_tokens
            generated_mask[active_indices, step] = True
            finished[active_indices] = active_tokens == eos_token_id

            if step + 1 == max_new_tokens or not torch.any(
                torch.logical_and(~finished, lengths < sequence_capacity)
            ):
                continue

            token_positions = cache_mask.sum(dim=1, keepdim=True)
            cache_mask = torch.cat([cache_mask, active[:, None]], dim=1)
            step_attn_mask = cache_mask[:, None, :]
            token_embeddings = self.llm.embed(next_tokens[:, None])
            outputs, kv_cache = self.llm(
                [token_embeddings, None],
                positions=token_positions,
                mask=step_attn_mask,
                kv_cache=kv_cache,
            )
            last_hidden = outputs[0][:, 0]

        return generated, generated_mask

    def embed_suffix(
        self,
        obs: model.Observation,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Embed the suffix (state + noisy actions + time embedding).

        Args:
            obs: observation
            noisy_actions: (B, action_horizon, action_dim)
            timestep: (B,) float timestep values

        Returns:
            tokens: (B, S, emb_dim)
            input_mask: (B, S)
            ar_mask: (S,)
            adarms_cond: (B, emb_dim) or None
        """
        input_mask = []
        tokens = []

        B = noisy_actions.shape[0]

        # Cast to embed_dtype to ensure consistency between training and inference
        # (during training FSDP2 casts forward inputs, but inference may pass float32).
        noisy_actions = noisy_actions.to(self.embed_dtype)
        timestep = timestep.to(self.embed_dtype)

        if not self.pi05:
            # Add a single state token
            # Eval transforms keep state in fp32 after normalization, while
            # the model is commonly loaded directly in bf16. Match the state
            # projection input to its compute dtype just like actions/time.
            state_token = self.state_proj(obs.state.to(self.embed_dtype))[:, None, :]
            tokens.append(state_token)
            input_mask.append(
                torch.ones(B, 1, dtype=torch.bool, device=state_token.device)
            )

        # Embed actions
        action_tokens = self.action_in_proj(noisy_actions)

        # Time embedding
        time_emb = posemb_sincos(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0
        )

        if self.pi05:
            # Time MLP for adaRMS conditioning
            time_emb = self.time_mlp_in(time_emb)
            time_emb = F.silu(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = F.silu(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # Mix timestep + action through MLP
            time_tokens = einops.repeat(
                time_emb, "b emb -> b s emb", s=self.action_horizon
            )
            action_time_tokens = torch.cat([action_tokens, time_tokens], dim=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = F.silu(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None

        tokens.append(action_expert_tokens)
        input_mask.append(
            torch.ones(
                action_expert_tokens.shape[:2],
                dtype=torch.bool,
                device=action_expert_tokens.device,
            )
        )

        tokens = torch.cat(tokens, dim=1)
        input_mask = torch.cat(input_mask, dim=1)

        # Build ar_mask with correct length matching input_mask.shape[1]
        ar_mask = torch.zeros(
            input_mask.shape[1], dtype=torch.bool, device=tokens.device
        )
        if not self.pi05:
            ar_mask[:2] = True
        else:
            ar_mask[0] = True

        return tokens, input_mask, ar_mask, adarms_cond

    def compute_loss(
        self,
        observation: model.Observation,
        actions: torch.Tensor,
        *,
        train: bool = False,
        rng: torch.Generator | None = None,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute flow matching loss.

        Returns:
            loss: (B, action_horizon) per-timestep MSE loss
        """
        B = actions.shape[0]
        device = actions.device

        # Preprocess first (requries float32 for image ops),
        # then cast to model dtype for FSDP2 mixed precision compatibility.
        observation = model.preprocess_observation(observation, train=train, rng=rng)

        embed_dtype = self.embed_dtype
        observation = model._observation_to_dtype(observation, embed_dtype)
        actions = actions.to(dtype=embed_dtype)
        dtype = actions.dtype

        # Sample noise and time (or use provided values for reproducibility)
        if noise is None:
            noise = torch.randn(
                actions.shape, device=device, dtype=dtype, generator=rng
            )
        else:
            noise = noise.to(dtype=dtype)
        if time is None:
            time = (
                torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0))
                .sample((B,))
                .to(device=device, dtype=dtype)
            )
            time = time * 0.999 + 0.001
        else:
            time = time.to(dtype=dtype)
        time_expanded = time[:, None, None]

        # Flow matching interpolation
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # One forward pass for prefix + suffix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, time
        )

        input_mask = torch.cat([prefix_mask, suffix_mask], dim=1)
        ar_mask = torch.cat([prefix_ar_mask, suffix_ar_mask], dim=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = torch.cumsum(input_mask.int(), dim=1) - 1

        prefix_out, suffix_out = self.llm(
            [prefix_tokens, suffix_tokens],
            positions=positions,
            mask=attn_mask,
            adarms_cond=[None, adarms_cond],
        )[0]

        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return torch.mean(torch.square(v_t - u_t), dim=-1)

    def build_prefix_cache(
        self, observation: model.Observation
    ) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        """Embed prefix tokens and run one LLM pass to build the KV cache.

        The caller is responsible for preprocessing the observation (image
        resize/pad, mask defaults) — this method only consumes the prepared
        observation so it can be shared between the eval Euler sampler and the
        RL train-time forward where the observation has already been built.

        Returns:
            prefix_out:  (B, prefix_len, paligemma_width) paligemma-side hidden states.
            prefix_mask: (B, prefix_len) bool mask of valid prefix positions.
            kv_cache:    per-layer KV cache to feed into subsequent suffix passes.
        """
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = torch.cumsum(prefix_mask.int(), dim=1) - 1
        outputs, kv_cache = self.llm(
            [prefix_tokens, None],
            positions=positions,
            mask=prefix_attn_mask,
        )
        return outputs[0], prefix_mask, kv_cache

    def run_suffix(
        self,
        observation: model.Observation,
        x_t: torch.Tensor,
        t_tensor: torch.Tensor,
        kv_cache: tuple,
        prefix_mask: torch.Tensor,
    ) -> torch.Tensor:
        """One suffix forward pass (action expert) given the prefix KV cache.

        Returns the action-expert hidden states sliced to the last
        ``action_horizon`` positions: (B, action_horizon, action_expert_width).
        """
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, t_tensor
        )
        suffix_len = suffix_tokens.shape[1]
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_to_suffix_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_len)
        full_attn_mask = torch.cat([prefix_to_suffix_mask, suffix_attn_mask], dim=-1)
        suffix_positions = (
            torch.sum(prefix_mask, dim=-1)[:, None]
            + torch.cumsum(suffix_mask.int(), dim=-1)
            - 1
        )
        outputs, _ = self.llm(
            [None, suffix_tokens],
            positions=suffix_positions,
            mask=full_attn_mask,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        return outputs[1][:, -self.action_horizon :]

    def velocity_from_suffix(self, suffix_out_act: torch.Tensor) -> torch.Tensor:
        """Project action-expert hidden states to a velocity prediction v_t."""
        return self.action_out_proj(suffix_out_act)

    def sample_actions(
        self,
        observation: model.Observation,
        *,
        num_steps: int = 10,
        noise: torch.Tensor | None = None,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample actions using Euler ODE solver.

        Args:
            observation: input observation
            num_steps: number of ODE solver steps
            noise: optional initial noise of shape (B, action_horizon, action_dim)
            rng: random generator

        Returns:
            actions: (B, action_horizon, action_dim)
        """
        observation = model.preprocess_observation(observation, train=False)

        dt = -1.0 / num_steps
        B = observation.state.shape[0]
        device = observation.state.device

        if noise is None:
            noise = torch.randn(
                B, self.action_horizon, self.action_dim, device=device, generator=rng
            )

        _, prefix_mask, kv_cache = self.build_prefix_cache(observation)

        x_t = noise
        t = 1.0

        # Euler integration
        while t >= -dt / 2:
            t_tensor = torch.full((B,), t, device=device, dtype=torch.float32)
            suffix_out_act = self.run_suffix(
                observation, x_t, t_tensor, kv_cache, prefix_mask
            )
            v_t = self.velocity_from_suffix(suffix_out_act)
            x_t = x_t + dt * v_t
            t = t + dt

        return x_t

    def forward(
        self,
        observation: model.Observation,
        actions: torch.Tensor,
        *,
        train: bool = True,
        rng: torch.Generator | None = None,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Default forward computes loss."""
        return self.compute_loss(
            observation, actions, train=train, rng=rng, noise=noise, time=time
        )

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: dict | None = None
    ):
        """Enable gradient checkpointing for memory efficiency.

        Args:
            gradient_checkpointing_kwargs: Optional kwargs forwarded to the activation
                checkpoint. Currently honors ``use_reentrant`` (default ``False``), so
                the FSDP ``gradient_checkpointing_use_reentrant`` setting is respected.
        """
        kwargs = gradient_checkpointing_kwargs or {}
        use_reentrant = kwargs.get("use_reentrant", False)
        self.llm.gradient_checkpointing = True
        self.llm.gradient_checkpointing_use_reentrant = use_reentrant
        self.img.encoder.gradient_checkpointing = True
        self.img.encoder.gradient_checkpointing_use_reentrant = use_reentrant

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing (used by the eval / no-recompute path)."""
        self.llm.gradient_checkpointing = False
        self.img.encoder.gradient_checkpointing = False
