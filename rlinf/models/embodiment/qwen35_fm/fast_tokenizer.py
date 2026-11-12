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

"""Native FAST action tokenization and an additive Qwen vocabulary bridge."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


class FASTActionTokenizer:
    """Apply the official DCT/BPE tokenizer to quantile-normalized actions.

    Encode takes physical actions, not the z-scored actions used by FM. Only
    the leading sequence of fully valid timesteps enters the DCT, so padding
    and partially supervised timesteps cannot contaminate action token targets.
    """

    def __init__(
        self,
        processor: Any,
        normalization: dict | None = None,
        source: dict | None = None,
    ):
        self.processor = processor
        self.normalization = normalization
        self.source = source or {}
        self.bridge: dict | None = None

    @classmethod
    def from_pretrained(
        cls, path: str | Path, normalization: dict | None = None
    ) -> "FASTActionTokenizer":
        """Load reviewed local FAST tokenizer assets without fetching model weights."""
        from transformers import AutoProcessor

        path = Path(path).resolve()
        processor = AutoProcessor.from_pretrained(
            path, trust_remote_code=True, local_files_only=True
        )
        source_path = path / "FAST_ASSET_SOURCE.json"
        source = (
            json.loads(source_path.read_text())
            if source_path.exists()
            else {"path": str(path)}
        )
        return cls(processor, normalization, source)

    @staticmethod
    def _actions_and_mask(
        actions: torch.Tensor | np.ndarray, action_mask=None
    ) -> tuple[np.ndarray, np.ndarray]:
        values = (
            actions.detach().cpu().numpy()
            if isinstance(actions, torch.Tensor)
            else np.asarray(actions)
        )
        values = np.asarray(values, dtype=np.float32)
        unbatched = values.ndim == 2
        if unbatched:
            values = values[None]
        if values.ndim != 3 or min(values.shape) < 1:
            raise ValueError(f"Expected actions [B,H,D] or [H,D], got {values.shape}")
        if action_mask is None:
            mask = np.ones(values.shape, dtype=bool)
        else:
            mask = (
                action_mask.detach().cpu().numpy()
                if isinstance(action_mask, torch.Tensor)
                else np.asarray(action_mask)
            )
            if unbatched:
                mask = mask[None]
            if mask.shape == values.shape[:2]:
                mask = np.broadcast_to(mask[..., None], values.shape)
            if mask.shape != values.shape:
                raise ValueError(
                    f"Expected action mask {values.shape[:2]} or {values.shape}, got {mask.shape}"
                )
            mask = np.asarray(mask, dtype=bool)
        if not np.isfinite(values[mask]).all():
            raise ValueError("Valid action coordinates must be finite")
        return values, mask

    @staticmethod
    def fit_quantiles(actions: torch.Tensor | np.ndarray, action_mask=None) -> dict:
        """Fit 1st/99th percentiles using valid selected training coordinates only."""
        values, mask = FASTActionTokenizer._actions_and_mask(actions, action_mask)
        low, high = [], []
        for dimension in range(values.shape[-1]):
            selected = values[..., dimension][mask[..., dimension]]
            if not selected.size:
                raise ValueError(
                    f"No valid training actions for coordinate {dimension}"
                )
            lower, upper = np.quantile(selected, [0.01, 0.99])
            low.append(float(lower))
            high.append(float(upper))
        return {"q01": low, "q99": high, "fit": "valid selected training coordinates"}

    def _bounds(
        self, dimension: int, normalization: dict | None
    ) -> tuple[np.ndarray, np.ndarray]:
        stats = self.normalization if normalization is None else normalization
        if stats is None:
            raise ValueError(
                "FAST requires quantiles fitted on the selected training split"
            )
        low, high = (
            np.asarray(stats["q01"], dtype=np.float32),
            np.asarray(stats["q99"], dtype=np.float32),
        )
        if (
            low.shape != (dimension,)
            or high.shape != low.shape
            or not np.isfinite([low, high]).all()
            or np.any(high < low)
        ):
            raise ValueError(f"Expected finite ordered quantiles of width {dimension}")
        return low, high

    def encode(
        self,
        actions: torch.Tensor | np.ndarray,
        action_mask=None,
        normalization: dict | None = None,
    ) -> list[list[int]]:
        """Return native FAST code IDs; an empty valid prefix produces no CE targets."""
        values, mask = self._actions_and_mask(actions, action_mask)
        low, high = self._bounds(values.shape[-1], normalization)
        span = high - low
        constant = span <= 1e-6
        span = np.where(constant, 1.0, span)
        codes = []
        for chunk, validity in zip(values, mask, strict=True):
            complete = validity.all(-1)
            invalid = np.flatnonzero(~complete)
            length = int(invalid[0]) if invalid.size else len(chunk)
            if not length:
                codes.append([])
                continue
            normalized = np.clip(2 * (chunk[:length] - low) / span - 1, -1, 1)
            normalized[:, constant] = 0
            encoded = [int(code) for code in self.processor(normalized)[0]]
            if any(code < 0 or code >= self.processor.vocab_size for code in encoded):
                raise ValueError("Native FAST returned a code outside its vocabulary")
            codes.append(encoded)
        return codes

    def attach(self, model: Any, qwen_tokenizer: Any) -> dict:
        """Add FAST symbols without changing existing tokens or pretrained rows.

        Run before distributed wrapping and optimizer creation. All ranks must
        use the same initialization seed when expanding native embeddings.
        """
        old_rows = model.get_input_embeddings().num_embeddings
        original_tokenizer_size = len(qwen_tokenizer)
        specials = {
            token: qwen_tokenizer.convert_tokens_to_ids(token)
            for token in qwen_tokenizer.all_special_tokens
        }
        code_names = [
            f"<|fast_action_{index:04d}|>" for index in range(self.processor.vocab_size)
        ]
        names = ["<|fast_action_begin|>", *code_names, "<|fast_action_end|>"]
        qwen_tokenizer.add_special_tokens(
            {"extra_special_tokens": names},
            replace_extra_special_tokens=False,
        )
        new_rows = max(old_rows, len(qwen_tokenizer))
        if new_rows > old_rows:
            model.resize_token_embeddings(new_rows, mean_resizing=False)
        model.tie_weights()
        if any(
            qwen_tokenizer.convert_tokens_to_ids(token) != token_id
            for token, token_id in specials.items()
        ):
            raise RuntimeError(
                "Attaching FAST must preserve every native special token ID"
            )
        self.bridge = {
            "original_model_vocab_rows": old_rows,
            "original_tokenizer_size": original_tokenizer_size,
            "model_vocab_rows": new_rows,
            "fast_vocab_size": self.processor.vocab_size,
            "begin_token_id": qwen_tokenizer.convert_tokens_to_ids(names[0]),
            "end_token_id": qwen_tokenizer.convert_tokens_to_ids(names[-1]),
            "code_token_ids": [
                qwen_tokenizer.convert_tokens_to_ids(token) for token in code_names
            ],
        }
        return dict(self.bridge)

    def to_qwen_tokens(self, codes: list[list[int]]) -> list[list[int]]:
        """Map FAST codes to native LM targets, including action delimiters."""
        if self.bridge is None:
            raise ValueError(
                "Attach the FAST vocabulary before constructing Qwen targets"
            )
        lookup = self.bridge["code_token_ids"]
        return [
            [
                self.bridge["begin_token_id"],
                *[lookup[index] for index in sequence],
                self.bridge["end_token_id"],
            ]
            if sequence
            else []
            for sequence in codes
        ]

    def encode_qwen(
        self,
        actions: torch.Tensor | np.ndarray,
        action_mask=None,
        normalization: dict | None = None,
    ) -> list[list[int]]:
        """Encode physical actions directly to FAST-delimited Qwen token IDs."""
        return self.to_qwen_tokens(self.encode(actions, action_mask, normalization))

    def decode(
        self,
        tokens: list[list[int]],
        horizon: int,
        action_dim: int,
        normalization: dict | None = None,
    ) -> np.ndarray:
        """Decode native FAST codes to physical actions with explicit geometry."""
        low, high = self._bounds(action_dim, normalization)
        for sequence in tokens:
            if (
                len(self.processor.bpe_tokenizer.decode(sequence))
                != horizon * action_dim
            ):
                raise ValueError(
                    "FAST sequence does not decode to the requested action geometry"
                )
        normalized = self.processor.decode(
            tokens, time_horizon=horizon, action_dim=action_dim
        )
        actions = (normalized + 1) * (high - low) / 2 + low
        return actions.astype(np.float32)

    def save_metadata(self, path: str | Path) -> None:
        """Save normalization, source lineage and token mapping with a run/checkpoint."""
        Path(path).write_text(
            json.dumps(
                {
                    "source": self.source,
                    "normalization": self.normalization,
                    "bridge": self.bridge,
                },
                indent=2,
            )
        )


def build_ce_inputs(
    prefix_inputs: dict[str, torch.Tensor],
    action_token_ids: list[list[int]],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """Append FAST targets to native prefix inputs with suffix-only CE labels.

    The original prefix dictionary remains unchanged for the FM branch. Pixel
    tensors and image/video grids retain the native batch ordering. Native Qwen
    recomputes multimodal positions for the newly left-padded token sequence.
    """
    if "position_ids" in prefix_inputs or "inputs_embeds" in prefix_inputs:
        raise ValueError(
            "FAST CE requires native input_ids with positions recomputed by Qwen"
        )
    input_ids = prefix_inputs["input_ids"]
    mask = prefix_inputs.get("attention_mask", torch.ones_like(input_ids))
    if (
        input_ids.ndim != 2
        or mask.shape != input_ids.shape
        or len(action_token_ids) != input_ids.shape[0]
    ):
        raise ValueError(
            "Expected one FAST sequence and one 2-D attention mask per prefix"
        )
    if bool((mask[:, :-1].bool() & ~mask[:, 1:].bool()).any()):
        raise ValueError("FAST CE prefix requires left padding")
    lengths = [
        int(valid.sum()) + len(sequence)
        for valid, sequence in zip(mask, action_token_ids, strict=True)
    ]
    width = max(lengths)
    ids = input_ids.new_full((input_ids.shape[0], width), pad_token_id)
    attention = mask.new_zeros(ids.shape)
    labels = input_ids.new_full(ids.shape, -100)
    type_ids = prefix_inputs.get("mm_token_type_ids")
    types = type_ids.new_zeros(ids.shape) if type_ids is not None else None
    for index, sequence in enumerate(action_token_ids):
        valid = mask[index].bool()
        prefix = input_ids[index, valid]
        suffix = input_ids.new_tensor(sequence)
        start = width - lengths[index]
        ids[index, start : start + len(prefix)] = prefix
        ids[index, start + len(prefix) :] = suffix
        attention[index, start:] = 1
        labels[index, start + len(prefix) :] = suffix
        if types is not None:
            types[index, start : start + len(prefix)] = type_ids[index, valid]
    result = {
        key: value
        for key, value in prefix_inputs.items()
        if key not in {"input_ids", "attention_mask", "mm_token_type_ids", "labels"}
    }
    result.update(input_ids=ids, attention_mask=attention, labels=labels)
    if types is not None:
        result["mm_token_type_ids"] = types
    return result
