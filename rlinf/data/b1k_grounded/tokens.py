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

"""Stable structural-token allocation for grounded control prompts."""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Protocol

STRUCTURAL_TOKENS = (
    "<goal>",
    "<subgoal>",
    "<skill>",
    "<arg>",
    "<end_arg>",
    "<role_target>",
    "<role_manipulated>",
    "<role_source>",
    "<role_destination>",
    "<role_reference>",
    "<role_tool>",
    "<role_other>",
    "<object>",
    "<qualifier>",
    "<part>",
    "<view_head>",
    "<view_left_wrist>",
    "<view_right_wrist>",
    "<object_bbox>",
    "<part_bbox>",
    "<point>",
    "<no_grounding>",
    "<end_control>",
)
TOKEN_MAPPING_VERSION = "b1k_grounded_control_tokens_v0.1"
_UNUSED_PIECE_PATTERN = re.compile(r"<unused(\d+)>")


class SentencePieceVocabulary(Protocol):
    """SentencePiece methods needed to validate grounded-control tokens."""

    def vocab_size(self) -> int:
        """Return the tokenizer vocabulary size."""

    def id_to_piece(self, token_id: int) -> str:
        """Return the vocabulary piece for one token ID."""

    def piece_to_id(self, piece: str) -> int:
        """Return the token ID for one vocabulary piece."""

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs."""


@dataclasses.dataclass(frozen=True)
class TokenBinding:
    """One logical control token bound to an existing vocabulary row."""

    logical_token: str
    piece: str
    token_id: int


@dataclasses.dataclass(frozen=True)
class ReservedTokenMapping:
    """Checkpoint-stable mapping from control tokens to unused PaliGemma pieces."""

    bindings: tuple[TokenBinding, ...]
    version: str = TOKEN_MAPPING_VERSION

    def __post_init__(self) -> None:
        if self.version != TOKEN_MAPPING_VERSION:
            raise ValueError(
                f"Unsupported reserved token mapping version {self.version!r}."
            )
        logical_tokens = tuple(binding.logical_token for binding in self.bindings)
        if logical_tokens != STRUCTURAL_TOKENS:
            raise ValueError(
                "Reserved token bindings must follow the frozen v0.1 structural "
                "token order."
            )
        pieces = [binding.piece for binding in self.bindings]
        token_ids = [binding.token_id for binding in self.bindings]
        if len(pieces) != len(set(pieces)) or len(token_ids) != len(set(token_ids)):
            raise ValueError("Reserved token pieces and IDs must be unique.")

    def piece(self, logical_token: str) -> str:
        """Return the tokenizer piece assigned to a logical control token."""
        for binding in self.bindings:
            if binding.logical_token == logical_token:
                return binding.piece
        raise KeyError(f"Unknown structural token {logical_token!r}.")

    def token_id(self, logical_token: str) -> int:
        """Return the vocabulary ID assigned to a logical control token."""
        for binding in self.bindings:
            if binding.logical_token == logical_token:
                return binding.token_id
        raise KeyError(f"Unknown structural token {logical_token!r}.")

    def to_dict(self) -> dict:
        """Return a checkpoint-config representation of this mapping."""
        return {
            "version": self.version,
            "bindings": [dataclasses.asdict(binding) for binding in self.bindings],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize this mapping for a checkpoint or experiment artifact."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, value: dict) -> ReservedTokenMapping:
        """Restore a mapping saved by :meth:`to_dict`."""
        return cls(
            version=value["version"],
            bindings=tuple(TokenBinding(**item) for item in value["bindings"]),
        )

    @classmethod
    def from_json(cls, value: str) -> ReservedTokenMapping:
        """Restore a mapping saved by :meth:`to_json`."""
        return cls.from_dict(json.loads(value))


@dataclasses.dataclass(frozen=True)
class TokenizerCapabilities:
    """Validated PaliGemma vocabulary capabilities used by the serializer."""

    vocab_size: int
    location_token_start: int
    location_token_end: int
    unused_token_count: int


class ReservedTokenAllocator:
    """Allocate and validate structural tokens without changing model shapes."""

    def __init__(
        self,
        tokenizer: SentencePieceVocabulary,
        *,
        model_vocab_size: int | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._model_vocab_size = model_vocab_size

    def inspect(self) -> TokenizerCapabilities:
        """Validate location tokens and report available unused pieces."""
        vocab_size = self._tokenizer.vocab_size()
        if self._model_vocab_size is not None and self._model_vocab_size < vocab_size:
            raise ValueError(
                f"Model vocabulary {self._model_vocab_size} is smaller than tokenizer "
                f"vocabulary {vocab_size}."
            )

        location_ids = []
        for index in range(1024):
            piece = f"<loc{index:04d}>"
            token_id = self._tokenizer.piece_to_id(piece)
            if self._tokenizer.id_to_piece(token_id) != piece:
                raise ValueError(f"Tokenizer is missing location token {piece}.")
            if list(self._tokenizer.encode(piece)) != [token_id]:
                raise ValueError(f"Location token {piece} does not encode atomically.")
            location_ids.append(token_id)

        unused = self._unused_pieces()
        return TokenizerCapabilities(
            vocab_size=vocab_size,
            location_token_start=location_ids[0],
            location_token_end=location_ids[-1],
            unused_token_count=len(unused),
        )

    def allocate(self) -> ReservedTokenMapping:
        """Bind the frozen logical tokens to the first available unused pieces."""
        self.inspect()
        unused = self._unused_pieces()
        if len(unused) < len(STRUCTURAL_TOKENS):
            raise ValueError(
                f"Tokenizer has {len(unused)} unused pieces, but grounded control "
                f"requires {len(STRUCTURAL_TOKENS)}."
            )
        mapping = ReservedTokenMapping(
            bindings=tuple(
                TokenBinding(logical_token, piece, token_id)
                for logical_token, (token_id, piece) in zip(
                    STRUCTURAL_TOKENS,
                    unused[: len(STRUCTURAL_TOKENS)],
                    strict=True,
                )
            )
        )
        self.validate(mapping)
        return mapping

    def validate(self, mapping: ReservedTokenMapping) -> None:
        """Verify a restored checkpoint mapping against this tokenizer/model."""
        vocab_size = self._tokenizer.vocab_size()
        for binding in mapping.bindings:
            if not 0 <= binding.token_id < vocab_size:
                raise ValueError(
                    f"Token ID {binding.token_id} for {binding.logical_token} is "
                    "outside the tokenizer vocabulary."
                )
            if self._model_vocab_size is not None and (
                binding.token_id >= self._model_vocab_size
            ):
                raise ValueError(
                    f"Token ID {binding.token_id} for {binding.logical_token} is "
                    "outside the model embedding vocabulary."
                )
            if self._tokenizer.id_to_piece(binding.token_id) != binding.piece:
                raise ValueError(
                    f"Tokenizer piece mismatch for {binding.logical_token}: "
                    f"expected {binding.piece!r}."
                )
            if self._tokenizer.piece_to_id(binding.piece) != binding.token_id:
                raise ValueError(
                    f"Tokenizer ID mismatch for {binding.logical_token}: expected "
                    f"{binding.token_id}."
                )
            if list(self._tokenizer.encode(binding.piece)) != [binding.token_id]:
                raise ValueError(
                    f"Reserved piece {binding.piece} does not encode atomically."
                )

    def _unused_pieces(self) -> list[tuple[int, str]]:
        unused = []
        for token_id in range(self._tokenizer.vocab_size()):
            piece = self._tokenizer.id_to_piece(token_id)
            match = _UNUSED_PIECE_PATTERN.fullmatch(piece)
            if match is not None:
                unused.append((int(match.group(1)), token_id, piece))
        unused.sort()
        return [(token_id, piece) for _, token_id, piece in unused]
