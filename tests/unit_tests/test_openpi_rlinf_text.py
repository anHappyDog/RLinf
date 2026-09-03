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

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlinf.models.embodiment.openpi_rlinf.pi0_model.model import Observation
from rlinf.models.embodiment.openpi_rlinf.pi0_model.pi0 import Pi0, make_attn_mask


class _FakeEmbedder(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, vocabulary_size)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.eye(vocabulary_size))

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding(tokens)

    def decode(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.embedding.weight)


class _FakeLlm(nn.Module):
    def __init__(self, next_token: list[int]):
        super().__init__()
        self.embedder = _FakeEmbedder(len(next_token))
        transition = torch.zeros(len(next_token), len(next_token))
        for token, successor in enumerate(next_token):
            transition[token, successor] = 10.0
        self.register_buffer("transition", transition)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedder.encode(tokens)

    def forward(self, embedded, positions, mask, adarms_cond=None, *, kv_cache=None):
        assert mask.ndim == 3
        del positions, adarms_cond, kv_cache
        output = embedded[0] @ self.transition
        return [output, None], ()


class _TinyTextPi0(Pi0):
    def __init__(self):
        nn.Module.__init__(self)
        self.pcd = False
        self.embed_dtype = torch.float32
        # 0 -> 1 is prefix-only. The response is 2 -> 3 -> EOS(4).
        self.llm = _FakeLlm([1, 2, 3, 4, 4, 5])

    def embed_prefix(self, obs: Observation):
        text = self.llm.embed(obs.tokenized_prompt)
        image = torch.zeros(
            text.shape[0], 1, text.shape[-1], dtype=text.dtype, device=text.device
        )
        tokens = torch.cat([image, text], dim=1)
        image_mask = torch.ones(text.shape[0], 1, dtype=torch.bool, device=text.device)
        input_mask = torch.cat([image_mask, obs.tokenized_prompt_mask], dim=1)
        ar_mask = torch.zeros(tokens.shape[1], dtype=torch.bool, device=text.device)
        return tokens, input_mask, ar_mask


class _FakeImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.call_count = 0

    def forward(self, image: torch.Tensor):
        self.call_count += 1
        return torch.zeros(image.shape[0], 1, 6), None


class _ActionPrefixPi0(Pi0):
    def __init__(self):
        nn.Module.__init__(self)
        self.pcd = False
        self.embed_dtype = torch.float32
        self.img = _FakeImageEncoder()
        self.llm = _FakeLlm([1, 2, 3, 4, 4, 5])


def _observation(*, with_response: bool) -> Observation:
    image = torch.zeros(1, 224, 224, 3)
    image_mask = torch.ones(1, dtype=torch.bool)
    if with_response:
        tokens = torch.tensor([[0, 1, 2, 3, 4, 0]])
        token_mask = torch.tensor([[True, True, True, True, True, False]])
        ar_mask = torch.tensor([[False, False, True, True, True, False]])
        loss_mask = torch.tensor([[False, False, True, True, True, False]])
    else:
        tokens = torch.tensor([[0, 1, 0, 0, 0, 0]])
        token_mask = torch.tensor([[True, True, False, False, False, False]])
        ar_mask = torch.zeros_like(token_mask)
        loss_mask = None
    return Observation(
        images={
            "base_0_rgb": image,
            "left_wrist_0_rgb": image,
            "right_wrist_0_rgb": image,
        },
        image_masks={
            "base_0_rgb": image_mask,
            "left_wrist_0_rgb": image_mask,
            "right_wrist_0_rgb": image_mask,
        },
        state=torch.zeros(1, 1),
        tokenized_prompt=tokens,
        tokenized_prompt_mask=token_mask,
        token_ar_mask=ar_mask,
        token_loss_mask=loss_mask,
    )


def test_make_attn_mask_is_bidirectional_prefix_then_causal_response():
    input_mask = torch.ones(1, 5, dtype=torch.bool)
    ar_mask = torch.tensor([[False, False, True, True, True]])

    mask = make_attn_mask(input_mask, ar_mask)

    expected = torch.tensor(
        [
            [True, True, False, False, False],
            [True, True, False, False, False],
            [True, True, True, False, False],
            [True, True, True, True, False],
            [True, True, True, True, True],
        ]
    )
    torch.testing.assert_close(mask[0], expected)


def test_compute_text_loss_shifts_targets_and_masks_prefix():
    model = _TinyTextPi0()

    output = model.compute_text_loss(_observation(with_response=True))

    assert output.token_count.item() == 3
    assert output.token_accuracy.item() == 1.0
    assert output.loss.item() < 0.01


def test_generate_text_starts_at_last_prefix_token_and_stops_at_eos():
    model = _TinyTextPi0()

    tokens, mask = model.generate_text(
        _observation(with_response=False), eos_token_id=4, max_new_tokens=4
    )

    torch.testing.assert_close(tokens, torch.tensor([[2, 3, 4, 0]]))
    torch.testing.assert_close(mask, torch.tensor([[True, True, True, False]]))


def test_action_prefix_remains_bidirectional_when_text_masks_are_present():
    model = _ActionPrefixPi0()
    observation = _observation(with_response=True)

    _, _, action_ar_mask = model.embed_prefix(observation)

    assert not action_ar_mask.any()


def test_cached_generation_encodes_each_camera_only_once():
    model = _ActionPrefixPi0()

    tokens, mask = model.generate_text(
        _observation(with_response=False), eos_token_id=4, max_new_tokens=4
    )

    torch.testing.assert_close(tokens, torch.tensor([[2, 3, 4, 0]]))
    torch.testing.assert_close(mask, torch.tensor([[True, True, True, False]]))
    assert model.img.call_count == 3
