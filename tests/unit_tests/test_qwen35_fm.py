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

"""Native hybrid VLM and continuous-action expert contracts."""

import copy
import inspect
import os

import pytest
import torch

pytest.importorskip("transformers.models.qwen3_5")

from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)

from rlinf.models.embodiment.qwen35_fm.action_expert import (
    ActionExpertConfig,
    GDNStateReadAdapter,
    GQACrossAttention,
    build_action_expert,
)
from rlinf.models.embodiment.qwen35_fm.policy import FlowMatchingPolicy
from rlinf.models.embodiment.qwen35_fm.vlm import Qwen35VLMWrapper, VLMCondition

VARIANTS = ["gqa6", "gqa6_gdn1", "gqa6_gdn2", "gqa6_gdn3", "gqa6_vlm_state"]
STATE_SPECS = {0: (2, 8, 8), 2: (2, 8, 8)}


def expert_config(variant="gqa6", **kwargs):
    return ActionExpertConfig(
        variant=variant,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        num_kv_heads=2,
        vlm_hidden_size=32,
        action_dim=3,
        horizon=4,
        gdn_num_key_heads=2,
        gdn_num_value_heads=2,
        gdn_key_head_dim=8,
        gdn_value_head_dim=8,
        state_adapter_dim=8,
        **kwargs,
    )


def synthetic_condition(batch=2, requires_grad=False):
    return VLMCondition(
        hidden_states=torch.randn(batch, 5, 32, requires_grad=requires_grad),
        attention_mask=torch.tensor([[1, 1, 1, 0, 0]]).expand(batch, -1),
        gdn_states={
            layer: torch.randn(batch, *spec, requires_grad=requires_grad)
            for layer, spec in STATE_SPECS.items()
        },
    )


def tiny_vlm(trainable=False, state_layers=None):
    config = Qwen3_5Config(
        text_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 3,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "layer_types": ["linear_attention", "full_attention", "linear_attention"],
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000,
                "partial_rotary_factor": 0.25,
                "mrope_section": [1, 1, 0],
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "out_hidden_size": 32,
            "patch_size": 2,
            "temporal_patch_size": 1,
            "spatial_merge_size": 1,
            "num_position_embeddings": 16,
        },
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
    )
    model = Qwen3_5ForConditionalGeneration(config)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return Qwen35VLMWrapper(model, trainable=trainable, state_layers=state_layers).to(
        device
    )


def text_inputs():
    return {
        "input_ids": torch.tensor(
            [[3, 4, 5, 6], [7, 8, 9, 10]],
            device="cuda" if torch.cuda.is_available() else "cpu",
        ),
        "attention_mask": torch.ones(
            2,
            4,
            dtype=torch.long,
            device="cuda" if torch.cuda.is_available() else "cpu",
        ),
    }


@pytest.mark.parametrize("variant", VARIANTS)
def test_velocity_loss_optimizer_and_sampling(variant):
    torch.manual_seed(9)
    expert = build_action_expert(expert_config(variant), STATE_SPECS)
    policy = FlowMatchingPolicy(None, expert)
    condition = synthetic_condition()
    actions = torch.randn(2, 4, 3)
    velocity = policy.predict_velocity(actions, torch.tensor([0.25, 0.75]), condition)
    assert velocity.shape == actions.shape
    assert expert.cross_attention_count == 6
    optimizer = torch.optim.AdamW(expert.parameters(), lr=1e-3)
    before = expert.output.weight.detach().clone()
    loss = policy.loss(actions, condition)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert expert.embedding.projection.weight.grad.abs().sum() > 0
    assert expert.time_embedding.mlp[0].weight.grad.abs().sum() > 0
    if variant.startswith("gqa6_gdn"):
        for block in expert.gdn_blocks:
            assert block.mixer.in_proj_qkv.weight.grad.abs().sum() > 0
    if variant == "gqa6_vlm_state":
        for adapter in expert.state_adapters:
            assert adapter.queries["0"].weight.grad.abs().sum() > 0
            assert adapter.outputs["2"].weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, expert.output.weight)
    for method in ("euler", "heun"):
        sampled = policy.sample(
            condition, steps=2, method=method, initial_noise=actions
        )
        assert sampled.shape == actions.shape
        assert torch.isfinite(sampled).all()


@pytest.mark.parametrize("variant", VARIANTS)
def test_independent_calls_and_batch_isolation(variant):
    expert = build_action_expert(expert_config(variant), STATE_SPECS).eval()
    condition = synthetic_condition()
    actions, time = torch.randn(2, 4, 3), torch.tensor([0.3, 0.7])
    with torch.no_grad():
        first = expert(actions, time, condition)
        expert(torch.randn_like(actions), 1 - time, synthetic_condition())
        repeated = expert(actions, time, condition)
        one_condition = VLMCondition(
            condition.hidden_states[:1],
            condition.attention_mask[:1],
            {i: state[:1] for i, state in condition.gdn_states.items()},
        )
        single = expert(actions[:1], time[:1], one_condition)
    torch.testing.assert_close(repeated, first, rtol=0, atol=0)
    torch.testing.assert_close(single, first[:1], rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("variant", ["gqa6", "gqa6_vlm_state", "gqa6_gdn2"])
def test_padding_context_does_not_change_velocity(variant):
    expert = build_action_expert(expert_config(variant), STATE_SPECS).eval()
    condition = synthetic_condition()
    actions, time = torch.randn(2, 4, 3), torch.tensor([0.3, 0.7])
    with torch.no_grad():
        first = expert(actions, time, condition)
        condition.hidden_states[:, 3:] = 1000 * torch.randn_like(
            condition.hidden_states[:, 3:]
        )
        changed = expert(actions, time, condition)
    torch.testing.assert_close(first, changed, rtol=0, atol=0)


def test_fast_weight_readout_is_differentiable_in_queries_and_memory():
    adapter = GDNStateReadAdapter(expert_config("gqa6_vlm_state"), STATE_SPECS)
    condition = synthetic_condition(requires_grad=True)
    actions = torch.randn(2, 4, 32, requires_grad=True)
    adapter(actions, condition.gdn_states).square().sum().backward()
    assert actions.grad.abs().sum() > 0
    for state in condition.gdn_states.values():
        assert state.grad.abs().sum() > 0
    assert adapter.query_input.weight.grad.abs().sum() > 0


def test_native_chunk_and_reference_recurrence_agree():
    torch.manual_seed(7)
    q, k = torch.randn(2, 7, 2, 8), torch.randn(2, 7, 2, 8)
    value = torch.randn(2, 7, 2, 6)
    decay = -torch.rand(2, 7, 2)
    beta = torch.rand(2, 7, 2)
    initial = torch.randn(2, 2, 8, 6)
    args = {
        "g": decay,
        "beta": beta,
        "initial_state": initial,
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": True,
    }
    chunk_output, chunk_state = inspect.unwrap(torch_chunk_gated_delta_rule)(
        q, k, value, **args
    )
    recurrent_output, recurrent_state = inspect.unwrap(
        torch_recurrent_gated_delta_rule
    )(q, k, value, **args)
    torch.testing.assert_close(chunk_output, recurrent_output, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(chunk_state, recurrent_state, rtol=2e-4, atol=2e-5)


def test_native_outputs_and_selected_state_mapping():
    wrapper = tiny_vlm(state_layers=[2])
    inputs = text_inputs()
    with torch.no_grad():
        native = wrapper.model.model(**inputs, use_cache=False).last_hidden_state
        encoded = wrapper.encode(**inputs)
        states = wrapper.encode(
            **inputs, extract_states=True, output_hidden_states=True
        )
    torch.testing.assert_close(encoded.hidden_states, native, rtol=0, atol=0)
    torch.testing.assert_close(states.hidden_states, native, rtol=1e-5, atol=1e-6)
    assert set(states.gdn_states) == {2}
    assert states.gdn_states[2].shape == (2, 2, 8, 8)
    assert states.intermediate_hidden_states is not None
    assert wrapper.gdn_layers == (0, 2)
    assert set(wrapper.state_specs) == {2}


@pytest.mark.parametrize("trainable", [False, True])
def test_vlm_gradients_and_no_cache_leakage(trainable):
    wrapper = tiny_vlm(trainable=trainable)
    expert = build_action_expert(
        expert_config("gqa6_vlm_state"), wrapper.state_specs
    ).to(next(wrapper.parameters()).device)
    condition = wrapper.encode(**text_inputs(), extract_states=True)
    if trainable:
        for state in condition.gdn_states.values():
            state.retain_grad()
    policy = FlowMatchingPolicy(wrapper, expert)
    loss = policy.loss(
        torch.randn(2, 4, 3, device=condition.hidden_states.device), condition
    )
    loss.backward()
    embedding = wrapper.model.model.language_model.embed_tokens.weight
    if trainable:
        assert embedding.grad is not None and embedding.grad.abs().sum() > 0
        assert all(state.requires_grad for state in condition.gdn_states.values())
        assert all(
            state.grad is not None and state.grad.abs().sum() > 0
            for state in condition.gdn_states.values()
        )
    else:
        assert all(p.grad is None for p in wrapper.parameters())
        assert all(not state.requires_grad for state in condition.gdn_states.values())
    with torch.no_grad():
        saved = {i: state.clone() for i, state in condition.gdn_states.items()}
        wrapper.encode(
            input_ids=torch.tensor([[11, 12]], device=condition.hidden_states.device),
            attention_mask=torch.ones(1, 2, device=condition.hidden_states.device),
            extract_states=True,
        )
        repeated = wrapper.encode(**text_inputs(), extract_states=True)
    for layer in saved:
        torch.testing.assert_close(
            condition.gdn_states[layer], saved[layer], rtol=0, atol=0
        )
        torch.testing.assert_close(
            repeated.gdn_states[layer], saved[layer], rtol=1e-5, atol=1e-6
        )


@pytest.mark.skipif(
    not os.environ.get("QWEN35_CHECKPOINT"),
    reason="Set QWEN35_CHECKPOINT to local Qwen3.5-2B weights",
)
def test_local_pretrained_multimodal_checkpoint():
    from PIL import Image

    wrapper = Qwen35VLMWrapper.from_pretrained(
        os.environ["QWEN35_CHECKPOINT"], trainable=False, torch_dtype=torch.bfloat16
    )
    wrapper = wrapper.to("cuda" if torch.cuda.is_available() else "cpu")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.new("RGB", (32, 32))},
                {"type": "text", "text": "Pick up the object."},
            ],
        }
    ]
    inputs = wrapper.processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    inputs = inputs.to(next(wrapper.parameters()).device)
    condition = wrapper.encode(**inputs, extract_states=True)
    assert condition.hidden_states.shape[-1] == 2048
    assert len(condition.gdn_states) == 18
    assert all(
        state.shape == (1, 16, 128, 128) for state in condition.gdn_states.values()
    )
    assert torch.isfinite(condition.hidden_states).all()


def test_action_reference_recurrence_matches_native_chunk():
    native = build_action_expert(expert_config("gqa6_gdn2"))
    reference = build_action_expert(expert_config("gqa6_gdn2", gdn_kernel="reference"))
    reference.load_state_dict(native.state_dict())
    condition = synthetic_condition()
    actions = torch.randn(2, 4, 3)
    time = torch.tensor([0.2, 0.8])
    torch.testing.assert_close(
        native(actions, time, condition),
        reference(actions, time, condition),
        rtol=2e-4,
        atol=2e-5,
    )


def test_right_padded_state_extraction_requires_left_padding():
    wrapper = tiny_vlm()
    inputs = text_inputs()
    inputs["attention_mask"][:, -1] = 0
    with pytest.raises(ValueError, match="left"):
        wrapper.encode(**inputs, extract_states=True)


def test_native_image_conditioning_is_preserved():
    wrapper = tiny_vlm()
    device = next(wrapper.parameters()).device
    inputs = {
        "input_ids": torch.tensor([[62, 60, 63, 3]], device=device),
        "attention_mask": torch.ones(1, 4, dtype=torch.long, device=device),
        "pixel_values": torch.randn(1, 12, device=device),
        "image_grid_thw": torch.tensor([[1, 1, 1]], device=device),
        "mm_token_type_ids": torch.tensor([[0, 1, 0, 0]], device=device),
    }
    with torch.no_grad():
        native = wrapper.model.model(**inputs, use_cache=False).last_hidden_state
        condition = wrapper.encode(**inputs, extract_states=True)
    torch.testing.assert_close(condition.hidden_states, native, rtol=1e-4, atol=1e-5)
    assert set(condition.gdn_states) == {0, 2}


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("coordinate_mask", [False, True])
def test_masked_actions_do_not_change_valid_flow_loss(variant, coordinate_mask):
    policy = FlowMatchingPolicy(
        None, build_action_expert(expert_config(variant), STATE_SPECS)
    )
    actions, noise = torch.randn(2, 4, 3), torch.randn(2, 4, 3)
    condition = synthetic_condition()
    time = torch.tensor([0.25, 0.75])
    mask = torch.ones_like(actions, dtype=torch.bool)
    mask[:, -1] = False
    if coordinate_mask:
        mask[:, 1:3, -1] = False
    first = policy.loss(
        actions,
        condition,
        action_mask=mask if coordinate_mask else mask.any(-1),
        noise=noise,
        timestep=time,
    )
    changed_actions = actions.masked_fill(~mask, 1000)
    changed_noise = noise.masked_fill(~mask, -1000)
    changed = policy.loss(
        changed_actions,
        condition,
        action_mask=mask if coordinate_mask else mask.any(-1),
        noise=changed_noise,
        timestep=time,
    )
    torch.testing.assert_close(first, changed, rtol=0, atol=0)
    with pytest.raises(ValueError, match="valid"):
        policy.loss(
            actions,
            condition,
            action_mask=torch.zeros_like(mask),
            noise=noise,
            timestep=time,
        )


def test_additive_gdn_preserves_complete_gqa_backbone():
    torch.manual_seed(42)
    baseline = build_action_expert(expert_config())
    baseline_state = baseline.state_dict()
    counts = [sum(p.numel() for p in baseline.parameters())]
    for depth in (1, 2, 3):
        torch.manual_seed(42)
        expert = build_action_expert(expert_config(f"gqa6_gdn{depth}"))
        assert len(expert.blocks) == 6
        assert all(block.self_attention is not None for block in expert.blocks)
        assert expert.cross_attention_count == 6
        assert len(expert.gdn_blocks) == depth
        for name, tensor in baseline_state.items():
            torch.testing.assert_close(
                expert.state_dict()[name], tensor, rtol=0, atol=0
            )
        counts.append(sum(p.numel() for p in expert.parameters()))
    assert counts == sorted(set(counts))
    assert counts[1] - counts[0] == counts[2] - counts[1] == counts[3] - counts[2]


def ki_inputs():
    prefix = text_inputs()
    device = prefix["input_ids"].device
    suffix = torch.tensor([[11, 12, 2], [13, 14, 2]], device=device)
    ids = torch.cat((prefix["input_ids"], suffix), dim=1)
    labels = ids.clone()
    labels[:, : prefix["input_ids"].shape[1]] = -100
    return prefix, {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "labels": labels,
    }


def ki_policy(variant="gqa6_vlm_state"):
    from rlinf.models.embodiment.qwen35_fm.ki_policy import KIFlowMatchingPolicy

    vlm = tiny_vlm(trainable=True)
    config = ActionExpertConfig(**{**vars(expert_config(variant)), "action_dim": 8})
    expert = build_action_expert(config, vlm.state_specs).to(
        next(vlm.parameters()).device
    )
    return KIFlowMatchingPolicy(vlm, expert)


@pytest.mark.parametrize("variant", ["gqa6", "gqa6_gdn3", "gqa6_vlm_state"])
def test_ki_gradient_routing_and_optimizer(variant):
    policy = ki_policy(variant).train()
    prefix, ce_inputs = ki_inputs()
    device = prefix["input_ids"].device
    actions = torch.randn(2, 4, 8, device=device)
    condition = policy.encode_prefix(prefix)
    assert not condition.hidden_states.requires_grad
    assert all(not value.requires_grad for value in condition.gdn_states.values())
    assert policy.vlm.training and policy.vlm.model.training
    policy.loss(actions, condition).backward()
    assert all(parameter.grad is None for parameter in policy.vlm.parameters())
    assert policy.expert.output.weight.grad.abs().sum() > 0
    if variant == "gqa6_vlm_state":
        for adapter in policy.expert.state_adapters:
            assert adapter.query_input.weight.grad.abs().sum() > 0
            assert adapter.queries["0"].weight.grad.abs().sum() > 0
    policy.zero_grad(set_to_none=True)
    action_ce = policy.action_ce_loss(ce_inputs)
    assert torch.isfinite(action_ce)
    action_ce.backward()
    language = policy.vlm.model.model.language_model
    assert language.layers[0].linear_attn.in_proj_qkv.weight.grad.abs().sum() > 0
    assert policy.vlm.model.get_output_embeddings().weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in policy.expert.parameters())
    policy.zero_grad(set_to_none=True)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    vlm_before = language.embed_tokens.weight.detach().clone()
    expert_before = policy.expert.output.weight.detach().clone()
    result = policy(prefix, ce_inputs, actions)
    assert torch.isfinite(result["loss"])
    assert result["action_ce_tokens"] == 6
    assert result["fm_valid_action_scalars"] == actions.numel()
    result["loss"].backward()
    optimizer.step()
    assert not torch.equal(vlm_before, language.embed_tokens.weight)
    assert not torch.equal(expert_before, policy.expert.output.weight)


def test_ki_action_ce_matches_native_masked_loss():
    policy = ki_policy().eval()
    _, ce_inputs = ki_inputs()
    with torch.no_grad():
        native = policy.vlm.model(**ce_inputs, use_cache=False).loss
        selected = policy.action_ce_loss(ce_inputs)
    torch.testing.assert_close(selected, native, rtol=1e-6, atol=1e-6)


def test_ki_prefix_states_are_isolated_from_teacher_forced_actions():
    policy = ki_policy().train()
    prefix, ce_inputs = ki_inputs()
    device = prefix["input_ids"].device
    actions, noise = (
        torch.randn(2, 4, 8, device=device),
        torch.randn(2, 4, 8, device=device),
    )
    time = torch.tensor([0.2, 0.8], device=device)
    condition = policy.encode_prefix(prefix)
    hidden = condition.hidden_states.clone()
    states = {layer: value.clone() for layer, value in condition.gdn_states.items()}
    with torch.no_grad():
        first = policy(prefix, ce_inputs, actions, noise=noise, timestep=time)
        changed = {key: value.clone() for key, value in ce_inputs.items()}
        changed["input_ids"][:, -3:] = torch.tensor([15, 16, 17], device=device)
        changed["labels"][:, -3:] = changed["input_ids"][:, -3:]
        second = policy(prefix, changed, actions, noise=noise, timestep=time)
        repeated = policy.encode_prefix(prefix)
    torch.testing.assert_close(first["fm_loss"], second["fm_loss"], rtol=0, atol=0)
    assert not torch.equal(first["action_ce_loss"], second["action_ce_loss"])
    torch.testing.assert_close(condition.hidden_states, hidden, rtol=0, atol=0)
    torch.testing.assert_close(repeated.hidden_states, hidden, rtol=0, atol=0)
    for layer, expected in states.items():
        torch.testing.assert_close(
            condition.gdn_states[layer], expected, rtol=0, atol=0
        )
        torch.testing.assert_close(repeated.gdn_states[layer], expected, rtol=0, atol=0)
    assert policy.vlm.training and policy.vlm.model.training


def test_ki_checkpointed_ce_keeps_native_prefix_cache():
    policy = ki_policy().train()
    policy.vlm.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    prefix, ce_inputs = ki_inputs()
    condition = policy.encode_prefix(prefix)
    assert condition.gdn_states and policy.vlm.training
    assert any(
        getattr(module, "gradient_checkpointing", False)
        for module in policy.vlm.modules()
    )
    result = policy(
        prefix, ce_inputs, torch.randn(2, 4, 8, device=prefix["input_ids"].device)
    )
    result["loss"].backward()
    assert (
        policy.vlm.model.model.language_model.layers[0]
        .linear_attn.in_proj_qkv.weight.grad.abs()
        .sum()
        > 0
    )
    assert policy.expert.state_adapters[0].query_input.weight.grad.abs().sum() > 0


def test_ki_rejects_changed_prefix_and_prefix_supervision():
    policy = ki_policy()
    prefix, ce_inputs = ki_inputs()
    device = prefix["input_ids"].device
    changed = {key: value.clone() for key, value in ce_inputs.items()}
    changed["input_ids"][0, 0] = 20
    with pytest.raises(ValueError, match="identical"):
        policy(prefix, changed, torch.randn(2, 4, 8, device=device))
    changed = {key: value.clone() for key, value in ce_inputs.items()}
    changed["labels"][0, 1] = changed["input_ids"][0, 1]
    with pytest.raises(ValueError, match="suffix"):
        policy(prefix, changed, torch.randn(2, 4, 8, device=device))


def test_ki_vision_language_ce_trains_native_vision_and_language():
    policy = ki_policy().train()
    device = next(policy.parameters()).device
    inputs = {
        "input_ids": torch.tensor([[62, 60, 63, 3, 4, 5, 6]], device=device),
        "attention_mask": torch.ones(1, 7, dtype=torch.long, device=device),
        "pixel_values": torch.randn(1, 12, device=device),
        "image_grid_thw": torch.tensor([[1, 1, 1]], device=device),
        "mm_token_type_ids": torch.tensor([[0, 1, 0, 0, 0, 0, 0]], device=device),
        "labels": torch.tensor([[-100, -100, -100, -100, 4, 5, 6]], device=device),
    }
    loss = policy.vl_loss(inputs)
    assert torch.isfinite(loss)
    loss.backward()
    assert policy.vlm.model.model.visual.patch_embed.proj.weight.grad.abs().sum() > 0
    assert (
        policy.vlm.model.model.language_model.layers[0]
        .linear_attn.in_proj_qkv.weight.grad.abs()
        .sum()
        > 0
    )
    assert all(parameter.grad is None for parameter in policy.expert.parameters())


def fast_actions():
    time = torch.linspace(-1, 1, 16)
    normalized = torch.stack(
        [torch.sin(time * (dimension + 1) * 0.3) for dimension in range(8)], dim=-1
    )
    scales = torch.tensor([1.0] * 7 + [127.5])
    offsets = torch.tensor([0.0] * 7 + [127.5])
    return torch.stack((normalized, normalized.flip(0))) * scales + offsets


def load_fast_tokenizer(normalization=None):
    from rlinf.models.embodiment.qwen35_fm.fast_tokenizer import FASTActionTokenizer

    path = os.environ.get(
        "QWEN35_FAST_TOKENIZER",
        "/mnt/public/daibo/models/physical-intelligence-fast-tokenizer",
    )
    if not os.path.exists(path):
        pytest.skip("Set QWEN35_FAST_TOKENIZER to reviewed local FAST assets")
    return FASTActionTokenizer.from_pretrained(path, normalization)


def test_ki_fast_quantiles_padding_and_eight_dimensional_roundtrip():
    from rlinf.models.embodiment.qwen35_fm.fast_tokenizer import FASTActionTokenizer

    actions = fast_actions()
    mask = torch.ones_like(actions, dtype=torch.bool)
    mask[1, 12:] = False
    stats = FASTActionTokenizer.fit_quantiles(actions, mask)
    changed = actions.masked_fill(~mask, 10000)
    assert stats == FASTActionTokenizer.fit_quantiles(changed, mask)
    assert len(stats["q01"]) == len(stats["q99"]) == 8
    tokenizer = load_fast_tokenizer(stats)
    codes = tokenizer.encode(actions, mask)
    assert codes == tokenizer.encode(changed, mask)
    assert codes[1] == tokenizer.encode(actions[1, :12])[0]
    decoded = torch.from_numpy(tokenizer.decode([codes[0]], 16, 8))
    assert decoded.shape == (1, 16, 8) and torch.isfinite(decoded).all()
    span = torch.tensor(stats["q99"]) - torch.tensor(stats["q01"])
    assert ((decoded[0] - actions[0]).abs() / span).max() < 0.15
    prefix_decoded = tokenizer.decode([codes[1]], 12, 8)
    assert prefix_decoded.shape == (1, 12, 8)
    mask[0, 0] = False
    assert tokenizer.encode(actions, mask)[0] == []
    with pytest.raises(ValueError, match="geometry"):
        tokenizer.decode([codes[1]], 16, 8)


def test_ki_fast_bridge_preserves_native_vocabulary_and_trains_end_to_end():
    from transformers import AutoTokenizer

    from rlinf.models.embodiment.qwen35_fm.fast_tokenizer import (
        FASTActionTokenizer,
        build_ce_inputs,
    )
    from rlinf.models.embodiment.qwen35_fm.ki_policy import KIFlowMatchingPolicy

    qwen_path = os.environ.get(
        "QWEN35_PROCESSOR", "/mnt/public/daibo/models/Qwen3.5-2B"
    )
    if not os.path.exists(qwen_path):
        pytest.skip("Set QWEN35_PROCESSOR to local Qwen tokenizer assets")
    tokenizer = AutoTokenizer.from_pretrained(qwen_path, local_files_only=True)
    native = tiny_vlm(trainable=True).model
    native.config.tie_word_embeddings = True
    native.config.text_config.tie_word_embeddings = True
    native.resize_token_embeddings(248320, mean_resizing=False)
    native.tie_weights()
    old_weights = native.get_input_embeddings().weight.detach().clone()
    old_special_ids = {
        name: tokenizer.convert_tokens_to_ids(name)
        for name in tokenizer.all_special_tokens
    }
    actions = fast_actions()
    fast = load_fast_tokenizer(FASTActionTokenizer.fit_quantiles(actions))
    bridge = fast.attach(native, tokenizer)
    torch.testing.assert_close(
        native.get_input_embeddings().weight[: len(old_weights)],
        old_weights,
        rtol=0,
        atol=0,
    )
    assert (
        native.get_input_embeddings().weight.data_ptr()
        == native.get_output_embeddings().weight.data_ptr()
    )
    assert native.get_input_embeddings().num_embeddings >= max(248320, len(tokenizer))
    assert all(
        tokenizer.convert_tokens_to_ids(name) == value
        for name, value in old_special_ids.items()
    )
    assert len(set(bridge["code_token_ids"])) == bridge["fast_vocab_size"]
    assert bridge["begin_token_id"] not in bridge["code_token_ids"]
    assert bridge["end_token_id"] not in bridge["code_token_ids"]
    assert fast.attach(native, tokenizer) == {
        **bridge,
        "original_model_vocab_rows": bridge["model_vocab_rows"],
        "original_tokenizer_size": len(tokenizer),
    }
    device = native.get_input_embeddings().weight.device
    prefix = {
        "input_ids": torch.tensor([[0, 3, 4, 5], [6, 7, 8, 9]], device=device),
        "attention_mask": torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]], device=device),
        "mm_token_type_ids": torch.zeros(2, 4, dtype=torch.long, device=device),
    }
    original_prefix = {key: value.clone() for key, value in prefix.items()}
    targets = fast.encode_qwen(actions)
    ce_inputs = build_ce_inputs(prefix, targets, tokenizer.pad_token_id)
    for key, value in prefix.items():
        torch.testing.assert_close(value, original_prefix[key], rtol=0, atol=0)
    for row, target in enumerate(targets):
        assert (
            ce_inputs["labels"][row][ce_inputs["labels"][row] != -100].tolist()
            == target
        )
        assert (
            ce_inputs["labels"][row][~ce_inputs["attention_mask"][row].bool()] == -100
        ).all()
    vlm = Qwen35VLMWrapper(native, trainable=True)
    config = ActionExpertConfig(
        **{**vars(expert_config("gqa6_vlm_state")), "horizon": 16, "action_dim": 8}
    )
    policy = KIFlowMatchingPolicy(
        vlm, build_action_expert(config, vlm.state_specs).to(device), fast
    )
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    result = policy(prefix, ce_inputs, actions.to(device))
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert (
        native.get_input_embeddings().weight.grad[bridge["end_token_id"]].abs().sum()
        > 0
    )
    assert policy.expert.state_adapters[0].query_input.weight.grad.abs().sum() > 0
    optimizer.step()


def test_ki_empty_action_targets_have_finite_zero_ce_gradients():
    policy = ki_policy().train()
    _, ce_inputs = ki_inputs()
    ce_inputs["labels"].fill_(-100)
    loss = policy.action_ce_loss(ce_inputs)
    assert loss.requires_grad and torch.isfinite(loss) and loss == 0
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in policy.vlm.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(
        torch.isfinite(gradient).all() and not torch.any(gradient)
        for gradient in gradients
    )
    assert all(parameter.grad is None for parameter in policy.expert.parameters())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_compact_gqa_matches_expanded_keys_and_gradients(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    attention = GQACrossAttention(expert_config()).to(device=device, dtype=dtype)
    reference = copy.deepcopy(attention)
    hidden = torch.randn(2, 4, 32, device=device, dtype=dtype, requires_grad=True)
    context = torch.randn(2, 7, 32, device=device, dtype=dtype, requires_grad=True)
    ref_hidden, ref_context = hidden.detach().clone(), context.detach().clone()
    ref_hidden.requires_grad_()
    ref_context.requires_grad_()
    mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1], [1] * 7], device=device).bool()
    output = attention(hidden, context, mask)
    query = reference.q_proj(ref_hidden).view(2, 4, 4, 8).transpose(1, 2)
    key = reference.k_proj(ref_context).view(2, 7, 2, 8).transpose(1, 2)
    value = reference.v_proj(ref_context).view(2, 7, 2, 8).transpose(1, 2)
    expanded = torch.nn.functional.scaled_dot_product_attention(
        query,
        key.repeat_interleave(2, dim=1),
        value.repeat_interleave(2, dim=1),
        attn_mask=mask[:, None, None, :],
    )
    expected = reference.out_proj(expanded.transpose(1, 2).reshape(2, 4, 32))
    tolerance = (
        {"rtol": 0.03, "atol": 0.002}
        if device == "cuda"
        else {"rtol": 2e-5, "atol": 2e-6}
    )
    torch.testing.assert_close(output, expected, **tolerance)
    output.float().square().sum().backward()
    expected.float().square().sum().backward()
    for actual, baseline in zip(
        attention.parameters(), reference.parameters(), strict=True
    ):
        if device == "cuda":
            error = (actual.grad.float() - baseline.grad.float()).norm()
            assert error <= 0.03 * baseline.grad.float().norm() + 1e-5
        else:
            torch.testing.assert_close(actual.grad, baseline.grad, **tolerance)
    for actual, baseline in (
        (hidden.grad, ref_hidden.grad),
        (context.grad, ref_context.grad),
    ):
        if device == "cuda":
            assert (
                actual.float() - baseline.float()
            ).norm() <= 0.03 * baseline.float().norm() + 1e-5
        else:
            torch.testing.assert_close(actual, baseline, **tolerance)


@pytest.mark.parametrize("variant", ["gqa6", "gqa6_gdn3", "gqa6_vlm_state"])
@pytest.mark.parametrize("method", ["euler", "heun"])
def test_sample_reuses_only_fixed_context_projections(variant, method):
    expert = build_action_expert(expert_config(variant), STATE_SPECS).eval()
    policy = FlowMatchingPolicy(None, expert)
    condition = synthetic_condition()
    noise = torch.randn(2, 4, 3)
    steps = 3
    counts = dict.fromkeys(range(len(expert.blocks)), 0)
    hooks = []
    for index, block in enumerate(expert.blocks):

        def count_projection(module, args, result, index=index):
            counts[index] += 1

        hooks.append(
            block.cross_attention.k_proj.register_forward_hook(count_projection)
        )
    try:
        actual = policy.sample(
            condition, steps=steps, method=method, initial_noise=noise
        )
        assert list(counts.values()) == [1] * len(expert.blocks)
        with torch.no_grad():
            expected = noise.clone()
            dt = 1 / steps
            for index in range(steps):
                time = torch.full((2,), index * dt)
                velocity = policy.predict_velocity(expected, time, condition)
                proposal = expected + dt * velocity
                if method == "heun":
                    next_velocity = policy.predict_velocity(
                        proposal, time + dt, condition
                    )
                    expected = expected + dt * (velocity + next_velocity) / 2
                else:
                    expected = proposal
        torch.testing.assert_close(actual, expected)
        counts.update(dict.fromkeys(counts, 0))
        changed = synthetic_condition(batch=1)
        repeated = policy.sample(
            changed, steps=steps, method=method, initial_noise=noise[:1]
        )
        assert repeated.shape == (1, 4, 3)
        assert list(counts.values()) == [1] * len(expert.blocks)
    finally:
        for hook in hooks:
            hook.remove()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FSDP sampling requires CUDA")
def test_sample_context_cache_survives_fsdp_casts(tmp_path):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    dist.init_process_group(
        "nccl", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1
    )
    try:
        expert = build_action_expert(expert_config()).cuda().eval()
        precision = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
        mesh = init_device_mesh("cuda", (1,))
        for block in expert.blocks:
            fully_shard(block, mesh=mesh, mp_policy=precision)
        fully_shard(expert, mesh=mesh, mp_policy=precision)
        policy = FlowMatchingPolicy(None, expert)
        source = synthetic_condition()
        condition = VLMCondition(
            source.hidden_states.cuda(), source.attention_mask.cuda()
        )
        counts = dict.fromkeys(range(len(expert.blocks)), 0)
        hooks = []
        for index, block in enumerate(expert.blocks):

            def count_projection(module, args, result, index=index):
                counts[index] += 1

            hooks.append(
                block.cross_attention.k_proj.register_forward_hook(count_projection)
            )
        try:
            sample = policy.sample(condition, steps=3, method="heun")
            assert sample.shape == (2, 4, 3) and sample.isfinite().all()
            assert list(counts.values()) == [1] * len(expert.blocks)
        finally:
            for hook in hooks:
                hook.remove()
    finally:
        dist.destroy_process_group()


def test_ki_prepared_batch_matches_direct_loss_and_gradients():
    from rlinf.models.embodiment.qwen35_fm.ki_policy import KIFlowMatchingPolicy

    policy = ki_policy().train()
    prefix, ce_inputs = ki_inputs()
    device = prefix["input_ids"].device
    actions = torch.randn(2, 4, 8, device=device)
    mask = torch.ones_like(actions, dtype=torch.bool)
    mask[0, -1] = False
    noise = torch.randn_like(actions)
    timestep = torch.tensor([0.2, 0.8], device=device)
    metadata = KIFlowMatchingPolicy.prepare_robot_metadata(
        {key: value.cpu() for key, value in prefix.items()},
        {key: value.cpu() for key, value in ce_inputs.items()},
        actions.cpu(),
        mask.cpu(),
    )
    metadata = {key: value.to(device) for key, value in metadata.items()}
    results, gradients = [], []
    for prepared in (None, metadata):
        policy.zero_grad(set_to_none=True)
        result = policy(
            prefix,
            ce_inputs,
            actions,
            mask,
            noise=noise,
            timestep=timestep,
            batch_metadata=prepared,
        )
        result["loss"].backward()
        results.append({key: value.detach().clone() for key, value in result.items()})
        gradients.append(
            {
                name: parameter.grad.detach().clone()
                for name, parameter in policy.named_parameters()
                if parameter.grad is not None
            }
        )
    assert gradients[0].keys() == gradients[1].keys()
    for key in results[0]:
        torch.testing.assert_close(results[0][key], results[1][key], rtol=0, atol=0)
    for name in gradients[0]:
        torch.testing.assert_close(
            gradients[0][name], gradients[1][name], rtol=0, atol=0
        )
    with torch.no_grad():
        direct = policy.vl_loss(ce_inputs)
        prepared = policy.vl_loss(ce_inputs, batch_metadata=metadata)
    torch.testing.assert_close(direct, prepared, rtol=0, atol=0)


def test_ki_cpu_metadata_rejects_invalid_prefix_and_labels():
    from rlinf.models.embodiment.qwen35_fm.ki_policy import KIFlowMatchingPolicy

    prefix, ce_inputs = ki_inputs()
    prefix = {key: value.cpu() for key, value in prefix.items()}
    ce_inputs = {key: value.cpu() for key, value in ce_inputs.items()}
    actions = torch.randn(2, 4, 8)
    mask = torch.ones_like(actions, dtype=torch.bool)
    changed = {key: value.clone() for key, value in ce_inputs.items()}
    changed["input_ids"][0, 0] = 20
    with pytest.raises(ValueError, match="identical"):
        KIFlowMatchingPolicy.prepare_robot_metadata(prefix, changed, actions, mask)
    changed = {key: value.clone() for key, value in ce_inputs.items()}
    changed["labels"][0, 1] = changed["input_ids"][0, 1]
    with pytest.raises(ValueError, match="suffix"):
        KIFlowMatchingPolicy.prepare_robot_metadata(prefix, changed, actions, mask)
    changed = {key: value.clone() for key, value in ce_inputs.items()}
    changed["attention_mask"][0, -1] = 0
    with pytest.raises(ValueError, match="padding"):
        KIFlowMatchingPolicy.prepare_ce_metadata(changed)
    empty = {key: value.clone() for key, value in ce_inputs.items()}
    empty["labels"].fill_(-100)
    metadata = KIFlowMatchingPolicy.prepare_robot_metadata(
        prefix, empty, actions, torch.zeros_like(mask)
    )
    assert metadata["ce_token_count"] == metadata["fm_valid_action_scalars"] == 0
    assert metadata["ce_positions"].tolist() == [0]


def test_molmobot_subset_preserves_interleaved_episode_rows(tmp_path):
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    from examples.sft.qwen35_fm_data import MolmoBotSubset

    root = tmp_path / "molmobot"
    (root / "meta/episodes").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "meta/info.json").write_text(
        json.dumps({"features": {}, "data_path": "data/file-{file_index}.parquet"})
    )
    episodes = [
        {
            "episode_index": episode,
            "length": 3,
            "tasks": [f"pick object {episode}"],
            "data/chunk_index": 0,
            "data/file_index": 0,
            "pipeline/provenance": json.dumps({"source_episode_id": episode}),
        }
        for episode in [2, 1]
    ]
    pq.write_table(pa.Table.from_pylist(episodes), root / "meta/episodes/file.parquet")
    rows = []
    for episode, step in [(1, 0), (2, 0), (1, 1), (1, 2), (2, 1), (2, 2)]:
        rows.append(
            {
                "episode_index": episode,
                "action": [float(episode * 10 + step)] * 8,
                "observation.state": [float(episode * 100 + step)] * 9,
                "source.extra.pipeline_supervision_mask": step != 1,
                "source.observation_index": step,
                "source.action_index": step + 1,
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), root / "data/file-0.parquet")
    subset = MolmoBotSubset(root, 2, 2, 2, [], include_state=True)
    assert [row["episode_index"] for row in subset.manifest] == [2, 2, 1, 1]
    for episode, samples in [(2, subset.samples[:2]), (1, subset.samples[2:])]:
        for start, sample in enumerate(samples):
            torch.testing.assert_close(
                sample["actions"][:, 0],
                torch.tensor([episode * 10 + start, episode * 10 + start + 1]).float(),
            )
            assert sample["mask"][:, 0].tolist() == (
                [True, False] if start == 0 else [False, True]
            )
            assert sample["state"].tolist() == [float(episode * 100 + start)] * 9
            assert sample["episode"]["tasks"] == [f"pick object {episode}"]
    assert [row["provenance"]["source_episode_id"] for row in subset.manifest] == [
        2,
        2,
        1,
        1,
    ]


def test_local_training_logs_survive_sync_failure_and_resume(tmp_path):
    from examples.sft.qwen35_ki_pretrain import LocalRunLogger

    output = tmp_path / "shared"
    output.mkdir()
    (output / "metrics.jsonl").write_text('{"step":1}\n')
    local = tmp_path / "local"
    logger = LocalRunLogger(output, local, 3600)
    try:
        logger.metric({"step": 2})
        logger.status({"stage": "training", "step": 2})
        logger.sync()
        assert (output / "metrics.jsonl").read_text() == ('{"step":1}\n{"step": 2}\n')
        for path in output.iterdir():
            path.unlink()
        output.rmdir()
        logger.metric({"step": 3})
        logger.sync()
        assert logger.last_sync_error
        assert (local / "metrics.jsonl").read_text().endswith('{"step": 3}\n')
        output.mkdir()
        (output / "STOP_AFTER_UPDATE").touch()
        logger.sync()
        assert logger.stop_requested
    finally:
        logger.close()
    assert logger.last_sync_error is None
    assert (output / "metrics.jsonl").read_text().endswith('{"step": 3}\n')
    assert (output / "status.json").read_text() == ('{"stage": "training", "step": 2}')


def state_readout_reference(adapter, hidden, states):
    import math

    latent = adapter.query_input(hidden)
    aggregate = torch.zeros_like(latent)
    weights = adapter.layer_weights.softmax(0)
    for weight, (layer, spec) in zip(weights, adapter.state_specs.items(), strict=True):
        heads, key_dim, _ = spec
        query = (
            adapter.queries[str(layer)](latent)
            .reshape(*hidden.shape[:2], heads, key_dim)
            .float()
        )
        query = (
            query
            * torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
            / math.sqrt(key_dim)
        )
        readout = torch.einsum("bthk,bhkv->bthv", query, states[layer].float())
        readout = readout.flatten(-2).to(latent.dtype)
        aggregate = aggregate + weight * adapter.outputs[str(layer)](readout)
    readout = adapter.output(aggregate)
    return adapter.gate(torch.cat((hidden, readout), dim=-1)).sigmoid() * readout


@pytest.mark.parametrize("heterogeneous", [False, True])
def test_batched_state_readout_matches_per_layer_gradients(heterogeneous):
    specs = dict.fromkeys(range(18), (2, 8, 8))
    if heterogeneous:
        specs[1] = (1, 4, 8)
    torch.manual_seed(27)
    adapter = GDNStateReadAdapter(expert_config("gqa6_vlm_state"), specs)
    expected = copy.deepcopy(adapter)
    hidden = torch.randn(2, 4, 32, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    states = {i: torch.randn(2, *spec, requires_grad=True) for i, spec in specs.items()}
    reference_states = {
        i: value.detach().clone().requires_grad_() for i, value in states.items()
    }
    actual = adapter(hidden, states)
    reference = state_readout_reference(expected, reference_hidden, reference_states)
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-7)
    actual.square().sum().backward()
    reference.square().sum().backward()
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=1e-4, atol=1e-8)
    for i in states:
        torch.testing.assert_close(
            states[i].grad, reference_states[i].grad, rtol=1e-4, atol=1e-8
        )
    for (name, parameter), (other_name, other) in zip(
        adapter.named_parameters(), expected.named_parameters(), strict=True
    ):
        assert name == other_name
        torch.testing.assert_close(parameter.grad, other.grad, rtol=1e-4, atol=1e-8)
