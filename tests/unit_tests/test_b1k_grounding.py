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

import json

import numpy as np
import pyarrow as pa
import pytest

from rlinf.data.b1k_grounded import (
    CameraID,
    EntityArgument,
    EntityResolver,
    GroundedControlSpec,
    Grounding2D,
    GroundingConfig,
    PartArgument,
    Role,
    filter_mask_components,
    ground_button_instance_ids,
    ground_control_spec,
    grounding_from_mask,
    mask_for_instance_ids,
    parse_instance_id_mapping,
    select_primary_grounding,
)
from toolkits.b1k_grounded.build_pilot_dataset import (
    GroundingAssessment,
    PilotBuildConfig,
    _select_episode_candidates,
    assess_grounding,
    extract_action_chunk,
    pilot_arrow_schema,
    select_interval_frames,
    select_interval_frames_by_stride,
)
from toolkits.b1k_grounded.mapping_coverage import build_mapping_coverage
from toolkits.b1k_grounded.recorded_segmentation import (
    decode_segmentation_rgb,
    decode_segmentation_rgb_fast,
    generate_segmentation_palette,
)


def test_mask_union_bbox_and_in_mask_point():
    segmentation = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 8, 0, 9, 0],
            [0, 8, 0, 0, 0],
            [0, 0, 0, 9, 0],
        ]
    )
    mask = mask_for_instance_ids(segmentation, {8, 9})
    grounding = grounding_from_mask(mask, CameraID.HEAD)

    assert grounding.bbox_xyxy == (0.2, 0.25, 0.8, 1.0)
    assert grounding.visible_pixels == 4
    assert grounding.visible_fraction == 0.2
    point_x = min(int(grounding.point_xy[0] * 5), 4)
    point_y = min(int(grounding.point_xy[1] * 4), 3)
    assert mask[point_y, point_x]


def test_empty_and_below_threshold_masks_have_no_grounding():
    empty = np.zeros((10, 10), dtype=bool)
    tiny = empty.copy()
    tiny[2, 3] = True

    assert grounding_from_mask(empty, CameraID.HEAD) is None
    assert (
        grounding_from_mask(
            tiny,
            CameraID.HEAD,
            config=GroundingConfig(min_visible_pixels=2),
        )
        is None
    )


def test_component_filter_removes_speckles_but_keeps_largest_component():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:10, 6:12] = True
    mask[0, 0] = True
    mask[18:20, 18:20] = True

    filtered = filter_mask_components(
        mask,
        min_component_pixels=5,
        min_component_fraction_of_largest=0.1,
    )

    assert filtered.sum() == 30
    assert filtered[5:10, 6:12].all()


def test_component_filter_keeps_tiny_object_largest_component():
    mask = np.zeros((8, 8), dtype=bool)
    mask[2, 2] = True
    mask[6, 6] = True

    filtered = filter_mask_components(
        mask,
        min_component_pixels=64,
        min_component_fraction_of_largest=0.5,
    )

    assert filtered.sum() == 1


def test_primary_view_uses_fraction_then_camera_priority():
    head = grounding_from_mask(np.ones((10, 10)), CameraID.HEAD)
    left = grounding_from_mask(np.ones((5, 5)), CameraID.LEFT_WRIST)

    assert (
        select_primary_grounding(
            {CameraID.LEFT_WRIST: left, CameraID.HEAD: head}
        ).camera
        is CameraID.HEAD
    )


def test_entity_resolver_uses_exact_object_component_and_multi_prim_union():
    resolver = EntityResolver(
        parse_instance_id_mapping(
            {
                "0": "background",
                "1": "unlabelled",
                "8": "/World/scene_0/radio_89/button/visuals/mesh_0",
                "9": "/World/scene_0/radio_89/base_link/visuals",
                "10": "/World/scene_0/radio_890/base_link/visuals",
            }
        )
    )

    assert resolver.resolve("radio_89") == (8, 9)
    assert resolver.resolve("radio_89", visible_instance_ids=[9, 10]) == (9,)
    assert resolver.resolve("robot") == ()


def test_entity_resolver_handles_particle_system_nested_under_surface():
    resolver = EntityResolver(
        {
            46: "/World/scene_0/floor_1/base_link/mudParticle0",
            47: "/World/scene_0/floor_1/base_link/mudParticle1",
            48: "/World/scene_0/floor_1/base_link/not_mud/visuals",
        }
    )

    assert resolver.resolve("mud") == (46, 47)


def test_part_resolver_handles_separator_and_door_leaf_names():
    resolver = EntityResolver(
        {
            1: "/World/scene_0/fridge_1/right_door/visuals",
            2: "/World/scene_0/fridge_1/leftdoor/visuals",
            3: "/World/scene_0/fridge_1/leaf/visuals",
            4: "/World/scene_0/fridge_1/base_link/visuals",
        }
    )

    assert resolver.resolve_part("fridge_1", "right door") == (1,)
    assert resolver.resolve_part("fridge_1", "door") == (1, 2, 3)


def test_part_resolver_falls_back_to_articulated_non_base_link():
    resolver = EntityResolver(
        {
            10: "/World/scene_0/fridge_1/base_link/visuals",
            16: "/World/scene_0/fridge_1/link_0/visuals",
        }
    )

    assert resolver.resolve_part("fridge_1", "door") == (16,)


def test_part_resolver_excludes_button_from_generic_door_fallback():
    resolver = EntityResolver(
        {
            10: "/World/scene_0/microwave_1/base_link/visuals",
            11: "/World/scene_0/microwave_1/link_0/visuals",
            12: "/World/scene_0/microwave_1/glass/visuals",
            13: "/World/scene_0/microwave_1/togglebutton/visuals",
        }
    )

    assert resolver.resolve_part("microwave_1", "door") == (11, 12)


def test_ground_control_spec_attaches_object_and_part_groundings():
    resolver = EntityResolver(
        {
            8: "/World/scene_0/fridge_1/rightdoor/visuals",
            9: "/World/scene_0/fridge_1/base_link/visuals",
        }
    )
    control = GroundedControlSpec(
        goal="Open the fridge.",
        subgoal=None,
        skill="open door",
        arguments=(
            EntityArgument(
                role=Role.TARGET,
                category_name="fridge",
                instance_id="fridge_1",
                qualifier=None,
                part=PartArgument(name="right door"),
                raw_object_id="fridge_1",
            ),
        ),
    )
    segmentation = np.array([[0, 8], [9, 9]])

    grounded = ground_control_spec(
        control, {CameraID.HEAD: segmentation}, resolver, timestep=12
    )

    argument = grounded.arguments[0]
    assert argument.groundings[CameraID.HEAD].visible_pixels == 3
    assert argument.part.groundings[CameraID.HEAD].visible_pixels == 1
    assert grounded.timestep == 12


def test_ground_control_spec_infers_segmented_button_for_press_target():
    resolver = EntityResolver(
        {
            8: "/World/scene_0/radio_89/meta__base_link_togglebutton_0_0_link/visuals/mesh_0",
            9: "/World/scene_0/radio_89/base_link/visuals",
        }
    )
    control = GroundedControlSpec(
        goal="Turn on the radio.",
        subgoal="Press the radio button.",
        skill="press",
        arguments=(
            EntityArgument(
                role=Role.TARGET,
                category_name="radio",
                instance_id="radio_89",
                qualifier=None,
                raw_object_id="radio_89",
            ),
        ),
    )
    segmentation = np.array([[0, 8], [9, 9]])

    grounded = ground_control_spec(control, {CameraID.HEAD: segmentation}, resolver)

    target = grounded.arguments[0]
    assert target.part is not None
    assert target.part.name == "toggle button"
    assert target.part.groundings[CameraID.HEAD].visible_pixels == 1


@pytest.mark.parametrize("skill", ["turn on switch", "turn off switch"])
def test_ground_control_spec_infers_segmented_button_for_switch_target(skill):
    resolver = EntityResolver(
        {
            8: "/World/scene_0/microwave_1/togglebutton/visuals",
            9: "/World/scene_0/microwave_1/base_link/visuals",
        }
    )
    control = GroundedControlSpec(
        goal="Operate the microwave.",
        subgoal=None,
        skill=skill,
        arguments=(
            EntityArgument(
                role=Role.TARGET,
                category_name="microwave",
                instance_id="microwave_1",
                qualifier=None,
                raw_object_id="microwave_1",
            ),
        ),
    )

    grounded = ground_control_spec(
        control,
        {CameraID.HEAD: np.array([[0, 8], [9, 9]])},
        resolver,
    )

    target = grounded.arguments[0]
    assert target.part is not None
    assert target.part.name == "toggle button"
    assert target.part.groundings[CameraID.HEAD].visible_pixels == 1


def test_ground_control_spec_does_not_infer_button_without_registry_match():
    resolver = EntityResolver({9: "/World/scene_0/radio_89/base_link/visuals"})
    control = GroundedControlSpec(
        goal="Turn on the radio.",
        subgoal=None,
        skill="press",
        arguments=(
            EntityArgument(
                role=Role.TARGET,
                category_name="radio",
                instance_id="radio_89",
                qualifier=None,
                raw_object_id="radio_89",
            ),
        ),
    )

    grounded = ground_control_spec(
        control,
        {CameraID.HEAD: np.array([[9]])},
        resolver,
    )

    assert grounded.arguments[0].part is None


def test_ground_control_spec_can_preserve_legacy_object_only_press_target():
    resolver = EntityResolver(
        {
            8: "/World/scene_0/radio_89/meta__base_link_togglebutton_0_0_link/visuals/mesh_0",
            9: "/World/scene_0/radio_89/base_link/visuals",
        }
    )
    control = GroundedControlSpec(
        goal="Turn on the radio.",
        subgoal="Press the radio button.",
        skill="press",
        arguments=(
            EntityArgument(
                role=Role.TARGET,
                category_name="radio",
                instance_id="radio_89",
                qualifier=None,
                raw_object_id="radio_89",
            ),
        ),
    )

    grounded = ground_control_spec(
        control,
        {CameraID.HEAD: np.array([[8, 9, 9]])},
        resolver,
        infer_missing_parts=False,
    )

    target = grounded.arguments[0]
    assert target.part is None
    assert target.groundings[CameraID.HEAD].visible_pixels == 3


def test_button_grounding_rejects_larger_component_away_from_parent():
    resolver = EntityResolver(
        {
            8: "/World/scene_0/radio_89/togglebutton/visuals",
            9: "/World/scene_0/radio_89/base_link/visuals",
        }
    )
    control = GroundedControlSpec(
        goal="Turn on the radio.",
        subgoal=None,
        skill="press",
        arguments=(
            EntityArgument(
                role=Role.TARGET,
                category_name="radio",
                instance_id="radio_89",
                qualifier=None,
                raw_object_id="radio_89",
            ),
        ),
    )
    segmentation = np.zeros((20, 20), dtype=np.int32)
    segmentation[1:9, 11:19] = 9
    segmentation[3:5, 14:16] = 8
    segmentation[12:19, 1:9] = 8

    grounded = ground_control_spec(
        control,
        {CameraID.HEAD: segmentation},
        resolver,
        config=GroundingConfig(min_visible_pixels=1),
    )

    target = grounded.arguments[0]
    assert target.groundings[CameraID.HEAD].bbox_xyxy == (0.55, 0.05, 0.95, 0.45)
    assert target.part is not None
    assert target.part.groundings[CameraID.HEAD].bbox_xyxy == (
        0.7,
        0.15,
        0.8,
        0.25,
    )


def test_button_grounding_rejects_sparse_codec_artifact():
    segmentation = np.full((21, 21), 9, dtype=np.int32)
    np.fill_diagonal(segmentation, 8)

    groundings = ground_button_instance_ids(
        {CameraID.HEAD: segmentation},
        button_instance_ids=(8,),
        parent_instance_ids=(9,),
        config=GroundingConfig(min_visible_pixels=1),
    )

    assert groundings == {}


def test_recorded_segmentation_palette_round_trip_without_codec_noise():
    instance_ids = (3, 8, 21, 34, 55)
    palette = generate_segmentation_palette(len(instance_ids))
    encoded = palette[np.array([[0, 1, 2], [3, 4, 0]])]

    decoded, diagnostics = decode_segmentation_rgb(encoded, instance_ids)

    assert decoded.tolist() == [[3, 8, 21], [34, 55, 3]]
    assert diagnostics.mean_color_error == 0.0
    assert diagnostics.p99_color_error == 0.0
    assert diagnostics.ambiguous_pixel_fraction == 0.0


def test_fast_segmentation_decoder_matches_exhaustive_nearest_palette():
    rng = np.random.default_rng(7)
    encoded = rng.integers(0, 256, size=(9, 11, 3), dtype=np.uint8)

    for num_ids in (1, 5, 10, 74, 127):
        instance_ids = tuple(range(100, 100 + num_ids))
        exhaustive, _ = decode_segmentation_rgb(encoded, instance_ids)

        assert np.array_equal(
            decode_segmentation_rgb_fast(encoded, instance_ids), exhaustive
        )


def test_sampled_mapping_coverage_resolves_annotation_object(tmp_path):
    (tmp_path / "meta" / "episodes" / "task-0000").mkdir(parents=True)
    (tmp_path / "annotations" / "task-0000").mkdir(parents=True)
    (tmp_path / "meta" / "tasks.jsonl").write_text(
        json.dumps(
            {"task_index": 0, "task_name": "radio", "task": "Turn on the radio."}
        )
        + "\n"
    )
    annotation = {
        "skill_annotation": [
            {
                "skill_idx": 0,
                "skill_id": [1],
                "skill_description": ["move to"],
                "object_id": [["radio_89"]],
                "manipulating_object_id": [],
                "memory_prefix": [],
                "spatial_prefix": [],
                "frame_duration": [0, 10],
                "skill_type": ["navigation"],
            }
        ]
    }
    (tmp_path / "annotations/task-0000/episode_00000001.json").write_text(
        json.dumps(annotation)
    )
    metadata = {
        "ins_id_mapping": json.dumps(
            {
                "0": "background",
                "8": "/World/scene_0/radio_89/button/visuals",
                "9": "/World/scene_0/radio_89/base_link/visuals",
            }
        )
    }
    (tmp_path / "meta/episodes/task-0000/episode_00000001.json").write_text(
        json.dumps(metadata)
    )

    report = build_mapping_coverage(tmp_path)

    assert report["arguments"]["resolved_arguments"] == 1
    assert report["arguments"]["resolution_fraction"] == 1.0
    assert report["mesh_count_histogram"] == {"2": 1}
    assert report["unresolved_objects"] == {}


def test_pilot_interval_sampling_is_half_open_and_deduplicated():
    assert select_interval_frames(10, 20, (0.0, 0.5, 1.0)) == (
        (10, 0.0),
        (14, 0.5),
        (19, 1.0),
    )
    assert select_interval_frames(3, 4, (0.0, 0.5, 1.0)) == ((3, 0.0),)


def test_pilot_stride_sampling_covers_start_regular_steps_and_final_frame():
    assert select_interval_frames_by_stride(10, 20, 4) == (
        (10, 0.0),
        (14, 4 / 9),
        (18, 8 / 9),
        (19, 1.0),
    )
    assert select_interval_frames_by_stride(3, 4, 8) == ((3, 0.0),)


def test_pilot_selective_stride_keeps_other_skills_at_midpoint():
    config = PilotBuildConfig(
        sample_fractions=(0.5,),
        frame_stride=4,
        frame_stride_skills=("press",),
    )

    assert config.sample_interval(10, 20, "press") == (
        (10, 0.0),
        (14, 4 / 9),
        (18, 8 / 9),
        (19, 1.0),
    )
    assert config.sample_interval(10, 20, "pick up from") == ((14, 0.5),)


def test_pilot_selective_stride_requires_stride():
    with pytest.raises(ValueError, match="requires frame_stride"):
        PilotBuildConfig(frame_stride_skills=("press",))


def test_pilot_action_chunk_matches_sft_episode_end_clamping():
    actions = np.arange(15, dtype=np.float32).reshape(5, 3)

    chunk, is_padding = extract_action_chunk(actions, 3, 4)

    assert chunk.tolist() == [actions[3].tolist(), *([actions[4].tolist()] * 3)]
    assert is_padding.tolist() == [False, False, True, True]


def test_pilot_action_chunk_stops_at_skill_boundary():
    actions = np.arange(24, dtype=np.float32).reshape(8, 3)

    chunk, is_padding = extract_action_chunk(
        actions,
        frame_index=2,
        action_horizon=5,
        end_index=5,
    )

    assert chunk.tolist() == [
        actions[2].tolist(),
        actions[3].tolist(),
        actions[4].tolist(),
        actions[4].tolist(),
        actions[4].tolist(),
    ]
    assert is_padding.tolist() == [False, False, False, True, True]


def test_pilot_action_chunk_rejects_invalid_skill_boundary():
    actions = np.zeros((8, 3), dtype=np.float32)

    with np.testing.assert_raises_regex(ValueError, "end_index"):
        extract_action_chunk(actions, frame_index=3, action_horizon=4, end_index=3)


def test_pilot_assessment_keeps_unresolved_part_explicit():
    head = Grounding2D(
        camera=CameraID.HEAD,
        bbox_xyxy=(0.1, 0.1, 0.4, 0.5),
        visible_pixels=20,
        visible_fraction=0.1,
        point_xy=(0.2, 0.3),
    )
    control = GroundedControlSpec(
        goal="Open the left door.",
        subgoal=None,
        skill="open door",
        arguments=(
            EntityArgument(
                role=Role.TARGET,
                category_name="cabinet",
                instance_id="cabinet_1",
                qualifier=None,
                groundings={CameraID.HEAD: head},
                part=PartArgument(name="left door"),
                raw_object_id="cabinet_1",
            ),
        ),
    )
    resolver = EntityResolver(
        {
            1: "/World/scene_0/cabinet_1/base_link/visuals",
            2: "/World/scene_0/cabinet_1/link_0/visuals",
        }
    )

    assessment = assess_grounding(control, resolver)

    assert assessment.object_grounding_complete
    assert not assessment.part_grounding_complete
    assert assessment.issues == ("unresolved_part:cabinet_1/left door",)
    assert assessment.primary_cameras == ("head",)
    assert assessment.primary_visible_fraction == 0.1


def test_pilot_best_visibility_selection_prefers_groundability_then_midpoint():
    def candidate(
        segment: int,
        fraction: float,
        *,
        complete: bool,
        visible_fraction: float,
    ):
        row = {
            "segment_index": segment,
            "interval_index": 0,
            "sample_fraction": fraction,
            "frame_index": int(100 * fraction),
        }
        assessment = GroundingAssessment(
            issues=(),
            object_grounding_complete=complete,
            part_grounding_complete=True,
            visible_arguments=int(complete),
            groundable_arguments=1,
            primary_cameras=("head",),
            primary_visible_fraction=visible_fraction,
        )
        return row, assessment

    candidates = [
        candidate(0, 0.1, complete=True, visible_fraction=0.1),
        candidate(0, 0.5, complete=False, visible_fraction=0.9),
        candidate(1, 0.1, complete=True, visible_fraction=0.2),
        candidate(1, 0.5, complete=True, visible_fraction=0.2),
        candidate(1, 0.9, complete=True, visible_fraction=0.2),
    ]

    selected = _select_episode_candidates(candidates, "best_visibility")

    assert [row["sample_fraction"] for row, _ in selected] == [0.1, 0.5]


def test_pilot_build_config_rejects_unknown_selection_mode():
    with np.testing.assert_raises_regex(ValueError, "selection_mode"):
        PilotBuildConfig(selection_mode="unknown")


def test_pilot_arrow_schema_fixes_state_and_action_shapes():
    schema = pilot_arrow_schema(state_dim=256, action_dim=23, action_horizon=32)

    assert schema.field("state").type.list_size == 256
    assert schema.field("actions").type.list_size == 32
    assert schema.field("actions").type.value_type.list_size == 23
    assert schema.field("action_is_pad").type.list_size == 32
    assert schema.field("has_part_argument").type == pa.bool_()
