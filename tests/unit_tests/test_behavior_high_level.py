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

import pytest

pytest.importorskip("openpi")

from rlinf.data.datasets.openpi_rlinf.behavior.high_level import (  # noqa: E402
    PaligemmaSubtaskTokenizer,
    build_primitive_prompt_intervals,
    build_r0_manifest,
    build_r1_manifest,
    canonicalize_action,
    canonicalize_object_id,
    canonicalize_primitive,
    resolve_primitive_prompt,
)


def test_canonicalize_behavior_object_ids():
    assert canonicalize_object_id("radio_89") == "radio"
    assert canonicalize_object_id("coffee_table_koagbh_0") == "coffee table"
    assert canonicalize_object_id("half_head_cabbage_212_1") == "half head cabbage"
    assert canonicalize_object_id("diced__vidalia_onion") == "diced vidalia onion"
    assert canonicalize_object_id("pillar_candle_91") == "pillar candle"


def test_canonicalize_turning_on_radio_primitives():
    pick_up = {
        "primitive_idx": 0,
        "primitive_description": ["pick up from"],
        "object_id": [["radio_89", "coffee_table_koagbh_0"]],
        "manipulating_object_id": ["radio_89"],
    }
    place = {
        "primitive_idx": 2,
        "primitive_description": ["place on"],
        "object_id": [["radio_89", "coffee_table_koagbh_0"]],
        "manipulating_object_id": ["radio_89"],
    }

    assert canonicalize_primitive(pick_up) == "pick up radio from coffee table"
    assert canonicalize_primitive(place) == "place radio on coffee table"


def test_canonicalize_composite_primitive_with_nested_objects():
    primitive = {
        "primitive_idx": 1,
        "primitive_description": ["pick up from", "place in"],
        "object_id": [
            [["can_of_soda_114", "can_of_soda_115"], "floors_ulujpr_0"],
            [["can_of_soda_114", "can_of_soda_115"], "trash_can_116"],
        ],
        "manipulating_object_id": [
            ["can_of_soda_114", "can_of_soda_115"],
            ["can_of_soda_114", "can_of_soda_115"],
        ],
    }

    assert canonicalize_primitive(primitive) == (
        "pick up can of soda from floors then place can of soda in trash can"
    )


def test_canonicalize_specialized_actions():
    assert (
        canonicalize_action("open door", ["fridge_petcxr_0"], []) == "open fridge door"
    )
    assert (
        canonicalize_action(
            "chop",
            ["carving_knife_209", ["head_cabbage_212", "half_head_cabbage_212_0"]],
            "carving_knife_209",
        )
        == "chop head cabbage"
    )


def test_primitive_prompt_intervals_exclude_cross_boundary_action_chunks():
    annotation = {
        "skill_annotation": [],
        "primitive_annotation": [
            {
                "primitive_idx": 0,
                "primitive_description": ["press"],
                "object_id": [["radio_89"]],
                "manipulating_object_id": ["radio_89"],
                "frame_duration": [10, 20],
            },
            {
                "primitive_idx": 1,
                "primitive_description": ["place on"],
                "object_id": [["radio_89", "coffee_table_koagbh_0"]],
                "manipulating_object_id": ["radio_89"],
                "frame_duration": [20, 40],
            },
        ],
    }

    intervals = build_primitive_prompt_intervals(annotation)

    assert resolve_primitive_prompt(intervals, 10, action_horizon=10).subtask == (
        "press radio"
    )
    assert resolve_primitive_prompt(intervals, 11, action_horizon=10) is None
    assert resolve_primitive_prompt(intervals, 20, action_horizon=20).subtask == (
        "place radio on coffee table"
    )
    assert resolve_primitive_prompt(intervals, 9) is None


def test_primitive_prompt_intervals_exclude_ambiguous_overlap_frames():
    annotation = {
        "skill_annotation": [],
        "primitive_annotation": [
            {
                "primitive_idx": index,
                "primitive_description": ["press"],
                "object_id": [["radio_89"]],
                "manipulating_object_id": ["radio_89"],
                "frame_duration": duration,
            }
            for index, duration in enumerate(([0, 10], [9, 20]))
        ],
    }

    intervals = build_primitive_prompt_intervals(annotation)

    assert resolve_primitive_prompt(intervals, 5).primitive_index == 0
    assert resolve_primitive_prompt(intervals, 9) is None
    assert resolve_primitive_prompt(intervals, 10).primitive_index == 1


def test_primitive_prompt_intervals_use_disjoint_referenced_skills():
    annotation = {
        "skill_annotation": [
            {"skill_idx": index, "frame_duration": duration}
            for index, duration in enumerate(
                ([0, 10], [10, 20], [20, 30], [30, 40])
            )
        ],
        "primitive_annotation": [
            {
                "primitive_idx": 0,
                "primitive_description": ["open door"],
                "object_id": [["fridge_dszchb_0"]],
                "manipulating_object_id": [],
                "frame_duration": [0, 40],
                "skill_idxes": [0, 3],
            },
            {
                "primitive_idx": 1,
                "primitive_description": ["press"],
                "object_id": [["radio_89"]],
                "manipulating_object_id": ["radio_89"],
                "frame_duration": [10, 30],
                "skill_idxes": [1, 2],
            },
        ],
    }

    intervals = build_primitive_prompt_intervals(annotation)

    assert [(item.start_frame, item.end_frame) for item in intervals] == [
        (0, 10),
        (10, 30),
        (30, 40),
    ]
    assert resolve_primitive_prompt(intervals, 15).subtask == "press radio"


def test_build_r0_manifest_samples_inside_primitive_ranges(tmp_path):
    meta_dir = tmp_path / "meta"
    annotation_dir = tmp_path / "annotations" / "task-0000"
    meta_dir.mkdir()
    annotation_dir.mkdir(parents=True)
    (meta_dir / "tasks.jsonl").write_text(
        json.dumps(
            {
                "task_index": 0,
                "task_name": "turning_on_radio",
                "task": "Turn on the radio.",
            }
        )
        + "\n"
    )
    (annotation_dir / "episode_00000010.json").write_text(
        json.dumps(
            {
                "primitive_annotation": [
                    {
                        "primitive_idx": 0,
                        "primitive_description": ["press"],
                        "object_id": [["radio_89"]],
                        "manipulating_object_id": ["radio_89"],
                        "frame_duration": [10, 20],
                    }
                ]
            }
        )
    )

    entries = build_r0_manifest(tmp_path, samples_per_primitive=2)

    assert [entry.frame_index for entry in entries] == [13, 16]
    assert all(entry.subtask == "press radio" for entry in entries)


def test_subtask_tokenizer_masks_only_response_and_eos():
    tokenizer = PaligemmaSubtaskTokenizer(max_len=64)

    prefix = tokenizer.tokenize("Turn on the radio.")
    training = tokenizer.tokenize("Turn on the radio.", subtask="press radio")

    prefix_length = int(prefix.input_mask.sum())
    training_length = int(training.input_mask.sum())
    assert not prefix.loss_mask.any()
    assert not training.ar_mask[:prefix_length].any()
    assert training.ar_mask[prefix_length:training_length].all()
    assert training.loss_mask[prefix_length:training_length].all()
    assert training.tokens[training_length - 1] == tokenizer.eos_token_id


def test_build_r1_manifest_splits_whole_episodes(tmp_path):
    meta_dir = tmp_path / "meta"
    annotation_dir = tmp_path / "annotations" / "task-0000"
    video_dir = tmp_path / "videos" / "task-0000"
    meta_dir.mkdir()
    annotation_dir.mkdir(parents=True)
    (meta_dir / "tasks.jsonl").write_text(
        json.dumps(
            {
                "task_index": 0,
                "task_name": "turning_on_radio",
                "task": "Turn on the radio.",
            }
        )
        + "\n"
    )
    for camera in (
        "observation.images.rgb.head",
        "observation.images.rgb.left_wrist",
        "observation.images.rgb.right_wrist",
    ):
        (video_dir / camera).mkdir(parents=True)

    for episode_number in range(10):
        episode_index = episode_number * 10
        (annotation_dir / f"episode_{episode_index:08d}.json").write_text(
            json.dumps(
                {
                    "skill_annotation": [
                        {
                            "skill_idx": 0,
                            "skill_description": ["press"],
                            "object_id": [["radio_89"]],
                            "manipulating_object_id": ["radio_89"],
                            "frame_duration": [10, 20],
                            "skill_type": ["uncoordinated"],
                        }
                    ],
                    "primitive_annotation": [
                        {
                            "primitive_idx": 0,
                            "primitive_description": ["press"],
                            "object_id": [["radio_89"]],
                            "manipulating_object_id": ["radio_89"],
                            "frame_duration": [10, 20],
                            "skill_idxes": [0],
                        }
                    ],
                }
            )
        )
        for camera in (
            "observation.images.rgb.head",
            "observation.images.rgb.left_wrist",
            "observation.images.rgb.right_wrist",
        ):
            (video_dir / camera / f"episode_{episode_index:08d}.mp4").touch()

    entries, report = build_r1_manifest(
        tmp_path,
        samples_per_primitive=2,
        seed=7,
        val_fraction=0.2,
        test_fraction=0.2,
    )

    splits_by_episode = {}
    for entry in entries:
        splits_by_episode.setdefault(entry.episode_index, set()).add(entry.split)
    assert all(len(splits) == 1 for splits in splits_by_episode.values())
    assert report["split_episode_counts"] == {"test": 2, "train": 6, "val": 2}
    assert report["entry_counts"] == {"test": 4, "train": 12, "val": 4}
    assert report["skipped_counts"] == {}
