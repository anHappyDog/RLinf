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

from toolkits.b1k_grounded.skill_audit import (
    build_skill_audit,
    write_skill_audit,
)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _skill(
    index,
    description,
    object_group,
    manipulating,
    *,
    spatial_prefix=None,
    memory_prefix=None,
    skill_type="uncoordinated",
):
    return {
        "skill_idx": index,
        "skill_id": [index + 1],
        "skill_description": [description],
        "object_id": [object_group],
        "manipulating_object_id": manipulating,
        "memory_prefix": memory_prefix or [],
        "spatial_prefix": spatial_prefix or [],
        "frame_duration": [index * 10, (index + 1) * 10],
        "skill_type": [skill_type],
    }


def _dataset(tmp_path):
    meta = tmp_path / "meta" / "tasks.jsonl"
    meta.parent.mkdir()
    meta.write_text(
        "\n".join(
            json.dumps(
                {
                    "task_index": index,
                    "task_name": name,
                    "task": goal,
                }
            )
            for index, name, goal in (
                (0, "turning_on_radio", "Turn on the radio."),
                (1, "putting_away_items", "Put the items away."),
            )
        )
        + "\n"
    )
    _write_json(
        tmp_path / "annotations" / "task-0000" / "episode_00000001.json",
        {
            "skill_annotation": [
                _skill(0, "move to", ["radio_1"], [], skill_type="navigation"),
                _skill(1, "press", ["radio_1"], ["radio_1"]),
            ],
            "primitive_annotation": [
                {"primitive_description": ["press"]},
                {"primitive_description": []},
            ],
        },
    )
    _write_json(
        tmp_path / "annotations" / "task-0001" / "episode_00010001.json",
        {
            "skill_annotation": [
                _skill(
                    0,
                    "pick up from",
                    [["cup_1", "cup_2"], "table_1"],
                    ["cup_1", "cup_2"],
                    memory_prefix=["the other"],
                    spatial_prefix=[["", "left"]],
                ),
                _skill(1, "place in", ["cup_1", "cup_1"], ["cup_1"]),
            ],
            "primitive_annotation": [
                {"primitive_description": ["pick up from", "place in"]}
            ],
        },
    )
    return tmp_path


def test_build_skill_audit_preserves_real_annotation_shapes(tmp_path):
    dataset_root = _dataset(tmp_path)
    annotation_path = (
        dataset_root / "annotations" / "task-0000" / "episode_00000001.json"
    )
    annotation = json.loads(annotation_path.read_text())
    annotation["skill_annotation"][0]["frame_duration"] = [[0, 4], [6, 10]]
    _write_json(annotation_path, annotation)

    report = build_skill_audit(dataset_root, max_examples_per_skill=3)

    assert report["dataset"] == {
        "root": str(tmp_path.resolve()),
        "annotation_file_count": 2,
        "task_metadata_count": 2,
        "observed_task_count": 2,
        "observed_task_indices": [0, 1],
        "skill_annotation_count": 4,
        "accepted_skill_annotation_count": 4,
        "canonical_skill_count": 4,
        "primitive_annotation_count": 3,
    }
    skills = {item["skill"]: item for item in report["skills"]}
    pick_up = skills["pick up from"]
    assert pick_up["object_arity_counts"] == [{"arity": 3, "count": 1}]
    assert pick_up["manipulating_object_arity_counts"] == [{"arity": 2, "count": 1}]
    assert pick_up["object_structure_counts"] == [
        {"shape": "list[list[str*2],str]", "count": 1}
    ]
    assert pick_up["examples"][0]["object_id"] == [[["cup_1", "cup_2"], "table_1"]]
    assert pick_up["memory_prefix_values"] == [{"value": ["the other"], "count": 1}]
    move_to = skills["move to"]
    assert move_to["frame_interval_count_counts"] == [{"interval_count": 2, "count": 1}]
    assert move_to["issues"]["total"] == 0


def test_build_skill_audit_reports_annotation_anomalies(tmp_path):
    report = build_skill_audit(_dataset(tmp_path))

    assert report["issues"]["counts"] == {"duplicate_flattened_object_id": 1}
    place_in = next(item for item in report["skills"] if item["skill"] == "place in")
    assert place_in["issues"]["counts"] == {"duplicate_flattened_object_id": 1}
    issue = place_in["issues"]["examples"]["duplicate_flattened_object_id"][0]
    assert issue["annotation_path"] == ("annotations/task-0001/episode_00010001.json")
    assert issue["object_id"] == [["cup_1", "cup_1"]]


def test_write_skill_audit_round_trips_json(tmp_path):
    report = build_skill_audit(_dataset(tmp_path))
    output = tmp_path / "output" / "skill_audit.json"

    write_skill_audit(report, output)

    assert json.loads(output.read_text()) == report
