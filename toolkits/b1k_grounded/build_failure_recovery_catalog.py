#!/usr/bin/env python3
"""Build a recovery subpool from simulator-certified failure artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlinf.envs.behavior.subpool import (
    SubpoolCatalog,
    SubpoolSnapshot,
    SubpoolStore,
)


def build_recovery_catalog(
    failure_root: Path,
    canonical_manifest: Path,
    output_manifest: Path,
    failure_ids: set[str] | None = None,
) -> dict[str, int]:
    """Convert eligible terminal failures into a validated recovery catalog."""
    failure_root = failure_root.resolve()
    output_manifest = output_manifest.resolve()
    if output_manifest.exists() or output_manifest.parent.exists():
        raise FileExistsError(
            f"Refusing to overwrite recovery catalog at {output_manifest.parent}."
        )

    canonical_catalog = SubpoolCatalog.from_jsonl(canonical_manifest)
    metadata_paths = sorted(failure_root.rglob("failure-*.json"))
    selected: list[tuple[Path, dict]] = []
    for metadata_path in metadata_paths:
        record = json.loads(metadata_path.read_text(encoding="utf-8"))
        recovery = record.get("analysis", {}).get("recovery", {})
        if (
            record.get("capture_kind") == "terminal"
            and recovery.get("status") == "eligible"
            and (failure_ids is None or record.get("failure_id") in failure_ids)
        ):
            selected.append((metadata_path, record))
    if failure_ids is not None:
        selected_ids = {record["failure_id"] for _, record in selected}
        missing_ids = failure_ids - selected_ids
        if missing_ids:
            raise ValueError(
                "Requested failure IDs are absent or not eligible terminal "
                f"artifacts: {sorted(missing_ids)}."
            )
    if not selected:
        raise ValueError(f"No eligible terminal failures found under {failure_root}.")

    source_ids = {
        str(record["source_snapshot"]["snapshot_id"])
        for _, record in selected
    }
    canonical_by_id = {
        record.snapshot_id: record for record in canonical_catalog.records
    }
    missing = source_ids - canonical_by_id.keys()
    if missing:
        raise ValueError(
            "Failure artifacts reference snapshots absent from the canonical "
            f"manifest: {sorted(missing)}."
        )

    output_manifest.parent.mkdir(parents=True)
    store = SubpoolStore(output_manifest)
    for snapshot_id in sorted(source_ids):
        source = canonical_by_id[snapshot_id]
        store.append_from_file(source, canonical_catalog.state_path(source))

    for metadata_path, failure in selected:
        source = SubpoolSnapshot.from_dict(failure["source_snapshot"])
        facts = failure["analysis"]["facts"]
        metadata = dict(source.metadata)
        metadata["recovery_provenance"] = {
            "failure_id": failure["failure_id"],
            "failure_metadata_path": str(metadata_path.relative_to(failure_root)),
            "capture_kind": failure["capture_kind"],
            "recovery_status": failure["analysis"]["recovery"]["status"],
            "failure_tags": list(failure["analysis"]["failure_tags"]),
            "reference_orientation_xyzw": list(
                facts["reference_orientation_xyzw"]
            ),
            "terminal_tilt_angle_deg": float(facts["tilt_angle_deg"]),
        }
        recovery_id = f"recovery-terminal-{failure['failure_id'].removeprefix('failure-')}"
        recovery = SubpoolSnapshot(
            snapshot_id=recovery_id,
            state_path=f"states/{recovery_id}.pt",
            state_sha256=failure["state_sha256"],
            activity_name=source.activity_name,
            scene_model=source.scene_model,
            asset_fingerprint=source.asset_fingerprint,
            subtask_id=source.subtask_id,
            skill=source.skill,
            pool_type="recovery",
            task_description=source.task_description,
            control_json=source.control_json,
            episode_index=source.episode_index,
            frame_index=None,
            metadata=metadata,
        )
        state_path = metadata_path.parent / failure["state_path"]
        store.append_from_file(recovery, state_path)

    catalog = SubpoolCatalog.from_jsonl(output_manifest)
    recovery_count = sum(record.pool_type == "recovery" for record in catalog.records)
    return {
        "canonical_snapshots": len(catalog.records) - recovery_count,
        "recovery_snapshots": recovery_count,
        "total_snapshots": len(catalog.records),
    }


def main() -> None:
    """Parse CLI arguments and build the catalog."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-root", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument(
        "--failure-id",
        action="append",
        dest="failure_ids",
        help="Include only this eligible terminal failure ID; repeat as needed.",
    )
    args = parser.parse_args()
    summary = build_recovery_catalog(
        args.failure_root,
        args.canonical_manifest,
        args.output_manifest,
        None if args.failure_ids is None else set(args.failure_ids),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
