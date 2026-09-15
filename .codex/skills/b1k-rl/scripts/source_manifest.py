#!/usr/bin/env python3
"""Create or verify deterministic manifests for deployed source trees."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def _run_git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _tracked_files(root: Path, includes: tuple[str, ...]) -> list[Path]:
    output = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    files = []
    for raw_path in output.split(b"\0"):
        if not raw_path:
            continue
        relative = Path(os.fsdecode(raw_path))
        if includes and not any(
            relative == Path(prefix) or Path(prefix) in relative.parents
            for prefix in includes
        ):
            continue
        if (root / relative).is_file() or (root / relative).is_symlink():
            files.append(relative)
    return sorted(files, key=lambda path: path.as_posix())


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_symlink():
        digest.update(b"symlink\0")
        digest.update(os.readlink(path).encode())
        return digest.hexdigest()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_manifest(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    if _run_git(root, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError(f"Refusing to manifest dirty tracked files below {root}")
    files = _tracked_files(root, tuple(args.include))
    if not files:
        raise RuntimeError("Manifest scope contains no tracked files")
    payload = {
        "format_version": 1,
        "label": args.label,
        "commit": _run_git(root, "rev-parse", "HEAD"),
        "tree": _run_git(root, "rev-parse", "HEAD^{tree}"),
        "includes": args.include,
        "files": [
            {
                "path": relative.as_posix(),
                "size": (root / relative).lstat().st_size,
                "sha256": _digest(root / relative),
            }
            for relative in files
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"created {args.output}: label={args.label} commit={payload['commit']} "
        f"files={len(files)}"
    )


def verify_manifest(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    failures = []
    for record in payload["files"]:
        path = root / record["path"]
        if not path.exists() and not path.is_symlink():
            failures.append(f"missing {record['path']}")
            continue
        actual_size = path.lstat().st_size
        actual_digest = _digest(path)
        if actual_size != record["size"] or actual_digest != record["sha256"]:
            failures.append(
                f"mismatch {record['path']}: size={actual_size}, sha256={actual_digest}"
            )
    if failures:
        detail = "\n".join(failures[:50])
        raise RuntimeError(
            f"Source verification failed for {len(failures)} file(s):\n{detail}"
        )
    print(
        f"verified {args.manifest}: label={payload['label']} "
        f"commit={payload['commit']} files={len(payload['files'])} root={root}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create")
    create.add_argument("--root", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--label", required=True)
    create.add_argument("--include", action="append", default=[])
    create.set_defaults(func=create_manifest)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.set_defaults(func=verify_manifest)
    return parser.parse_args()


def main() -> None:
    """Run the selected manifest operation."""
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
