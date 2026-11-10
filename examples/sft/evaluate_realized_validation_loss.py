# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Fixed episode-balanced validation of realized-trajectory OpenPI checkpoints."""

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from torch.utils.data import default_collate

LOG = logging.getLogger(__name__)


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def prepare(args):
    from rlinf.data.datasets.openpi_official import RealizedOfficialMixtureDataset

    dataset = RealizedOfficialMixtureDataset(
        args.data, model_path=args.model, split="val", action_horizon=16, cache_files=8
    )
    grouped = {}
    for i, row in enumerate(dataset.episodes):
        start = int(dataset.ends[i - 1]) if i else 0
        grouped.setdefault(row["schema_id"], {}).setdefault(
            row["source_group_id"], []
        ).extend(
            (start + j, row["catalog_id"], int(anchor))
            for j, anchor in enumerate(row["anchor_indices"])
        )
    rng = np.random.default_rng(args.seed)
    selected, counts = [], {}
    for source, episodes in sorted(grouped.items()):
        names = sorted(episodes)
        rng.shuffle(names)
        pools = {}
        for name in names:
            pool = episodes[name]
            rng.shuffle(pool)
            pools[name] = iter(pool)
        chosen = []
        while pools and len(chosen) < args.samples:
            for name in list(pools):
                item = next(pools[name], None)
                if item is None:
                    del pools[name]
                else:
                    chosen.append(
                        {
                            "source": source,
                            "episode": name,
                            "index": item[0],
                            "catalog_id": item[1],
                            "anchor": item[2],
                        }
                    )
                if len(chosen) == args.samples:
                    break
        selected.extend(chosen)
        counts[source] = {
            "samples": len(chosen),
            "independent_episodes": len({x["episode"] for x in chosen}),
            "available_episodes": len(episodes),
        }
    metadata = {
        "seed": args.seed,
        "samples_per_source": args.samples,
        "counts": counts,
        "manifest_fingerprint": dataset.manifest_fingerprint,
        "data": str(Path(args.data).resolve()),
        "model": str(Path(args.model).resolve()),
        "normalization_sha256": digest(Path(args.data) / "normalization.json"),
        "selection": selected,
        "total_reduction": "mean over samples of masked element mean",
        "component_reduction": "mean over samples with supervision of masked component mean; element sums/counts also retained",
        "noise": "N(0,1), numpy Generator per-sample seed",
        "time": "Beta(1.5,1.0)*0.999+0.001",
        "image_preprocessing": "dataset official geometry; core train=False disables augmentation",
    }
    (args.output / "protocol.json").write_text(json.dumps(metadata, indent=2))
    cached = []
    for i, row in enumerate(selected):
        sample = dataset[row["index"]]
        fixed = np.random.default_rng(args.seed + i + 1)
        sample["noise"] = fixed.standard_normal(sample["actions"].shape).astype(
            np.float32
        )
        sample["time"] = np.float32(fixed.beta(1.5, 1.0) * 0.999 + 0.001)
        sample["source"] = row["source"]
        cached.append(sample)
        if (i + 1) % 32 == 0:
            LOG.info("Cached %d/%d", i + 1, len(selected))
    temporary = args.output / "fixed_inputs.partial.pt"
    torch.save(cached, temporary)
    temporary.replace(args.output / "fixed_inputs.pt")
    metadata["fixed_inputs_sha256"] = digest(args.output / "fixed_inputs.pt")
    metadata["train_config_sha256"] = digest(
        Path(args.model) / "assets/train_config.pkl"
    )
    (args.output / "protocol.json").write_text(json.dumps(metadata, indent=2))
    LOG.info("Prepared %s", counts)


def evaluate(args):
    from openpi.models.model import Observation

    from rlinf.models.embodiment.openpi_official.config import load_train_config
    from rlinf.models.embodiment.openpi_official.trajectory import trajectory_assets

    config = load_train_config(args.model)
    core = config.model.load_pytorch(
        config, str(Path(args.model) / "model.safetensors")
    )
    core.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    original_preprocess = core._preprocess_observation

    def frozen_preprocess(self, observation, *, train=True):
        return original_preprocess(observation, train=False)

    core._preprocess_observation = MethodType(frozen_preprocess, core)
    core.requires_grad_(False).eval().to(args.device)
    cached = torch.load(args.output / "fixed_inputs.pt", weights_only=False)
    protocol = json.loads((args.output / "protocol.json").read_text())
    import inspect

    protocol["core_source_sha256"] = digest(inspect.getfile(type(core)))
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    results = []
    for step in args.steps:
        ckpt = Path(args.run) / "checkpoints" / f"global_step_{step}" / "actor"
        assets = json.loads(
            (ckpt / "upstream_assets/realized_trajectory.json").read_text()
        )
        if assets != trajectory_assets(args.data):
            raise ValueError(
                "Checkpoint recipe/normalization differs from validation dataset"
            )
        contract = json.loads((ckpt / "official_data_contract.json").read_text())
        if contract["manifest_fingerprint"] != protocol["manifest_fingerprint"]:
            raise ValueError("Checkpoint data contract differs")
        weights_path = ckpt / "model_state_dict/full_weights.pt"
        weights = torch.load(
            weights_path, map_location="cpu", mmap=True, weights_only=True
        )
        core.load_state_dict(
            {k.removeprefix("core."): v for k, v in weights.items()}, strict=True
        )
        del weights
        checkpoint_sha256 = digest(weights_path)
        stats = {}
        with torch.inference_mode():
            for start in range(0, len(cached), args.batch_size):
                items = cached[start : start + args.batch_size]
                batch = default_collate(items)

                def device_tree(x):
                    if isinstance(x, dict):
                        return {k: device_tree(v) for k, v in x.items()}
                    return x.to(args.device) if isinstance(x, torch.Tensor) else x

                batch = device_tree(batch)
                losses = core(
                    Observation.from_dict(batch["observation"]),
                    batch["actions"].float(),
                    noise=batch["noise"],
                    time=batch["time"],
                ).double()
                mask = batch["action_loss_mask"].double()
                if losses.shape != mask.shape or not torch.isfinite(losses).all():
                    raise ValueError("Invalid core losses/mask")
                for i, item in enumerate(items):
                    s = stats.setdefault(
                        item["source"],
                        {
                            "total_sum": 0.0,
                            "samples": 0,
                            "sums": np.zeros(8),
                            "counts": np.zeros(8),
                            "means": np.zeros(8),
                            "valid": np.zeros(8),
                            "arm_sum": 0.0,
                            "arm_samples": 0,
                        },
                    )
                    sums = (losses[i] * mask[i]).sum(0).cpu().numpy()
                    counts = mask[i].sum(0).cpu().numpy()
                    s["total_sum"] += float(sums.sum() / counts.sum())
                    s["samples"] += 1
                    s["sums"] += sums
                    s["counts"] += counts
                    valid = counts > 0
                    s["means"][valid] += sums[valid] / counts[valid]
                    s["valid"] += valid
                    if counts[:7].sum():
                        s["arm_sum"] += sums[:7].sum() / counts[:7].sum()
                        s["arm_samples"] += 1
                if start == 0 or (start + len(items)) % 64 == 0:
                    LOG.info(
                        "step=%d evaluated=%d/%d", step, start + len(items), len(cached)
                    )
        for source, s in stats.items():
            row = {
                "step": step,
                "source": source,
                "samples": s["samples"],
                "independent_episodes": protocol["counts"][source][
                    "independent_episodes"
                ],
                "total_loss": s["total_sum"] / s["samples"],
                "checkpoint": str(weights_path),
                "checkpoint_size": weights_path.stat().st_size,
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_mtime_ns": weights_path.stat().st_mtime_ns,
                "recipe_sha256": digest(
                    ckpt / "upstream_assets/realized_trajectory.json"
                ),
            }
            for d in range(8):
                key = f"joint{d + 1}" if d < 7 else "gripper"
                row[key + "_sum"] = float(s["sums"][d])
                row[key + "_count"] = int(s["counts"][d])
                row[key + "_loss"] = (
                    float(s["means"][d] / s["valid"][d]) if s["valid"][d] else None
                )
                row[key + "_samples"] = int(s["valid"][d])
            row["arm7_loss"] = float(s["arm_sum"] / s["arm_samples"])
            row["arm7_sum"] = float(s["arm_sum"])
            row["arm7_samples"] = s["arm_samples"]
            row["arm7_mask_elements"] = int(s["counts"][:7].sum())
            results.append(row)
        (args.output / "results.json").write_text(json.dumps(results, indent=2))
        with (args.output / "results.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(
                {k: "N/A" if v is None else v for k, v in row.items()}
                for row in results
            )
        LOG.info("Completed checkpoint %d", step)
    plot(args.output, results)


def plot(output, rows):
    import matplotlib.pyplot as plt

    sources = sorted({r["source"] for r in rows})
    fig, axes = plt.subplots(4, 2, figsize=(14, 15), constrained_layout=True)
    for ax, source in zip(axes.flat, sources):
        subset = sorted(
            (r for r in rows if r["source"] == source), key=lambda r: r["step"]
        )
        for key in ("total_loss", "arm7_loss", "gripper_loss"):
            if any(r[key] is not None for r in subset):
                ax.plot(
                    [r["step"] for r in subset],
                    [r[key] for r in subset],
                    marker="o",
                    label=key,
                )
        ax.set_title(
            f"{source}: {subset[0]['samples']} windows / {subset[0]['independent_episodes']} episodes"
        )
        if all(r["gripper_loss"] is None for r in subset):
            ax.text(0.02, 0.04, "gripper: N/A (no supervision)", transform=ax.transAxes)
        ax.set_xlabel("Training checkpoint step")
        ax.set_ylabel("Fixed validation flow-matching MSE")
        ax.grid(alpha=0.3)
        ax.legend()
    for ax in list(axes.flat)[len(sources) :]:
        ax.axis("off")
    fig.savefig(output / "per_dataset_validation_loss.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(4, 2, figsize=(14, 15), constrained_layout=True)
    for ax, source in zip(axes.flat, sources):
        subset = sorted(
            (r for r in rows if r["source"] == source), key=lambda r: r["step"]
        )
        for d in range(8):
            key = f"joint{d + 1}" if d < 7 else "gripper"
            if any(r[key + "_loss"] is not None for r in subset):
                ax.plot(
                    [r["step"] for r in subset],
                    [r[key + "_loss"] for r in subset],
                    marker=".",
                    label=key,
                )
        ax.set_title(source)
        ax.grid(alpha=0.3)
        ax.legend(ncol=4, fontsize=8)
    for ax in list(axes.flat)[len(sources) :]:
        ax.axis("off")
    fig.savefig(output / "per_joint_validation_loss.png", dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--seed", type=int, default=20261107)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--steps", type=int, nargs="+", default=[3000, 6000, 9000, 12000, 15000, 18000]
    )
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if not (args.output / "fixed_inputs.pt").exists():
        prepare(args)
    protocol = json.loads((args.output / "protocol.json").read_text())
    expected = {
        "seed": args.seed,
        "samples_per_source": args.samples,
        "data": str(Path(args.data).resolve()),
        "model": str(Path(args.model).resolve()),
    }
    if any(protocol[key] != value for key, value in expected.items()):
        raise ValueError(
            "Cached inputs use different arguments; select a new output directory"
        )
    if not args.prepare_only:
        evaluate(args)


if __name__ == "__main__":
    main()
