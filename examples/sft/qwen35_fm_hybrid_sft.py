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

"""Two-node hybrid FSDP training of Qwen3.5 FM experts with a frozen native VLM."""

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import yaml
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.utils.data import DataLoader, Dataset, Sampler

from examples.sft.qwen35_fm_data import MolmoBotSubset
from rlinf.models.embodiment.qwen35_fm.action_expert import (
    ActionExpertConfig,
    build_action_expert,
)
from rlinf.models.embodiment.qwen35_fm.policy import FlowMatchingPolicy
from rlinf.models.embodiment.qwen35_fm.vlm import Qwen35VLMWrapper
from rlinf.utils.logging import get_logger


class ImageChunks(Dataset):
    """Decode real observations in CPU workers while retaining native action masks."""

    def __init__(self, subset, mean, std, image_size, cache_file=None):
        self.subset = subset
        self.mean = mean
        self.std = std
        self.image_size = image_size
        self.cache_file = cache_file
        self.cached_images = None

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        index, valid = index
        sample = self.subset.samples[index]
        if self.cache_file is None:
            images = [
                image.resize((self.image_size, self.image_size))
                for image in self.subset.images(index)
            ]
        else:
            from PIL import Image

            if self.cached_images is None:
                self.cached_images = np.load(self.cache_file, mmap_mode="r")
            images = [Image.fromarray(image) for image in self.cached_images[index]]
        return {
            "actions": (sample["actions"] - self.mean) / self.std,
            "mask": sample["mask"] & valid,
            "images": images,
            "text": sample["episode"]["tasks"][0],
        }


class RankBatches(Sampler):
    """Shard deterministic global batches; mark validation padding as unsupervised."""

    def __init__(self, size, batch_size, rank, world_size, seed, start=0, stop=None):
        self.size, self.batch_size = size, batch_size
        self.rank, self.world_size, self.seed = rank, world_size, seed
        self.start = start
        self.batches_per_epoch = math.ceil(size / (batch_size * world_size))
        self.stop = self.batches_per_epoch if stop is None else stop
        self.training = stop is not None

    def __iter__(self):
        old_epoch, order = None, None
        global_batch = self.batch_size * self.world_size
        for step in range(self.start, self.stop):
            epoch, batch = divmod(step, self.batches_per_epoch)
            if epoch != old_epoch:
                order = (
                    torch.randperm(
                        self.size,
                        generator=torch.Generator().manual_seed(self.seed + epoch),
                    ).tolist()
                    if self.training
                    else list(range(self.size))
                )
                old_epoch = epoch
            offset = batch * global_batch + self.rank * self.batch_size
            yield [
                (order[position % self.size], self.training or position < self.size)
                for position in range(offset, offset + self.batch_size)
            ]

    def __len__(self):
        return max(0, self.stop - self.start)


def collate_chunks(samples):
    """Keep image objects on CPU and stack only action supervision."""
    return {
        "actions": torch.stack([sample["actions"] for sample in samples]),
        "mask": torch.stack([sample["mask"] for sample in samples]),
        "images": [sample["images"] for sample in samples],
        "texts": [sample["text"] for sample in samples],
    }


def prepare_native_inputs(processor, images, instructions):
    """Apply the same native chat and image processing in either CPU context."""
    texts, batch_images = [], []
    for item_images, instruction in zip(images, instructions, strict=True):
        content = [{"type": "image"} for _ in item_images] + [
            {"type": "text", "text": instruction}
        ]
        texts.append(
            processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        batch_images.extend(item_images)
    return dict(
        processor(text=texts, images=batch_images, padding=True, return_tensors="pt")
    )


class NativeProcessorCollator:
    """Prefetch native CPU inputs without sending a model or CUDA into workers."""

    def __init__(self, model_path):
        self.model_path = model_path
        self.processor = None

    def __call__(self, samples):
        if self.processor is None:
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(
                self.model_path, local_files_only=True
            )
            self.processor.tokenizer.padding_side = "left"
        batch = collate_chunks(samples)
        start = time.perf_counter()
        inputs = prepare_native_inputs(self.processor, batch["images"], batch["texts"])
        return {
            "actions": batch["actions"],
            "mask": batch["mask"],
            "inputs": inputs,
            "worker_prepare_inputs_s": time.perf_counter() - start,
        }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "model-path", "data-path", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--max-seconds", type=float, default=1200)
    parser.add_argument("--batch-size", type=int, default=16, help="Examples per rank")
    parser.add_argument("--train-episodes", type=int, default=4096)
    parser.add_argument("--val-episodes", type=int, default=64)
    parser.add_argument("--chunks-per-episode", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--cameras", nargs="+", default=["observation.images.exterior"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image-cache", help="Verified resized-image memmap cache")
    parser.add_argument(
        "--prefetch-processor",
        action="store_true",
        help="Prepare native CPU inputs in DataLoader workers",
    )
    parser.add_argument(
        "--stop-file", help="Save and stop after an update when this file exists"
    )
    parser.add_argument(
        "--precision", choices=("bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--resume", help="Existing distributed checkpoint directory")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.prefetch_processor and args.workers < 1:
        raise ValueError("--prefetch-processor requires --workers greater than zero")
    if (
        min(
            args.steps,
            args.batch_size,
            args.train_episodes,
            args.val_episodes,
            args.chunks_per_episode,
            args.checkpoint_every,
            args.log_every,
        )
        < 1
        or args.max_seconds <= 0
    ):
        raise ValueError("Training budgets, dimensions, and intervals must be positive")
    rank, world_size, local_rank = (
        int(os.environ[name]) for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    )
    if world_size != 8 or int(os.environ.get("LOCAL_WORLD_SIZE", "0")) != 4:
        raise ValueError(
            f"Expected two nodes with four ranks each; got world={world_size}, local={os.environ.get('LOCAL_WORLD_SIZE')}"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180), device_id=device)
    mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("replicate", "shard"))
    logger = get_logger()
    output = Path(args.output).resolve()

    def status(stage, **values):
        if rank == 0:
            data = {
                "stage": stage,
                "variant": config.variant,
                "world_size": world_size,
                "mesh": [2, 4],
                "global_batch_size": args.batch_size * world_size,
                **values,
            }
            temporary = output / "status.json.tmp"
            temporary.write_text(json.dumps(data, indent=2))
            temporary.replace(output / "status.json")
            logger.info(json.dumps(data))

    config_values = yaml.safe_load(Path(args.config).read_text())
    config_values = config_values.get("expert", config_values)
    state_layers = config_values.pop("state_layers", None)
    config = ActionExpertConfig(**config_values)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=args.resume is not None)
    dist.barrier()
    status("loading_data", args=vars(args))
    data_root = Path(args.data_path)
    train = MolmoBotSubset(
        data_root / "molmobot",
        config.horizon,
        args.train_episodes,
        args.chunks_per_episode,
        args.cameras,
    )
    val = MolmoBotSubset(
        data_root / "molmobot_val",
        config.horizon,
        args.val_episodes,
        args.chunks_per_episode,
        args.cameras,
    )
    cache_files = {}
    if args.image_cache:
        from examples.sft.qwen35_fm_image_cache import selection_signature

        cache_root = Path(args.image_cache).resolve()
        cache_metadata = json.loads((cache_root / "metadata.json").read_text())
        for key, expected in (
            ("schema_version", 1),
            ("complete", True),
            ("data_path", str(data_root.resolve())),
            ("cameras", args.cameras),
            ("horizon", config.horizon),
            ("image_size", args.image_size),
        ):
            if cache_metadata[key] != expected:
                raise ValueError(
                    f"Image cache {key} differs: expected {expected}, got {cache_metadata[key]}"
                )
        for split, subset in (("train", train), ("validation", val)):
            entry = cache_metadata["splits"][split]
            shape = [
                len(subset),
                len(args.cameras),
                args.image_size,
                args.image_size,
                3,
            ]
            if (
                entry["selection_sha256"] != selection_signature(subset)
                or entry["shape"] != shape
                or entry["count"] != len(subset)
            ):
                raise ValueError(
                    f"Image cache {split} selection or shape differs from the training subset"
                )
            path = cache_root / entry["filename"]
            images = np.load(path, mmap_mode="r")
            if list(images.shape) != shape or images.dtype != np.uint8:
                raise ValueError(
                    f"Expected uint8 image cache {split} with shape {shape}, got {images.dtype} {images.shape}"
                )
            cache_files[id(subset)] = path
            del images
    mean, std = train.normalization()
    if mean.numel() != config.action_dim:
        raise ValueError(
            f"Expected {config.action_dim} action coordinates, found {mean.numel()}"
        )
    train_ids = {item["provenance"]["source_episode_id"] for item in train.manifest}
    val_ids = {item["provenance"]["source_episode_id"] for item in val.manifest}
    if train_ids & val_ids:
        raise ValueError(
            "Expected disjoint native train and validation source episodes"
        )
    status("loading_model", train_chunks=len(train), validation_chunks=len(val))
    dtype = getattr(torch, args.precision)
    vlm = Qwen35VLMWrapper.from_pretrained(
        args.model_path,
        trainable=False,
        state_layers=state_layers,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    expert = build_action_expert(config, vlm.state_specs).to(
        device=device, dtype=torch.float32
    )
    parameter_count = sum(parameter.numel() for parameter in expert.parameters())
    precision = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32)
    for module in (
        list(expert.blocks)
        + list(expert.state_adapters)
        + list(getattr(expert, "gdn_blocks", []))
    ):
        fully_shard(module, mesh=mesh, mp_policy=precision, reshard_after_forward=True)
    fully_shard(expert, mesh=mesh, mp_policy=precision, reshard_after_forward=True)
    policy = FlowMatchingPolicy(vlm, expert)
    optimizer = torch.optim.AdamW(expert.parameters(), lr=args.lr, fused=True)
    manifest = {
        "args": vars(args),
        "expert_config": asdict(config),
        "expert_parameters": parameter_count,
        "model_path": str(Path(args.model_path).resolve()),
        "vlm_trainable": False,
        "state_layers": list(vlm.state_layers),
        "state_specs": vlm.state_specs,
        "mesh": {"replicate": 2, "shard": 4},
        "rank_placement": [[0, 1, 2, 3], [4, 5, 6, 7]],
        "normalization": {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "fit": "selected training chunks",
        },
        "train": train.manifest,
        "validation": val.manifest,
        "train_metadata_frames": train.info["total_frames"],
        "validation_metadata_frames": val.info["total_frames"],
        "scope": "Offline FM training with frozen native VLM; no closed-loop evaluation",
    }

    def check_replicas():
        checksum = torch.zeros(2, device=device, dtype=torch.float64)
        for parameter in expert.parameters():
            shard = parameter.detach().to_local().double()
            checksum[0] += shard.sum()
            checksum[1] += shard.square().sum()
        replicas = [torch.empty_like(checksum) for _ in range(2)]
        dist.all_gather(replicas, checksum, group=mesh["replicate"].get_group())
        if not torch.equal(replicas[0], replicas[1]):
            raise RuntimeError(
                "Expected identical parameter shards across the two hybrid replicas"
            )
        return checksum.tolist()

    start_step, previous_seconds = 0, 0.0
    if args.resume:
        metadata = json.loads((Path(args.resume) / "metadata.json").read_text())
        for field, expected in (
            ("expert_config", asdict(config)),
            ("world_size", world_size),
            ("batch_size", args.batch_size),
            ("seed", args.seed),
        ):
            if metadata[field] != expected:
                raise ValueError(
                    f"Resume {field} differs: checkpoint={metadata[field]}, requested={expected}"
                )
        saved_manifest = json.loads(Path(metadata["manifest"]).read_text())
        current_manifest = json.loads(json.dumps(manifest))
        for field in (
            "model_path",
            "vlm_trainable",
            "expert_config",
            "state_layers",
            "state_specs",
            "normalization",
            "train",
            "validation",
            "mesh",
            "rank_placement",
        ):
            if saved_manifest[field] != current_manifest[field]:
                raise ValueError(
                    f"Resume {field} differs from the saved training manifest"
                )
        for field in ("cameras", "image_size", "precision", "lr", "seed", "batch_size"):
            if saved_manifest["args"][field] != vars(args)[field]:
                raise ValueError(
                    f"Resume {field} differs from the saved training protocol"
                )
        if Path(saved_manifest["args"]["data_path"]).resolve() != data_root.resolve():
            raise ValueError(
                "Resume data root differs from the saved training protocol"
            )
        if (
            args.steps <= metadata["step"]
            or args.max_seconds <= metadata["training_seconds"]
        ):
            raise ValueError("Resume requires a remaining update and time budget")
        model_state, optim_state = get_state_dict(expert, optimizer)
        state = {"expert": model_state, "optimizer": optim_state}
        dcp.load(state, checkpoint_id=args.resume)
        set_state_dict(
            expert,
            optimizer,
            model_state_dict=state["expert"],
            optim_state_dict=state["optimizer"],
        )
        auxiliary = torch.load(
            Path(args.resume) / f"rank_{rank}.pt",
            map_location="cpu",
            weights_only=False,
        )
        torch.set_rng_state(auxiliary["torch_rng_state"])
        torch.cuda.set_rng_state(auxiliary["cuda_rng_state"], device)
        random.setstate(auxiliary["python_rng_state"])
        np.random.set_state(auxiliary["numpy_rng_state"])
        start_step, previous_seconds = metadata["step"], metadata["training_seconds"]
        manifest["resume_from"] = str(Path(args.resume).resolve())
    if rank == 0:
        (output / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
    initial_checksum = check_replicas()
    status(
        "ready",
        train_chunks=len(train),
        validation_chunks=len(val),
        expert_parameters=parameter_count,
        resumed_step=start_step,
        replica_checksum=initial_checksum,
    )

    def make_loader(subset, start=0, stop=None):
        sampler = RankBatches(
            len(subset), args.batch_size, rank, world_size, args.seed, start, stop
        )
        options = {
            "num_workers": args.workers,
            "pin_memory": True,
            "collate_fn": NativeProcessorCollator(args.model_path)
            if args.prefetch_processor
            else collate_chunks,
        }
        if args.workers:
            options.update(
                persistent_workers=True,
                prefetch_factor=2,
                multiprocessing_context="spawn",
            )
        return DataLoader(
            ImageChunks(
                subset, mean, std, args.image_size, cache_files.get(id(subset))
            ),
            batch_sampler=sampler,
            **options,
        )

    def encode(batch, timings=None):
        prepare_start = time.perf_counter()
        inputs = batch.get("inputs")
        if inputs is None:
            inputs = prepare_native_inputs(
                vlm.processor, batch["images"], batch["texts"]
            )
        inputs = {
            key: value.to(
                device=device,
                dtype=dtype if value.is_floating_point() else value.dtype,
                non_blocking=True,
            )
            for key, value in inputs.items()
        }
        if timings is not None:
            timings["prepare_inputs_s"] = time.perf_counter() - prepare_start
            timings["worker_prepare_inputs_s"] = batch.get(
                "worker_prepare_inputs_s", 0.0
            )
            timings["vlm_start"].record()
        condition = vlm.encode(
            **inputs, extract_states=config.variant == "gqa6_vlm_state"
        )
        if timings is not None:
            timings["vlm_end"].record()
        return condition

    def save_checkpoint(step, elapsed):
        destination = output / f"checkpoint_{step:06d}"
        destination.mkdir(parents=True, exist_ok=False) if rank == 0 else None
        dist.barrier()
        model_state, optim_state = get_state_dict(expert, optimizer)
        dcp.save(
            {"expert": model_state, "optimizer": optim_state}, checkpoint_id=destination
        )
        torch.save(
            {
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(device),
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
            },
            destination / f"rank_{rank}.pt",
        )
        dist.barrier()
        if rank == 0:
            (destination / "metadata.json").write_text(
                json.dumps(
                    {
                        "step": step,
                        "training_seconds": elapsed,
                        "expert_config": asdict(config),
                        "state_layers": list(vlm.state_layers),
                        "state_specs": vlm.state_specs,
                        "world_size": world_size,
                        "batch_size": args.batch_size,
                        "seed": args.seed,
                        "normalization_mean": mean.tolist(),
                        "normalization_std": std.tolist(),
                        "model_path": args.model_path,
                        "manifest": str(output / "data_manifest.json"),
                    },
                    indent=2,
                )
            )
        dist.barrier()
        return str(destination)

    policy.train()
    loader = make_loader(train, start_step, args.steps)
    training_start = time.perf_counter()
    previous_step_end = training_start
    elapsed = previous_seconds
    last_step, last_checkpoint, final_loss = start_step, None, None
    requested_stop = False
    for step, batch in enumerate(loader, start=start_step):
        step_start = previous_step_end
        log_step = step == start_step or (step + 1) % args.log_every == 0
        timings = None
        if log_step:
            timings = {"data_wait_s": time.perf_counter() - step_start}
            for event in (
                "vlm_start",
                "vlm_end",
                "expert_start",
                "expert_end",
                "optimizer_end",
            ):
                timings[event] = torch.cuda.Event(enable_timing=True)
        condition = encode(batch, timings)
        actions = batch["actions"].to(device=device, dtype=dtype, non_blocking=True)
        mask = batch["mask"].to(device=device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(args.seed + 1000 + step * world_size + rank)
        if timings is not None:
            timings["expert_start"].record()
        loss = policy.loss(actions, condition, action_mask=mask)
        valid_count = mask.sum().float()
        total_count = valid_count.clone()
        dist.all_reduce(total_count)
        weighted_loss = loss * valid_count * world_size / total_count
        finite = torch.isfinite(weighted_loss).to(torch.int32)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not bool(finite):
            raise RuntimeError(
                f"Expected finite FM loss on all ranks at update {step + 1}"
            )
        weighted_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(expert.parameters(), 1.0)
        norm = (
            gradient_norm.full_tensor()
            if hasattr(gradient_norm, "full_tensor")
            else gradient_norm
        )
        if not bool(torch.isfinite(norm)) or float(norm) == 0:
            raise RuntimeError(
                f"Expected finite nonzero global gradient norm, got {float(norm)}"
            )
        if timings is not None:
            timings["expert_end"].record()
        optimizer.step()
        if timings is not None:
            timings["optimizer_end"].record()
        global_terms = torch.stack((loss.detach() * valid_count, valid_count))
        dist.all_reduce(global_terms)
        final_loss = float(global_terms[0] / global_terms[1])
        elapsed = previous_seconds + time.perf_counter() - training_start
        last_step = step + 1
        if last_step == start_step + 1:
            check_replicas()
        if log_step:
            timings["optimizer_end"].synchronize()
            stage_names = (
                "data_wait_s",
                "prepare_inputs_s",
                "worker_prepare_inputs_s",
                "vlm_encode_s",
                "expert_forward_backward_s",
                "optimizer_s",
                "step_s",
            )
            # CUDA intervals include collective waits; they are not SM-active time.
            local_times = torch.tensor(
                [
                    timings["data_wait_s"],
                    timings["prepare_inputs_s"],
                    timings["worker_prepare_inputs_s"],
                    timings["vlm_start"].elapsed_time(timings["vlm_end"]) / 1000,
                    timings["expert_start"].elapsed_time(timings["expert_end"]) / 1000,
                    timings["expert_end"].elapsed_time(timings["optimizer_end"]) / 1000,
                    time.perf_counter() - step_start,
                ],
                device=device,
                dtype=torch.float64,
            )
            rank_times = [torch.empty_like(local_times) for _ in range(world_size)]
            dist.all_gather(rank_times, local_times)
            stage_times = torch.stack(rank_times)
            record = {
                "step": last_step,
                "fm_loss": final_loss,
                "gradient_norm": float(norm),
                "global_valid_action_scalars": int(global_terms[1]),
                "training_seconds": elapsed,
                "step_seconds": time.perf_counter() - step_start,
                "mean_chunks_per_second": (last_step - start_step)
                * args.batch_size
                * world_size
                / (elapsed - previous_seconds),
                "training_chunk_draws": last_step * args.batch_size * world_size,
                "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
            }
            if rank == 0:
                record["pipeline_max_s"] = dict(
                    zip(stage_names, stage_times.max(0).values.tolist(), strict=True)
                )
                record["pipeline_per_rank_s"] = {
                    str(index): dict(zip(stage_names, values, strict=True))
                    for index, values in enumerate(stage_times.tolist())
                }
            if rank == 0:
                with (output / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
            status("training", **record)
        requested_stop = (
            rank == 0 and args.stop_file is not None and Path(args.stop_file).exists()
        )
        stop = torch.tensor(
            2 if requested_stop else int(rank == 0 and elapsed >= args.max_seconds),
            device=device,
        )
        dist.broadcast(stop, src=0)
        requested_stop = int(stop) == 2
        final_update = last_step == args.steps or bool(stop)
        if last_step % args.checkpoint_every == 0 or final_update:
            last_checkpoint = save_checkpoint(last_step, elapsed)
        previous_step_end = time.perf_counter()
        if final_update:
            break
    training_seconds = elapsed
    validation_start = time.perf_counter()
    policy.eval()
    validation_terms = torch.zeros(2, device=device, dtype=torch.float64)
    with torch.no_grad():
        for index, batch in enumerate(make_loader(val)):
            condition = encode(batch)
            actions = batch["actions"].to(device=device, dtype=dtype, non_blocking=True)
            mask = batch["mask"].to(device=device, non_blocking=True)
            torch.manual_seed(args.seed + 100000 + index * world_size + rank)
            noise = torch.randn_like(actions)
            timestep = torch.rand(actions.shape[0], device=device)
            tau = timestep.to(dtype).reshape(-1, 1, 1)
            noisy = ((1 - tau) * noise + tau * actions).masked_fill(~mask, 0)
            velocity = policy.predict_velocity(
                noisy, timestep, condition, action_mask=mask.any(-1)
            )
            validation_terms[0] += (
                (velocity.float() - (actions - noise).float()).square() * mask
            ).sum()
            validation_terms[1] += mask.sum()
    dist.all_reduce(validation_terms)
    validation_loss = float(validation_terms[0] / validation_terms[1])
    if not math.isfinite(validation_loss):
        raise RuntimeError(f"Expected finite validation FM loss, got {validation_loss}")
    status(
        "complete",
        step=last_step,
        stop_reason=(
            "requested_stop"
            if requested_stop
            else "step_budget"
            if last_step == args.steps
            else "time_budget"
        ),
        training_chunk_draws=last_step * args.batch_size * world_size,
        train_chunks=len(train),
        validation_chunks=len(val),
        validation_action_scalars=int(validation_terms[1]),
        validation_fm_loss=validation_loss,
        fm_loss=final_loss,
        checkpoint=last_checkpoint,
        training_seconds=training_seconds,
        validation_seconds=time.perf_counter() - validation_start,
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
