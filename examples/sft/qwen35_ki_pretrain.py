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

"""Hybrid FSDP knowledge-insulated robot FM/FAST CE and native visual-language CE."""

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
from torch.distributed.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
    register_fsdp_forward_method,
)
from torch.utils.data import DataLoader

from examples.sft.qwen35_fm_data import MolmoBotSubset
from examples.sft.qwen35_fm_hybrid_sft import RankBatches
from examples.sft.qwen35_fm_image_cache import selection_signature
from rlinf.models.embodiment.qwen35_fm.action_expert import (
    ActionExpertConfig,
    build_action_expert,
)
from rlinf.models.embodiment.qwen35_fm.fast_tokenizer import FASTActionTokenizer
from rlinf.models.embodiment.qwen35_fm.ki_policy import KIFlowMatchingPolicy
from rlinf.models.embodiment.qwen35_fm.vlm import Qwen35VLMWrapper
from rlinf.utils.logging import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "config",
        "model-path",
        "fast-path",
        "data-path",
        "vl-data-path",
        "output",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--vl-format", default="robo2vlm")
    parser.add_argument(
        "--vl-image-cache", help="Verified visual-language metadata and image memmaps"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Examples per rank")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--max-seconds", type=float, default=1200)
    parser.add_argument("--train-episodes", type=int, default=4096)
    parser.add_argument("--val-episodes", type=int, default=64)
    parser.add_argument("--chunks-per-episode", type=int, default=8)
    parser.add_argument("--image-cache", help="Verified robot image cache directory")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--cameras", nargs="+", default=["observation.images.exterior"])
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--resume")
    parser.add_argument(
        "--reshard-after-forward", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vlm-lr", type=float, default=2.5e-5)
    parser.add_argument("--expert-lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument(
        "--schedule-steps", type=int, help="Fixed cosine horizon, inherited on resume"
    )
    parser.add_argument("--vl-every", type=int, default=4)
    parser.add_argument("--vl-limit", type=int)
    parser.add_argument("--vl-val-limit", type=int)
    return parser.parse_args()


def quantize_states(states, bounds):
    low, high = (torch.tensor(bounds[key]) for key in ("q01", "q99"))
    return (
        (((states.float() - low) / (high - low).clamp_min(1e-6)).clamp(0, 1) * 255)
        .round()
        .long()
    )


def learning_rate_factor(step, warmup, horizon):
    if warmup and step < warmup:
        return (step + 1) / warmup
    progress = min(max((step + 1 - warmup) / max(horizon - warmup, 1), 0.0), 1.0)
    return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2


def main():
    from examples.sft.qwen35_ki_data import (
        KIRobotCollator,
        KIRobotDataset,
        VLCollator,
        build_vl_dataset,
    )

    args = parse_args()
    if (
        min(
            args.batch_size,
            args.steps,
            args.train_episodes,
            args.val_episodes,
            args.chunks_per_episode,
            args.checkpoint_every,
            args.log_every,
        )
        < 1
    ):
        raise ValueError("Training sizes and intervals must be positive")
    if args.vl_every < 2 or args.warmup_steps < 0 or args.max_seconds <= 0:
        raise ValueError(
            "Expected vl_every>=2, warmup_steps>=0, and positive time budget"
        )
    rank, world, local_rank = (
        int(os.environ[key]) for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    )
    if world != 8 or int(os.environ.get("LOCAL_WORLD_SIZE", "0")) != 4:
        raise ValueError("KI hybrid training requires two nodes with four ranks each")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180), device_id=device)
    mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("replicate", "shard"))
    output = Path(args.output).resolve()
    resume_metadata = (
        json.loads((Path(args.resume) / "metadata.json").read_text())
        if args.resume
        else None
    )
    target_path = (
        Path(resume_metadata["manifest"]).parent if resume_metadata else output
    ) / "fast_targets.json"
    if rank == 0:
        output.mkdir(parents=True, exist_ok=args.resume is not None)
    dist.barrier()
    logger = get_logger()
    values = yaml.safe_load(Path(args.config).read_text())
    values = values.get("expert", values)
    state_layers = values.pop("state_layers", None)
    config = ActionExpertConfig(**values)

    def status(stage, **fields):
        if rank == 0:
            record = {
                "stage": stage,
                "variant": config.variant,
                "world_size": world,
                "mesh": [2, 4],
                "global_batch_size": args.batch_size * world,
                **fields,
            }
            temporary = output / "status.json.tmp"
            temporary.write_text(json.dumps(record, indent=2))
            temporary.replace(output / "status.json")
            logger.info(json.dumps(record))

    status("loading_data", args=vars(args))
    root = Path(args.data_path).resolve()
    train = MolmoBotSubset(
        root / "molmobot",
        config.horizon,
        args.train_episodes,
        args.chunks_per_episode,
        args.cameras,
        include_state=True,
    )
    val = MolmoBotSubset(
        root / "molmobot_val",
        config.horizon,
        args.val_episodes,
        args.chunks_per_episode,
        args.cameras,
        include_state=True,
    )
    if {item["provenance"]["source_episode_id"] for item in train.manifest} & {
        item["provenance"]["source_episode_id"] for item in val.manifest
    }:
        raise ValueError(
            "Robot train and validation must have disjoint source episodes"
        )
    mean, std = train.normalization()
    if len(mean) != config.action_dim:
        raise ValueError(f"Expected {config.action_dim} actions, found {len(mean)}")
    raw_actions = torch.stack([sample["actions"] for sample in train.samples])
    masks = torch.stack([sample["mask"] for sample in train.samples])
    action_quantiles = FASTActionTokenizer.fit_quantiles(raw_actions, masks)
    state_quantiles = FASTActionTokenizer.fit_quantiles(train.states[:, None])
    cache_paths = {}
    if args.image_cache:
        cache_root = Path(args.image_cache).resolve()
        cache = json.loads((cache_root / "metadata.json").read_text())
        for field, expected in (
            ("complete", True),
            ("data_path", str(root)),
            ("cameras", args.cameras),
            ("image_size", args.image_size),
            ("horizon", config.horizon),
            ("schema_version", 1),
        ):
            if cache[field] != expected:
                raise ValueError(
                    f"Robot image cache {field} differs from the KI protocol"
                )
        for split, subset in (("train", train), ("validation", val)):
            entry = cache["splits"][split]
            expected_shape = [
                len(subset),
                len(args.cameras),
                args.image_size,
                args.image_size,
                3,
            ]
            if (
                entry["selection_sha256"] != selection_signature(subset)
                or entry["shape"] != expected_shape
            ):
                raise ValueError(
                    f"Robot image cache {split} selection differs from KI data"
                )
            cache_paths[split] = cache_root / entry["filename"]
    vl_val = build_vl_dataset(
        args.vl_data_path,
        "validation",
        format=args.vl_format,
        limit=args.vl_val_limit,
        image_size=args.image_size,
        cache_dir=args.vl_image_cache,
        seed=args.seed,
    )
    vl_train = build_vl_dataset(
        args.vl_data_path,
        "train",
        format=args.vl_format,
        limit=args.vl_limit,
        image_size=args.image_size,
        cache_dir=args.vl_image_cache,
        seed=args.seed,
        excluded_image_hashes=vl_val.image_hashes,
    )
    status(
        "loading_model",
        robot_train_chunks=len(train),
        robot_validation_chunks=len(val),
        vl_train_examples=len(vl_train),
        vl_validation_examples=len(vl_val),
    )
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    vlm = Qwen35VLMWrapper.from_pretrained(
        args.model_path,
        trainable=True,
        state_layers=state_layers,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
    ).to(device)
    fast = FASTActionTokenizer.from_pretrained(args.fast_path, action_quantiles)
    fast.attach(vlm.model, vlm.processor.tokenizer)
    vlm.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if rank == 0 and args.resume is None:
        sequences = {
            "train": fast.encode_qwen(raw_actions, masks),
            "validation": fast.encode_qwen(
                torch.stack([sample["actions"] for sample in val.samples]),
                torch.stack([sample["mask"] for sample in val.samples]),
            ),
        }
        target_path.write_text(json.dumps(sequences))
    dist.barrier()
    sequences = json.loads(target_path.read_text())
    robot_train = KIRobotDataset(
        train,
        mean,
        std,
        sequences["train"],
        quantize_states(train.states, state_quantiles),
        cache_paths.get("train"),
        args.image_size,
    )
    robot_val = KIRobotDataset(
        val,
        mean,
        std,
        sequences["validation"],
        quantize_states(val.states, state_quantiles),
        cache_paths.get("validation"),
        args.image_size,
    )
    torch.manual_seed(args.seed)
    expert = build_action_expert(config, vlm.state_specs).to(
        device=device, dtype=torch.float32
    )
    policy = KIFlowMatchingPolicy(vlm, expert, fast)
    native_count = sum(parameter.numel() for parameter in vlm.parameters())
    expert_count = sum(parameter.numel() for parameter in expert.parameters())
    precision = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    for layer in vlm.model.model.language_model.layers:
        fully_shard(
            layer,
            mesh=mesh,
            mp_policy=precision,
            reshard_after_forward=args.reshard_after_forward,
        )
    for block in vlm.model.model.visual.blocks:
        fully_shard(
            block,
            mesh=mesh,
            mp_policy=precision,
            reshard_after_forward=args.reshard_after_forward,
        )
    for block in (
        list(expert.blocks)
        + list(expert.state_adapters)
        + list(getattr(expert, "gdn_blocks", []))
    ):
        fully_shard(
            block,
            mesh=mesh,
            mp_policy=precision,
            reshard_after_forward=args.reshard_after_forward,
        )
    fully_shard(
        expert,
        mesh=mesh,
        mp_policy=precision,
        reshard_after_forward=args.reshard_after_forward,
    )
    fully_shard(
        policy,
        mesh=mesh,
        mp_policy=precision,
        reshard_after_forward=args.reshard_after_forward,
    )
    register_fsdp_forward_method(policy, "vl_loss")
    vlm_parameters = list(policy.vlm.parameters())
    expert_parameters = list(policy.expert.parameters())
    optimizers = [
        torch.optim.AdamW(
            parameters,
            lr=lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=1e-10,
            fused=True,
        )
        for parameters, lr in (
            (vlm_parameters, args.vlm_lr),
            (expert_parameters, args.expert_lr),
        )
    ]
    protocol = {
        "expert_config": asdict(config),
        "model_path": str(Path(args.model_path).resolve()),
        "fast_path": str(Path(args.fast_path).resolve()),
        "fast_source": fast.source,
        "fast_bridge": fast.bridge,
        "data_path": str(root),
        "cameras": args.cameras,
        "image_size": args.image_size,
        "robot_train": train.manifest,
        "robot_validation": val.manifest,
        "vl_data_path": str(Path(args.vl_data_path).resolve()),
        "vl_format": args.vl_format,
        "vl_source_info": {"train": vl_train.info, "validation": vl_val.info},
        "vl_train": vl_train.manifest,
        "vl_validation": vl_val.manifest,
        "fm_normalization": {"mean": mean.tolist(), "std": std.tolist()},
        "fast_normalization": action_quantiles,
        "state_quantiles": state_quantiles,
        "state_layers": list(vlm.state_layers),
        "state_specs": vlm.state_specs,
        "batch_size_per_rank": args.batch_size,
        "world_size": world,
        "mesh": [2, 4],
        "seed": args.seed,
        "vl_every": args.vl_every,
        "warmup_steps": args.warmup_steps,
        "vlm_lr": args.vlm_lr,
        "expert_lr": args.expert_lr,
        "precision": "bfloat16",
        "schedule_steps": args.schedule_steps or args.steps,
        "gradient_paths": {
            "FM": "expert only; detached prefix hidden states and GDN memories",
            "FAST_CE": "all native VLM parameters",
            "VL_CE": "all native VLM parameters",
        },
    }
    start_step, previous_seconds = 0, 0.0
    if args.resume:
        metadata = resume_metadata
        if args.schedule_steps is None:
            protocol["schedule_steps"] = metadata["protocol"]["schedule_steps"]
        if json.loads(json.dumps(protocol)) != metadata["protocol"]:
            raise ValueError(
                "KI resume protocol differs from checkpoint; data/model/normalization/schedule must match"
            )
        if (
            args.steps <= metadata["step"]
            or args.max_seconds <= metadata["training_seconds"]
        ):
            raise ValueError("KI resume requires a remaining update and time budget")
        model_state, optim_state = get_state_dict(policy, optimizers)
        state = {"model": model_state, "optimizers": optim_state}
        dcp.load(state, checkpoint_id=args.resume)
        set_state_dict(
            policy,
            optimizers,
            model_state_dict=state["model"],
            optim_state_dict=state["optimizers"],
        )
        rng = torch.load(
            Path(args.resume) / f"rank_{rank}.pt",
            map_location="cpu",
            weights_only=False,
        )
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state(rng["cuda"], device)
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        start_step, previous_seconds = metadata["step"], metadata["training_seconds"]
    if rank == 0:
        if args.resume:
            (output / "fast_targets.json").write_text(json.dumps(sequences))
        fast.save_metadata(output / "fast_tokenizer.json")
        (output / "data_manifest.json").write_text(
            json.dumps(
                {
                    "args": vars(args),
                    "protocol": protocol,
                    "native_parameters": native_count,
                    "expert_parameters": expert_count,
                    "scope": "Knowledge-insulated robot FM/FAST CE with authentic VL CE; no closed-loop success",
                },
                indent=2,
            )
        )

    def check_replicas():
        checksum = torch.zeros(2, device=device, dtype=torch.float64)
        for parameter in policy.parameters():
            local = parameter.detach().to_local().double()
            checksum[0] += local.sum()
            checksum[1] += local.square().sum()
        gathered = [torch.empty_like(checksum) for _ in range(2)]
        dist.all_gather(gathered, checksum, group=mesh["replicate"].get_group())
        if not torch.equal(gathered[0], gathered[1]):
            raise RuntimeError("KI parameter shards differ across hybrid replicas")
        return checksum.tolist()

    def loader(dataset, collator, domain_start=0, domain_stop=None, batch_size=None):
        batch_size = args.batch_size if batch_size is None else batch_size
        sampler = RankBatches(
            len(dataset), batch_size, rank, world, args.seed, domain_start, domain_stop
        )
        options = {
            "num_workers": args.workers,
            "pin_memory": True,
            "collate_fn": collator,
        }
        if args.workers:
            options.update(
                persistent_workers=True,
                prefetch_factor=2,
                multiprocessing_context="spawn",
            )
        return DataLoader(dataset, batch_sampler=sampler, **options)

    robot_collator = KIRobotCollator(
        args.model_path, processor=vlm.processor if args.workers == 0 else None
    )
    vl_collator = VLCollator(
        args.model_path, processor=vlm.processor if args.workers == 0 else None
    )
    prior_vl_steps = start_step // args.vl_every
    robot_iterator = iter(
        loader(robot_train, robot_collator, start_step - prior_vl_steps, args.steps)
    )
    vl_iterator = iter(loader(vl_train, vl_collator, prior_vl_steps, args.steps))

    def to_device(value):
        if isinstance(value, torch.Tensor):
            return value.to(device=device, non_blocking=True)
        return {key: to_device(item) for key, item in value.items()}

    def global_mean(loss, count):
        count = count.double()
        totals = torch.stack((loss.detach().double() * count, count))
        dist.all_reduce(totals)
        weighted = (
            loss * count.to(loss.dtype) * world / totals[1].clamp_min(1).to(loss.dtype)
        )
        return weighted, float(totals[0] / totals[1].clamp_min(1)), int(totals[1])

    def clip(parameters):
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        norm = norm.full_tensor() if hasattr(norm, "full_tensor") else norm
        if not bool(torch.isfinite(norm)):
            raise RuntimeError("Expected finite KI gradient norms on every rank")
        return float(norm)

    def save(step, elapsed):
        path = output / f"checkpoint_{step:06d}"
        if rank == 0:
            path.mkdir(exist_ok=False)
        dist.barrier()
        model_state, optim_state = get_state_dict(policy, optimizers)
        dcp.save({"model": model_state, "optimizers": optim_state}, checkpoint_id=path)
        torch.save(
            {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(device),
                "python": random.getstate(),
                "numpy": np.random.get_state(),
            },
            path / f"rank_{rank}.pt",
        )
        dist.barrier()
        if rank == 0:
            fast.save_metadata(path / "fast_tokenizer.json")
            (path / "metadata.json").write_text(
                json.dumps(
                    {
                        "step": step,
                        "training_seconds": elapsed,
                        "protocol": protocol,
                        "manifest": str(output / "data_manifest.json"),
                    },
                    indent=2,
                )
            )
        dist.barrier()
        return str(path)

    status(
        "ready",
        resumed_step=start_step,
        native_parameters=native_count,
        expert_parameters=expert_count,
        replica_checksum=check_replicas(),
        robot_train_chunks=len(train),
        vl_train_examples=len(vl_train),
    )
    policy.train()
    started = time.perf_counter()
    last_end = started
    last_step, checkpoint, elapsed = start_step, None, previous_seconds
    for step in range(start_step, args.steps):
        is_vl = (step + 1) % args.vl_every == 0
        batch = to_device(next(vl_iterator if is_vl else robot_iterator))
        data_ready = time.perf_counter()
        factor = learning_rate_factor(
            step, args.warmup_steps, protocol["schedule_steps"]
        )
        for optimizer, peak in zip(
            optimizers, (args.vlm_lr, args.expert_lr), strict=True
        ):
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group["lr"] = peak * factor
        torch.manual_seed(args.seed + 1000 + step * world + rank)
        if is_vl:
            local = policy.vl_loss(batch["vlm_inputs"])
            loss, vl_loss, vl_tokens = global_mean(
                local, (batch["vlm_inputs"]["labels"][:, 1:] != -100).sum()
            )
            fm_loss = action_ce = 0.0
            fm_count = action_tokens = 0
        else:
            losses = policy(**batch)
            fm, fm_loss, fm_count = global_mean(
                losses["fm_loss"], losses["fm_valid_action_scalars"]
            )
            ce, action_ce, action_tokens = global_mean(
                losses["action_ce_loss"], losses["action_ce_tokens"]
            )
            loss = fm + ce
            vl_loss, vl_tokens = 0.0, 0
        finite = torch.isfinite(loss).int()
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not bool(finite):
            raise RuntimeError(f"Nonfinite KI loss at update {step + 1}")
        loss.backward()
        vlm_norm = clip(vlm_parameters)
        expert_norm = clip(expert_parameters) if not is_vl else 0.0
        if (vl_tokens if is_vl else action_tokens) > 0:
            optimizers[0].step()
        if not is_vl:
            optimizers[1].step()
        torch.cuda.synchronize(device)
        last_step = step + 1
        elapsed = previous_seconds + time.perf_counter() - started
        if last_step == start_step + 1:
            check_replicas()
        record = {
            "step": last_step,
            "domain": "vl" if is_vl else "robot",
            "fm_loss": fm_loss,
            "action_ce_loss": action_ce,
            "vl_ce_loss": vl_loss,
            "fm_valid_action_scalars": fm_count,
            "action_ce_tokens": action_tokens,
            "vl_tokens": vl_tokens,
            "vlm_gradient_norm": vlm_norm,
            "expert_gradient_norm": expert_norm,
            "vlm_lr": args.vlm_lr * factor,
            "expert_lr": args.expert_lr * factor,
            "data_wait_seconds": data_ready - last_end,
            "step_seconds": time.perf_counter() - last_end,
            "training_seconds": elapsed,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device),
            "robot_updates": last_step - last_step // args.vl_every,
            "vl_updates": last_step // args.vl_every,
        }
        if rank == 0 and (
            last_step % args.log_every == 0 or last_step == start_step + 1
        ):
            with (output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            status("training", **record)
        stop = torch.tensor(
            int(
                rank == 0
                and (
                    elapsed >= args.max_seconds
                    or (output / "STOP_AFTER_UPDATE").exists()
                )
            ),
            device=device,
        )
        dist.broadcast(stop, 0)
        final = bool(stop) or last_step == args.steps
        if last_step % args.checkpoint_every == 0 or final:
            checkpoint = save(last_step, elapsed)
        last_end = time.perf_counter()
        if final:
            break
    training_seconds = elapsed
    policy.eval()
    status(
        "validating",
        step=last_step,
        checkpoint=checkpoint,
        training_seconds=training_seconds,
    )
    if args.skip_validation:
        status(
            "complete",
            step=last_step,
            checkpoint=checkpoint,
            training_seconds=training_seconds,
            validation_skipped=True,
            robot_updates=last_step - last_step // args.vl_every,
            vl_updates=last_step // args.vl_every,
        )
        dist.barrier()
        dist.destroy_process_group()
        return
    validation_start = time.perf_counter()
    robot_terms = torch.zeros(4, device=device, dtype=torch.float64)
    vl_terms = torch.zeros(2, device=device, dtype=torch.float64)
    with torch.no_grad():
        for index, raw_batch in enumerate(
            loader(robot_val, robot_collator, batch_size=1)
        ):
            batch = to_device(raw_batch)
            torch.manual_seed(args.seed + 100000 + index * world + rank)
            # Padded ranks still execute all FSDP collectives; they contribute no terms.
            mask = batch["action_mask"]
            original_count = mask.sum()
            if not bool(original_count):
                batch["action_mask"] = torch.ones_like(mask)
            losses = policy(**batch)
            robot_terms[0] += losses["fm_loss"].double() * original_count
            robot_terms[1] += original_count
            robot_terms[2] += (
                losses["action_ce_loss"].double() * losses["action_ce_tokens"]
            )
            robot_terms[3] += losses["action_ce_tokens"]
        for index, raw_batch in enumerate(loader(vl_val, vl_collator, batch_size=1)):
            batch = to_device(raw_batch)
            local = policy.vl_loss(batch["vlm_inputs"])
            count = (batch["vlm_inputs"]["labels"][:, 1:] != -100).sum()
            vl_terms += torch.stack((local.double() * count, count.double()))
    dist.all_reduce(robot_terms)
    dist.all_reduce(vl_terms)
    status(
        "complete",
        step=last_step,
        stop_reason="step_budget" if last_step == args.steps else "time_or_stop_file",
        checkpoint=checkpoint,
        training_seconds=training_seconds,
        validation_seconds=time.perf_counter() - validation_start,
        validation_fm_loss=float(robot_terms[0] / robot_terms[1].clamp_min(1)),
        validation_action_ce_loss=float(robot_terms[2] / robot_terms[3].clamp_min(1)),
        validation_vl_ce_loss=float(vl_terms[0] / vl_terms[1].clamp_min(1)),
        validation_fm_scalars=int(robot_terms[1]),
        validation_action_tokens=int(robot_terms[3]),
        validation_vl_tokens=int(vl_terms[1]),
        robot_updates=last_step - last_step // args.vl_every,
        vl_updates=last_step // args.vl_every,
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
