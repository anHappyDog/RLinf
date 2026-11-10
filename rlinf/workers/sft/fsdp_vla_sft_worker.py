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
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.config import SupportedModel
from rlinf.data.datasets.common.epoch import forward_set_epoch
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.utils import get_rng_state, seed_everything, set_rng_state
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker


class _ProfiledDataIterator:
    def __init__(self, iterator):
        self.iterator = iterator

    def __next__(self):
        with torch.profiler.record_function("sft/data_wait"):
            return next(self.iterator)


class FSDPVlaSftWorker(FSDPSftWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._mixture_exposure = {"schemas": {}, "groups": {}}
        self._world_size_migrations = []
        self._short_profiler = None
        self._short_profile_steps = 0

    def _dataset_loss_components(self):
        if self.cfg.data.official.get("recipe") in (
            "realized_trajectory",
            "canonical_qtarget",
        ):
            return (
                ("", slice(None)),
                ("/joint7", slice(0, 7)),
                ("/gripper1", slice(7, 8)),
            )
        return (("", slice(None)),)

    def _loss_statistics_options(self):
        return {
            "return_loss_statistics": True,
            "loss_statistic_slices": tuple(
                part for _, part in self._dataset_loss_components()
            ),
        }

    def _accumulate_dataset_loss(self, totals, batch, statistics, schemas):
        """Accumulate per-sample means and valid sample counts without host transfers."""
        membership = torch.tensor(
            [
                [source == schema for source in batch["sample_schema"]]
                for schema in schemas
            ],
            device=statistics.device,
            dtype=statistics.dtype,
        )
        totals.add_(torch.einsum("sb,bck->sck", membership, statistics))

    def _reduce_dataset_loss(self, totals, schemas):
        """Pool numerators and denominators across ranks before division."""
        torch.distributed.all_reduce(totals)
        values = totals.cpu().tolist()
        metrics = {}
        for schema, components in zip(schemas, values, strict=True):
            for (component, _), (numerator, denominator, elements) in zip(
                self._dataset_loss_components(), components, strict=True
            ):
                if denominator:
                    tag = f"loss_by_dataset/{schema}{component}"
                    metrics[tag] = numerator / denominator
                    metrics[f"{tag}/samples"] = denominator
                    metrics[f"{tag}/mask_elements"] = elements
        return metrics

    def run_training(self):
        official = self.cfg.data.get("type") == "official_mixture"
        if not official:
            return self._run_training()
        schemas = self.data_loader.dataset.schemas
        self._dataset_loss_totals = torch.zeros(
            (len(schemas), len(self._dataset_loss_components()), 3),
            device=self.device,
            dtype=torch.float64,
        )
        try:
            metrics = self._run_training()
            metrics.update(
                self._reduce_dataset_loss(self._dataset_loss_totals, schemas)
            )
            return metrics
        finally:
            del self._dataset_loss_totals

    def _run_training(self):
        profile_dir = self.cfg.runner.get("short_profile_dir")
        if not profile_dir or self._rank != 0 or self._short_profile_steps >= 3:
            return super().run_training()
        if self._short_profiler is None:
            output = Path(profile_dir)
            output.mkdir(parents=True, exist_ok=True)
            self._short_profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=0, warmup=1, active=2, repeat=1),
                on_trace_ready=lambda profiler: profiler.export_chrome_trace(
                    str(output / "rank0.trace.json")
                ),
                record_shapes=False,
                with_stack=False,
                profile_memory=False,
            )
            self._short_profiler.start()
        iterator = _ProfiledDataIterator(self.data_iter)
        self.data_iter = iterator
        try:
            with torch.profiler.record_function("sft/update"):
                metrics = super().run_training()
        finally:
            if self.data_iter is iterator:
                self.data_iter = iterator.iterator
        self._short_profiler.step()
        self._short_profile_steps += 1
        if self._short_profile_steps == 3:
            self._short_profiler.stop()
        return metrics

    def init_worker(self):
        # SFT forwards draw augmentation, noise and time from the global RNGs;
        # seed them per rank so runs follow actor.seed.
        seed_everything(self.cfg.actor.seed + self._rank)
        super().init_worker()

    def build_dataloader(self, data_paths: Any, eval_dataset: bool = False):
        model_type = SupportedModel(self.cfg.actor.model.model_type)
        if model_type == SupportedModel.OPENPI_OFFICIAL or (
            model_type == SupportedModel.OPENPI
            and self.cfg.data.get("type") == "official_mixture"
        ):
            from rlinf.data.datasets.openpi_official import (
                build_official_mixture_sft_dataloader,
            )

            return build_official_mixture_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        if model_type == SupportedModel.OPENPI:
            from rlinf.data.datasets.openpi import (
                build_openpi_sft_dataloader,
            )

            return build_openpi_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        elif model_type == SupportedModel.LINGBOTVLA:
            from rlinf.models.embodiment.lingbotvla.sft_builder import (
                build_lingbot_sft_dataloader,
            )

            return build_lingbot_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths
            )
        elif model_type == SupportedModel.DREAMZERO:
            from rlinf.data.datasets.dreamzero import (
                build_dreamzero_sft_dataloader,
            )

            return build_dreamzero_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        elif model_type == SupportedModel.COSMOS3:
            from rlinf.data.datasets.cosmos3 import (
                build_cosmos3_sft_dataloader,
            )

            return build_cosmos3_sft_dataloader(self.cfg, data_paths, eval_dataset)
        elif model_type == SupportedModel.EVO1:
            from rlinf.models.embodiment.evo1.sft_builder import (
                build_evo1_sft_dataloader,
            )

            return build_evo1_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths
            )
        elif model_type == SupportedModel.FASTWAM:
            from rlinf.data.datasets.fastwam import build_fastwam_sft_dataloader

            return build_fastwam_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def get_eval_model_output(self, batch: dict[str, Any]):
        # now the eval is not supported for embodied sft
        raise NotImplementedError("eval is not supported for embodied sft right now.")

    @torch.no_grad()
    def run_eval(self):
        """Evaluate mixture flow loss with repeatable RNG and restored training state."""
        official = self.cfg.data.get("type") == "official_mixture"
        if self.cfg.data.get("type") not in ("mixture", "official_mixture"):
            return super().run_eval()

        with self.worker_timer():
            rng_state = get_rng_state()
            was_training = self.model.training
            self.model.eval()
            totals = torch.zeros(2, device=self.device, dtype=torch.float64)
            if official:
                schemas = self.eval_data_loader.dataset.schemas
                dataset_totals = torch.zeros(
                    (len(schemas), len(self._dataset_loss_components()), 3),
                    device=self.device,
                    dtype=torch.float64,
                )
            try:
                forward_set_epoch(self.eval_data_loader, 0)
                seed_everything(self.cfg.actor.seed + self._rank)
                for batch in self.eval_data_loader:
                    with self.amp_context:
                        options = (
                            self._loss_statistics_options()
                            if official
                            else {"train": False}
                        )
                        output = self.model(
                            forward_type=ForwardType.SFT, data=batch, **options
                        )
                        loss = (
                            output
                            if isinstance(output, torch.Tensor)
                            else output["loss"]
                        )
                    if official:
                        self._accumulate_dataset_loss(
                            dataset_totals, batch, output["loss_statistics"], schemas
                        )
                    count = batch["actions"].shape[0]
                    totals[0] += loss.detach().double() * count
                    totals[1] += count
            finally:
                set_rng_state(rng_state)
                self.model.train(was_training)
            torch.distributed.all_reduce(totals)
            metric = "augmented_flow_loss" if official else "loss"
            metrics = {metric: (totals[0] / totals[1]).item()}
            if official:
                metrics.update(self._reduce_dataset_loss(dataset_totals, schemas))
            return metrics

    def get_train_model_output(self, batch: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        with self.amp_context:
            options = (
                self._loss_statistics_options()
                if self.cfg.data.get("type") == "official_mixture"
                else {}
            )
            output = self.model(forward_type=ForwardType.SFT, data=batch, **options)
        if "return_loss_statistics" in options:
            self._accumulate_dataset_loss(
                self._dataset_loss_totals,
                batch,
                output["loss_statistics"],
                self.data_loader.dataset.schemas,
            )

        if isinstance(output, torch.Tensor):
            loss = output
        else:
            loss = output["loss"]

        step_metrics = {"loss": loss.detach().item()}
        if isinstance(batch, dict) and "sample_schema" in batch:
            for schema, group in zip(
                batch["sample_schema"], batch["sample_group"], strict=True
            ):
                counts = self._mixture_exposure["schemas"]
                counts[schema] = counts.get(schema, 0) + 1
                groups = self._mixture_exposure["groups"].setdefault(schema, {})
                groups[group] = groups.get(group, 0) + 1
            for schema in self.data_loader.dataset.schemas:
                step_metrics[f"samples/{schema}"] = batch["sample_schema"].count(schema)
        if isinstance(output, dict):
            for key, value in output.items():
                if key == "loss":
                    continue
                if torch.is_tensor(value):
                    if value.numel() == 1:
                        step_metrics[key] = value.detach().item()
                elif isinstance(value, (float, int)):
                    step_metrics[key] = value
        return loss, step_metrics

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        super().save_checkpoint(save_path, step)

        if self.cfg.data.get("type") == "official_mixture":
            exposures = [None] * self._world_size
            torch.distributed.all_gather_object(exposures, self._mixture_exposure)
            if self._rank == 0:
                root = Path(save_path)
                (root / "official_data_contract.json").write_text(
                    json.dumps(
                        {
                            "manifest_fingerprint": self.data_loader.dataset.manifest_fingerprint
                        },
                        indent=2,
                    )
                    + "\n"
                )
                (root / "official_exposure.json").write_text(
                    json.dumps(exposures, indent=2) + "\n"
                )
                if self._world_size_migrations:
                    (root / "world_size_migrations.json").write_text(
                        json.dumps(self._world_size_migrations, indent=2) + "\n"
                    )
                model_path = Path(self.cfg.actor.model.model_path)
                source_assets = next(
                    model_path / directory
                    for directory in (
                        "assets",
                        "upstream_assets",
                        "actor/upstream_assets",
                    )
                    if (model_path / directory).is_dir()
                )
                shutil.copytree(
                    source_assets,
                    root / "upstream_assets",
                    dirs_exist_ok=True,
                )
                if self.cfg.data.official.get("recipe") == "canonical_qtarget":
                    from rlinf.data.datasets.openpi_official.qtarget import (
                        save_qtarget_assets,
                    )

                    save_qtarget_assets(
                        self.data_loader.dataset.root, root / "upstream_assets"
                    )
                if self.cfg.data.official.get("recipe") == "realized_trajectory":
                    from rlinf.models.embodiment.openpi_official.trajectory import (
                        save_trajectory_assets,
                    )

                    save_trajectory_assets(
                        self.data_loader.dataset.root, root / "upstream_assets"
                    )

        if self.cfg.data.get("type") == "mixture":
            exposures = [None] * self._world_size
            torch.distributed.all_gather_object(exposures, self._mixture_exposure)
            if self._rank == 0:
                root = self.data_loader.dataset.root
                assets = Path(save_path) / "mixture_assets"
                assets.mkdir(exist_ok=True)
                for name in ("normalization.json", "summary.json", "contract.json"):
                    source = root / name
                    if source.exists():
                        shutil.copy2(source, assets / name)
                (assets / "exposure.json").write_text(
                    json.dumps(exposures, indent=2) + "\n"
                )

        if isinstance(self.data_loader, StatefulDataLoader):
            state = self.data_loader.state_dict()

            all_states = [None] * self._world_size
            torch.distributed.all_gather_object(all_states, state)

            if self._rank == 0:
                torch.save(all_states, os.path.join(save_path, "data.pt"))

            torch.distributed.barrier()

            rng_state = get_rng_state()
            all_rng_states = [None] * self._world_size
            torch.distributed.all_gather_object(all_rng_states, rng_state)
            if self._rank == 0:
                torch.save(all_rng_states, os.path.join(save_path, "rng.pt"))

            torch.distributed.barrier()

    def _migrate_realized_data(self, load_path: str, exposures: list[dict]) -> dict:
        from rlinf.data.datasets.openpi.realized_trajectory import (
            RealizedTrajectorySampler,
        )

        sampler = self.data_loader.sampler
        if not isinstance(sampler, RealizedTrajectorySampler):
            raise ValueError(
                "World-size migration requires the realized trajectory sampler"
            )
        all_states = torch.load(Path(load_path) / "data.pt", weights_only=False)
        if len(all_states) != len(exposures):
            raise ValueError("Checkpoint reader and exposure rank counts differ")
        states = []
        for reader in all_states:
            snapshot = reader["_snapshot"]["_main_snapshot"]
            state = dict(snapshot["_sampler_iter_state"]["sampler_state"])
            # Snapshot sampler cursors exclude prefetched, not-yet-yielded batches.
            state["position"] += reader["_steps_since_snapshot"] * self.micro_batch_size
            if reader["_iterator_finished"]:
                raise ValueError(
                    "World-size migration requires an active epoch checkpoint"
                )
            states.append(state)
        migration = sampler.migrate_world_size(states)
        consumed = migration["global_samples_consumed_in_epoch"]
        if consumed % self.global_batch_size:
            raise ValueError(
                "World-size migration must start at a completed global batch"
            )
        combined = {"schemas": {}, "groups": {}}
        for exposure in exposures:
            for schema, count in exposure["schemas"].items():
                combined["schemas"][schema] = combined["schemas"].get(schema, 0) + count
            for schema, groups in exposure["groups"].items():
                target = combined["groups"].setdefault(schema, {})
                for group, count in groups.items():
                    target[group] = target.get(group, 0) + count
        total_exposure = sum(combined["schemas"].values())
        expected_exposure = (
            migration["epoch"] * migration["global_samples_per_epoch"] + consumed
        )
        if total_exposure != expected_exposure:
            raise ValueError(
                "Checkpoint exposure does not match the consumed sampler prefix"
            )
        # Historical totals have one owner so subsequent all-gathers count them once.
        self._mixture_exposure = (
            combined if self._rank == 0 else {"schemas": {}, "groups": {}}
        )
        migration.update(
            {
                "checkpoint": str(Path(load_path)),
                "historical_exposure_samples": total_exposure,
                "historical_exposure_owner_rank": 0,
                "rng_policy": "reinitialize all ranks with actor.seed + rank",
                "actor_seed": int(self.cfg.actor.seed),
                "bitwise_resume": False,
            }
        )
        self._world_size_migrations.append(migration)
        return migration

    def load_checkpoint(self, load_path: str) -> None:
        migration = None
        migration_path = Path(load_path) / "world_size_migrations.json"
        self._world_size_migrations = (
            json.loads(migration_path.read_text()) if migration_path.exists() else []
        )
        migrate = self.cfg.get("runner", {}).get("resume_world_size_migration", False)
        if migrate and (
            self.cfg.data.get("type") != "official_mixture"
            or self.cfg.data.official.get("recipe") != "realized_trajectory"
        ):
            raise ValueError(
                "World-size migration is supported only for official realized trajectories"
            )
        if self.cfg.data.get("type") == "official_mixture":
            root = Path(load_path)
            if self.cfg.data.official.get("recipe") == "realized_trajectory":
                from rlinf.models.embodiment.openpi_official.trajectory import (
                    trajectory_assets,
                )

                saved_assets = json.loads(
                    (root / "upstream_assets" / "realized_trajectory.json").read_text()
                )
                if saved_assets != trajectory_assets(self.data_loader.dataset.root):
                    raise ValueError(
                        "Checkpoint trajectory recipe or normalization differs from this dataset"
                    )
            contract = json.loads((root / "official_data_contract.json").read_text())
            if (
                contract["manifest_fingerprint"]
                != self.data_loader.dataset.manifest_fingerprint
            ):
                raise ValueError(
                    "Official mixture checkpoint data contract differs from this dataset"
                )
            exposures = json.loads((root / "official_exposure.json").read_text())
            if len(exposures) != self._world_size:
                if not migrate:
                    raise ValueError(
                        "Official mixture exact resume requires the original world size"
                    )
                migration = self._migrate_realized_data(load_path, exposures)
            else:
                self._mixture_exposure = exposures[self._rank]
        assets = Path(load_path) / "mixture_assets"
        if self.cfg.data.get("type") == "mixture" and getattr(
            self.data_loader.dataset, "requires_contract", False
        ):
            if not (assets / "contract.json").exists():
                raise ValueError(
                    "Adapted mixture resume requires checkpoint assets; use weights-only warm start for older checkpoints"
                )
            exposures = json.loads((assets / "exposure.json").read_text())
            if len(exposures) != self._world_size:
                raise ValueError(
                    "Mixture exact resume requires the original checkpoint world size"
                )
        if (
            self.cfg.data.get("type") == "mixture"
            and (assets / "contract.json").exists()
        ):
            expected = (self.data_loader.dataset.root / "contract.json").read_bytes()
            if (assets / "contract.json").read_bytes() != expected:
                raise ValueError(
                    "Checkpoint mixture contract differs; use a weights-only warm start for new data"
                )
        super().load_checkpoint(load_path)
        if (assets / "exposure.json").exists():
            self._mixture_exposure = json.loads((assets / "exposure.json").read_text())[
                self._rank
            ]

        if migration is not None:
            self.data_iter = None
            seed_everything(self.cfg.actor.seed + self._rank)
            self.data_iter = iter(self.data_loader)
            self._data_epoch = self.data_loader.sampler.epoch
            self._data_iter_offset = (
                migration["new_rank_position"] // self.micro_batch_size
            )
            self.log_info(
                f"World-size checkpoint migration: {json.dumps(migration, sort_keys=True)}"
            )
            torch.distributed.barrier()
            return

        if isinstance(self.data_loader, StatefulDataLoader):
            all_states = torch.load(
                os.path.join(load_path, "data.pt"), weights_only=False
            )
            state = all_states[self._rank]
            self.data_iter = None
            rebatch = self.cfg.runner.get("resume_rebatch", False)
            if rebatch:
                if (
                    self.cfg.data.get("type") != "official_mixture"
                    or self.cfg.data.official.get("recipe") != "realized_trajectory"
                    or state["_steps_since_snapshot"] != 0
                    or state["_iterator_finished"]
                ):
                    raise ValueError(
                        "Rebatch requires a complete active realized-trajectory reader snapshot"
                    )
                delivered = state["_snapshot"]["_main_snapshot"]["_sampler_iter_state"]
                sampler_state = delivered["sampler_state"]
                position = delivered["samples_yielded"]
                if (
                    position != sampler_state["position"]
                    or position % self.micro_batch_size
                ):
                    raise ValueError(
                        "Rebatch requires an aligned delivered sample prefix"
                    )
                self.data_loader.sampler.load_state_dict(sampler_state)
                self._data_iter_offset = position // self.micro_batch_size
                self.log_info(
                    f"Rebatch reader from delivered position {position} with micro batch {self.micro_batch_size}; checkpoint RNG retained"
                )
            else:
                self.data_loader.load_state_dict(state)
            self.data_iter = iter(self.data_loader)
            if self.cfg.data.get("type") in ("mixture", "official_mixture"):
                self._data_epoch = self.data_loader.sampler.epoch

            rng_path = os.path.join(load_path, "rng.pt")
            if os.path.exists(rng_path):
                all_rng_states = torch.load(rng_path, weights_only=False)
                set_rng_state(all_rng_states[self._rank])

            torch.distributed.barrier()

    def get_max_steps_per_epoch(self):
        if self.data_loader is None:
            return 0
        model_type = SupportedModel(self.cfg.actor.model.model_type)
        if model_type == SupportedModel.OPENPI:
            from rlinf.data.datasets.openpi import (
                get_official_openpi_sft_num_batches,
                is_official_openpi_sft_dataloader,
            )

            num_batches = (
                get_official_openpi_sft_num_batches(self.data_loader)
                if is_official_openpi_sft_dataloader(self.data_loader)
                else len(self.data_loader)
            )
        else:
            return super().get_max_steps_per_epoch()
        return max(1, num_batches // self.gradient_accumulation)
