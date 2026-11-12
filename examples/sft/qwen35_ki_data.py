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

"""Robot and native-split visual-language batches for knowledge insulation."""

import argparse
import ast
import bisect
import hashlib
import io
import itertools
import json
import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from rlinf.models.embodiment.qwen35_fm.fast_tokenizer import (
    FASTActionTokenizer,
    build_ce_inputs,
)
from rlinf.models.embodiment.qwen35_fm.ki_policy import KIFlowMatchingPolicy


class KIRobotDataset(Dataset):
    """Pair a current robot observation with separate continuous and FAST targets.

    State bins represent the current observation's seven joints and two gripper
    drivers. FAST targets are precomputed Qwen IDs and never enter the prefix.
    """

    def __init__(
        self,
        subset,
        mean: torch.Tensor,
        std: torch.Tensor,
        fast_token_sequences: list[list[int]],
        state_bins: torch.Tensor,
        image_cache_file: str | Path | None = None,
        image_size: int = 224,
    ) -> None:
        if len(fast_token_sequences) != len(subset):
            raise ValueError("Expected one FAST target sequence per robot chunk")
        if state_bins.shape != (len(subset), 9) or torch.is_floating_point(state_bins):
            raise ValueError("Expected integer current-state bins with shape [N,9]")
        self.subset = subset
        self.mean = mean
        self.std = std
        self.fast_token_sequences = fast_token_sequences
        self.state_bins = state_bins
        self.image_cache_file = (
            str(Path(image_cache_file).resolve()) if image_cache_file else None
        )
        self.image_size = image_size
        self._cached_images = None
        self.manifest = subset.manifest

    def __getstate__(self):
        # A spawned worker must reopen the memmap instead of pickling its pixels.
        state = self.__dict__.copy()
        state["_cached_images"] = None
        return state

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index):
        from PIL import Image

        index, valid = index if isinstance(index, tuple) else (index, True)
        sample = self.subset.samples[index]
        if self.image_cache_file is None:
            images = [
                image.resize((self.image_size, self.image_size))
                for image in self.subset.images(index)
            ]
        else:
            if self._cached_images is None:
                self._cached_images = np.load(self.image_cache_file, mmap_mode="r")
                expected = (
                    len(self.subset),
                    len(self.subset.cameras),
                    self.image_size,
                    self.image_size,
                    3,
                )
                if (
                    self._cached_images.shape != expected
                    or self._cached_images.dtype != np.uint8
                ):
                    raise ValueError(
                        f"Expected uint8 image cache {expected}, got {self._cached_images.shape}/{self._cached_images.dtype}"
                    )
            images = [Image.fromarray(image) for image in self._cached_images[index]]
        return {
            "actions": (sample["actions"] - self.mean) / self.std,
            "action_mask": sample["mask"] & valid,
            "images": images,
            "cameras": self.subset.cameras,
            "instruction": sample["episode"]["tasks"][0],
            "state_bins": self.state_bins[index],
            "fast_targets": self.fast_token_sequences[index] if valid else [],
        }


class FullKIRobotDataset(Dataset):
    """Normalize lazily decoded native robot chunks using train-only statistics."""

    def __init__(self, dataset, mean, std, state_quantiles: dict) -> None:
        self.dataset = dataset
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)
        self.state_low = torch.tensor(state_quantiles["q01"], dtype=torch.float32)
        high = torch.tensor(state_quantiles["q99"], dtype=torch.float32)
        self.state_span = (high - self.state_low).clamp_min(1e-6)
        if (
            self.mean.shape != (8,)
            or self.std.shape != (8,)
            or not bool((self.std > 0).all())
        ):
            raise ValueError(
                "Expected action mean/std of width 8 with positive standard deviations"
            )
        if (
            self.state_low.shape != (9,)
            or high.shape != (9,)
            or not bool((high >= self.state_low).all())
        ):
            raise ValueError("Expected ordered native state quantiles of width 9")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        actions = sample["actions"].float()
        bins = (
            (
                ((sample["state"].float() - self.state_low) / self.state_span).clamp(
                    0, 1
                )
                * 255
            )
            .round()
            .long()
        )
        return {
            "actions": (actions - self.mean) / self.std,
            "physical_actions": actions,
            "action_mask": sample["action_mask"],
            "state_bins": bins,
            "images": sample["images"],
            "cameras": sample["cameras"],
            "instruction": sample["instruction"],
        }


class _ProcessorCollator:
    """Load only the native CPU processor in the process producing a batch."""

    def __init__(self, model_path: str | Path, processor=None) -> None:
        self.model_path = str(model_path)
        self.processor = processor

    def _get_processor(self):
        if self.processor is None:
            from transformers import AutoProcessor

            self.processor = AutoProcessor.from_pretrained(
                self.model_path, local_files_only=True
            )
        self.processor.tokenizer.padding_side = "left"
        return self.processor

    def _prefix_inputs(self, contents, images):
        processor = self._get_processor()
        texts = [
            processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for content in contents
        ]
        return dict(
            processor(text=texts, images=images, padding=True, return_tensors="pt")
        )


class KIRobotCollator(_ProcessorCollator):
    """Produce prefix-only FM inputs and independently extended CE inputs."""

    def __call__(self, samples) -> dict:
        contents, images, sequences = [], [], []
        for sample in samples:
            content = []
            for camera, image in zip(sample["cameras"], sample["images"], strict=True):
                content.extend(
                    [
                        {
                            "type": "text",
                            "text": f"Camera: {camera.rsplit('.', 1)[-1]}\n",
                        },
                        {"type": "image"},
                    ]
                )
                images.append(image)
            state_text = " ".join(str(int(value)) for value in sample["state_bins"])
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Instruction: {sample['instruction']}\n"
                        "Robot state bins (seven arm joints, left driver, right driver): "
                        f"{state_text}\nPredict the next action chunk."
                    ),
                }
            )
            contents.append(content)
            sequences.append(sample["fast_targets"])
        prefix = self._prefix_inputs(contents, images)
        actions = torch.stack([sample["actions"] for sample in samples])
        action_mask = torch.stack([sample["action_mask"] for sample in samples])
        ce_inputs = build_ce_inputs(
            prefix, sequences, self.processor.tokenizer.pad_token_id
        )
        return {
            "actions": actions,
            "action_mask": action_mask,
            "prefix_inputs": prefix,
            "ce_inputs": ce_inputs,
            "batch_metadata": KIFlowMatchingPolicy.prepare_robot_metadata(
                prefix, ce_inputs, actions, action_mask
            ),
        }


class FullKIRobotCollator(KIRobotCollator):
    """Encode physical FAST targets on CPU without retaining an eager target table."""

    def __init__(
        self,
        model_path: str | Path,
        fast_path: str | Path,
        action_quantiles: dict,
        fast_bridge: dict,
        processor=None,
    ) -> None:
        super().__init__(model_path, processor)
        self.fast_path = str(Path(fast_path).resolve())
        self.action_quantiles = action_quantiles
        self.fast_bridge = dict(fast_bridge)
        self._fast = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["processor"] = None
        state["_fast"] = None
        return state

    def _get_fast(self):
        if self._fast is None:
            self._fast = FASTActionTokenizer.from_pretrained(
                self.fast_path, self.action_quantiles
            )
            if self._fast.processor.vocab_size != self.fast_bridge["fast_vocab_size"]:
                raise ValueError(
                    "Worker FAST vocabulary differs from the attached model vocabulary"
                )
            self._fast.bridge = self.fast_bridge
        return self._fast

    def __call__(self, samples) -> dict:
        physical = torch.stack([sample["physical_actions"] for sample in samples])
        masks = torch.stack([sample["action_mask"] for sample in samples])
        targets = self._get_fast().encode_qwen(physical, masks)
        encoded = [
            {**sample, "fast_targets": sequence}
            for sample, sequence in zip(samples, targets, strict=True)
        ]
        return super().__call__(encoded)


class _VisualChoiceDataset(Dataset):
    """Render canonical image/question/choices/answer-index records."""

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        from PIL import Image

        index, valid = index if isinstance(index, tuple) else (index, True)
        sample = self.samples[index]
        image = Image.open(io.BytesIO(sample["image"]["bytes"])).convert("RGB")
        image = image.resize((self.image_size, self.image_size))
        choices = "\n".join(
            f"{chr(65 + i)}. {choice}" for i, choice in enumerate(sample["choices"])
        )
        answer = sample["answer"]
        return {
            "image": image,
            "question": f"{sample['question']}\n{choices}\nAnswer with the option letter and text.",
            "answer": f"{chr(65 + answer)}. {sample['choices'][answer]}",
            "valid": valid,
        }


class ScienceQADataset(_VisualChoiceDataset):
    """Read real image/answer examples from an existing native ScienceQA split.

    data_root contains the train/validation/test parquet files. Only examples
    with image bytes and a valid answer index are selected; metadata excludes
    image bytes, lecture, and solution text.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        limit: int | None = None,
        image_size: int = 224,
    ) -> None:
        import pyarrow.parquet as pq

        if limit is not None and limit < 1:
            raise ValueError("ScienceQA limit must be positive when specified")
        root = Path(data_root).resolve()
        files = sorted(root.glob(f"{split}-*.parquet"))
        if not files:
            raise ValueError(
                f"No ScienceQA parquet files for split {split!r} in {root}"
            )
        self.samples = []
        self.manifest = []
        self.image_size = image_size
        self.info = {
            "data_root": str(root),
            "split": split,
            "source_rows": sum(
                pq.ParquetFile(path).metadata.num_rows for path in files
            ),
        }
        for path in files:
            row_offset = 0
            for batch in pq.ParquetFile(path).iter_batches(
                batch_size=256, columns=["image", "question", "choices", "answer"]
            ):
                rows = batch.to_pylist()
                for offset, row in enumerate(rows):
                    if not row["image"] or not row["image"]["bytes"]:
                        continue
                    if not row["choices"] or not 0 <= row["answer"] < len(
                        row["choices"]
                    ):
                        continue
                    self.samples.append(row)
                    self.manifest.append(
                        {
                            "id": f"{split}:{path.name}:{row_offset + offset}",
                            "file": str(path),
                            "row_index": row_offset + offset,
                            "question": row["question"],
                            "choices": row["choices"],
                            "answer_index": int(row["answer"]),
                        }
                    )
                    if limit is not None and len(self.samples) == limit:
                        break
                row_offset += len(rows)
                if limit is not None and len(self.samples) == limit:
                    break
            if limit is not None and len(self.samples) == limit:
                break
        if not self.samples:
            raise ValueError(f"No supervised image examples in ScienceQA {split!r}")
        self.info["selected_image_examples"] = len(self.samples)


class VLCollator(_ProcessorCollator):
    """Supervise only the real visual-language answer and assistant end token."""

    def __call__(self, samples) -> dict:
        contents = [
            [
                {"type": "image"},
                {"type": "text", "text": sample["question"]},
            ]
            for sample in samples
        ]
        prefix = self._prefix_inputs(contents, [sample["image"] for sample in samples])
        tokenizer = self.processor.tokenizer
        end_token = tokenizer.convert_tokens_to_ids("<|im_end|>")
        targets = [
            tokenizer.encode(sample["answer"], add_special_tokens=False) + [end_token]
            if sample["valid"]
            else []
            for sample in samples
        ]
        inputs = build_ce_inputs(prefix, targets, tokenizer.pad_token_id)
        return {
            "vlm_inputs": inputs,
            "batch_metadata": KIFlowMatchingPolicy.prepare_ce_metadata(inputs),
        }


def _robo2vlm_group(item_id: str) -> str:
    """Recover the official exporter's episode-directory group, not just its number."""
    match = re.fullmatch(r"(.+)_q\d+", item_id)
    if match is None:
        raise ValueError(
            f"Expected official Robo2VLM episode-directory_qN ID, got {item_id!r}"
        )
    return match[1]


class Robo2VLMDataset(_VisualChoiceDataset):
    """Metadata-only group-heldout index of the locally available Robo2VLM train shards."""

    def __init__(
        self,
        data_root,
        split,
        limit=None,
        image_size=224,
        seed=42,
        validation_fraction=0.05,
        excluded_image_hashes=None,
    ):
        import pyarrow.parquet as pq

        if split not in {"train", "validation"} or not 0 < validation_fraction < 1:
            raise ValueError(
                "Expected derived train/validation split and a fraction between zero and one"
            )
        if limit is not None and limit < 1:
            raise ValueError("Robo2VLM limit must be positive")
        root = Path(data_root).resolve()
        train_files, test_files = (
            sorted(root.glob("train-*.parquet")),
            sorted(root.glob("test-*.parquet")),
        )
        if not train_files or not test_files:
            raise ValueError(
                "Robo2VLM requires local train shards and the complete official-test ID index"
            )
        test_numbers = [
            re.fullmatch(r"test-(\d+)-of-(\d+)\.parquet", p.name) for p in test_files
        ]
        if any(x is None for x in test_numbers) or {
            int(x[1]) for x in test_numbers
        } != set(range(int(test_numbers[0][2]))):
            raise ValueError(
                "Complete native Robo2VLM official-test shards are required for group exclusion"
            )
        test_groups = set()
        for path in test_files:
            for batch in pq.ParquetFile(path).iter_batches(
                batch_size=8192, columns=["id"]
            ):
                test_groups.update(
                    _robo2vlm_group(x) for x in batch.column(0).to_pylist()
                )
        source_rows, train_groups, buckets = 0, set(), {}
        for path in train_files:
            offset = 0
            for batch in pq.ParquetFile(path).iter_batches(
                batch_size=4096, columns=["id", "question", "choices", "correct_answer"]
            ):
                rows = batch.to_pylist()
                for index, row in enumerate(rows):
                    source_rows += 1
                    group = _robo2vlm_group(row["id"])
                    train_groups.add(group)
                    if group in test_groups:
                        continue
                    value = (
                        int.from_bytes(
                            hashlib.sha256(f"{seed}:{group}".encode()).digest()[:8],
                            "big",
                        )
                        / 2**64
                    )
                    if (value < validation_fraction) != (split == "validation"):
                        continue
                    choices, answer = (
                        ast.literal_eval(row["choices"]),
                        int(row["correct_answer"]),
                    )
                    if (
                        not isinstance(choices, list)
                        or not choices
                        or not all(isinstance(x, str) for x in choices)
                        or not 0 <= answer < len(choices)
                    ):
                        raise ValueError(
                            f"Invalid Robo2VLM choices/answer for {row['id']}"
                        )
                    record = {
                        "id": row["id"],
                        "file": str(path),
                        "row_index": offset + index,
                        "episode_directory_group": group,
                        "question": row["question"],
                        "choices": choices,
                        "answer": answer,
                    }
                    buckets.setdefault(group, []).append(record)
                offset += len(rows)
        groups = sorted(
            buckets, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).digest()
        )
        ordered = [
            sorted(
                buckets[g],
                key=lambda x: hashlib.sha256(f"{seed}:{x['id']}".encode()).digest(),
            )
            for g in groups
        ]
        self.manifest = []
        for layer in itertools.zip_longest(*ordered):
            for record in layer:
                if record is not None:
                    self.manifest.append(record)
                    if limit is not None and len(self.manifest) == limit:
                        break
            if limit is not None and len(self.manifest) == limit:
                break
        if not self.manifest:
            raise ValueError(f"No eligible Robo2VLM examples in {split}")
        self.samples = self.manifest
        self.image_size = image_size
        self.image_hashes = set()
        self.excluded_image_hashes = excluded_image_hashes or set()
        self.image_cache_file, self._cached_images, self._image_table = None, None, None
        self._image_table_key = None
        expected_shards = {
            int(re.fullmatch(r"train-\d+-of-(\d+)\.parquet", p.name)[1])
            for p in train_files
        }
        if len(expected_shards) != 1:
            raise ValueError("Robo2VLM shards disagree on their expected total")
        self.info = {
            "dataset": "keplerccc/Robo2VLM-1",
            "data_root": str(root),
            "split": split,
            "group_identity": "official exporter sanitized episode-directory basename recovered by removing terminal _qN; native source_file absent from HF rows",
            "grouping_source": "https://github.com/KeplerC/robo2VLM/blob/main/scripts/create_huggingface_dataset.py",
            "split_protocol": "exclude all official-test episode-directory groups, then stable SHA256(seed:group) holdout; round-robin rows across stable-hash ordered groups",
            "seed": seed,
            "validation_fraction": validation_fraction,
            "physical_train_shards_available": len(train_files),
            "official_train_shards": next(iter(expected_shards)),
            "source_rows": source_rows,
            "official_test_rows": sum(
                pq.ParquetFile(p).metadata.num_rows for p in test_files
            ),
            "source_train_groups": len(train_groups),
            "official_test_groups": len(test_groups),
            "excluded_test_overlap_groups": len(train_groups & test_groups),
            "eligible_split_groups": len(groups),
            "eligible_split_rows": sum(map(len, buckets.values())),
            "requested_limit": limit,
            "selected_image_examples": len(self),
            "selected_groups": len(
                {x["episode_directory_group"] for x in self.manifest}
            ),
            "image_disjoint_scope": "resized RGB pixel SHA256 of selected examples; official test excluded by episode-directory group only",
            "image_hashes_checked": False,
            "native_source_episode_id_available": False,
            "source_files": [
                {
                    "file": str(p),
                    "size": p.stat().st_size,
                    "mtime_ns": p.stat().st_mtime_ns,
                }
                for p in train_files + test_files
            ],
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cached_images"] = state["_image_table"] = None
        return state

    def __getitem__(self, index):
        from PIL import Image

        index, valid = index if isinstance(index, tuple) else (index, True)
        record = self.manifest[index]
        if self.image_cache_file:
            if self._cached_images is None:
                self._cached_images = np.load(self.image_cache_file, mmap_mode="r")
            image = Image.fromarray(self._cached_images[record["cache_row"]])
        else:
            import pyarrow.parquet as pq

            group, local = _row_group_location(record["file"], record["row_index"])
            key = (record["file"], group)
            if key != self._image_table_key:
                self._image_table = pq.ParquetFile(record["file"]).read_row_group(
                    group, columns=["image"], use_threads=False
                )
                self._image_table_key = key
            raw = self._image_table.column(0)[local].as_py()["bytes"]
            image = (
                Image.open(io.BytesIO(raw))
                .convert("RGB")
                .resize((self.image_size, self.image_size))
            )
        choices = "\n".join(
            f"{chr(65 + i)}. {choice}" for i, choice in enumerate(record["choices"])
        )
        answer = record["answer"]
        return {
            "image": image,
            "question": f"{record['question']}\n{choices}\nAnswer with the option letter and text.",
            "answer": f"{chr(65 + answer)}. {record['choices'][answer]}",
            "valid": valid,
        }

    @classmethod
    def from_cache(
        cls,
        cache_file,
        data_root,
        split,
        limit,
        image_size,
        seed,
        validation_fraction,
        excluded_image_hashes=None,
    ):
        cache_file = Path(cache_file).resolve()
        meta = json.loads(cache_file.with_suffix(".json").read_text())
        for key, expected in [
            ("complete", True),
            ("format", "robo2vlm"),
            ("data_root", str(Path(data_root).resolve())),
            ("split", split),
            ("image_size", image_size),
            ("seed", seed),
            ("validation_fraction", validation_fraction),
        ]:
            if meta[key] != expected:
                raise ValueError(
                    f"VL image cache {key} differs from requested protocol"
                )
        if (
            limit is not None
            and meta["requested_limit"] is not None
            and limit > meta["requested_limit"]
        ):
            raise ValueError("Requested VL limit exceeds the prepared cache selection")
        for source in meta["info"]["source_files"]:
            stat = Path(source["file"]).stat()
            if stat.st_size != source["size"] or stat.st_mtime_ns != source["mtime_ns"]:
                raise ValueError(
                    "Robo2VLM source shards changed after VL cache preparation"
                )
        data = cls.__new__(cls)
        data.manifest = (
            meta["records"][:limit] if limit is not None else meta["records"]
        )
        data.samples = data.manifest
        data.info = dict(meta["info"])
        data.info.update(
            selected_image_examples=len(data.manifest),
            selected_groups=len({x["episode_directory_group"] for x in data.manifest}),
            prepared_cache_examples=len(meta["records"]),
        )
        data.image_hashes = {x["image_sha256"] for x in data.manifest}
        if data.image_hashes & (excluded_image_hashes or set()):
            raise ValueError(
                "Prepared train VL cache overlaps selected validation image hashes"
            )
        data.image_size, data.image_cache_file = image_size, str(cache_file)
        data._cached_images = data._image_table = None
        data._image_table_key = None
        data.excluded_image_hashes = excluded_image_hashes or set()
        array = np.load(cache_file, mmap_mode="r")
        if (
            array.shape != (len(meta["records"]), image_size, image_size, 3)
            or array.dtype != np.uint8
        ):
            raise ValueError(
                "VL cache array geometry or dtype differs from its metadata"
            )
        return data


@lru_cache(maxsize=256)
def _row_group_starts(path):
    import pyarrow.parquet as pq

    metadata = pq.ParquetFile(path).metadata
    starts = [0]
    for index in range(metadata.num_row_groups):
        starts.append(starts[-1] + metadata.row_group(index).num_rows)
    return starts


def _row_group_location(path, row_index):
    starts = _row_group_starts(path)
    group = bisect.bisect_right(starts, row_index) - 1
    return group, row_index - starts[group]


_VL_PIXELS = None
_VL_SIZE = None


def _init_vl_cache(path, image_size):
    global _VL_PIXELS, _VL_SIZE
    _VL_PIXELS = np.load(path, mmap_mode="r+")
    _VL_SIZE = image_size


def _write_vl_group(task):
    import pyarrow.parquet as pq
    from PIL import Image

    path, group, indices = task
    table = pq.ParquetFile(path).read_row_group(
        group, columns=["image"], use_threads=False
    )
    hashes = []
    for output_row, local_row in indices:
        image = table.column(0)[local_row].as_py()
        if not image or not image["bytes"]:
            raise ValueError(
                f"Missing real VL image bytes in {path}, row group {group}, row {local_row}"
            )
        image = (
            Image.open(io.BytesIO(image["bytes"]))
            .convert("RGB")
            .resize((_VL_SIZE, _VL_SIZE))
        )
        _VL_PIXELS[output_row] = np.asarray(image)
        hashes.append((output_row, hashlib.sha256(image.tobytes()).hexdigest()))
    return hashes


def cache_vl_images(dataset, output_file, workers=8, excluded_image_hashes=None):
    """Decode selected row groups once on CPU and publish a complete uint8 cache."""
    from rlinf.utils.logging import get_logger

    output = Path(output_file).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(".json").exists():
        raise ValueError(f"VL cache output already exists: {output}")
    pending = output.with_suffix(".building.npy")
    array = np.lib.format.open_memmap(
        pending,
        mode="w+",
        dtype=np.uint8,
        shape=(len(dataset), dataset.image_size, dataset.image_size, 3),
    )
    array.flush()
    tasks = {}
    for index, record in enumerate(dataset.manifest):
        group, local = _row_group_location(record["file"], record["row_index"])
        tasks.setdefault((record["file"], group), []).append((index, local))
    hashes = {}
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_vl_cache,
        initargs=(str(pending), dataset.image_size),
    ) as executor:
        for result in executor.map(
            _write_vl_group,
            [(path, group, indices) for (path, group), indices in tasks.items()],
        ):
            hashes.update(result)
            get_logger().info(f"VL cached {len(hashes)}/{len(dataset)} selected images")
    array.flush()
    excluded = excluded_image_hashes or dataset.excluded_image_hashes
    retained = [index for index in range(len(dataset)) if hashes[index] not in excluded]
    if not retained:
        raise ValueError("All selected VL training images overlap validation")
    records = []
    for row, index in enumerate(retained):
        records.append(
            dict(dataset.manifest[index], cache_row=row, image_sha256=hashes[index])
        )
    if len(retained) == len(dataset):
        del array
        pending.replace(output)
    else:
        final = np.lib.format.open_memmap(
            output,
            mode="w+",
            dtype=np.uint8,
            shape=(len(retained), dataset.image_size, dataset.image_size, 3),
        )
        for start in range(0, len(retained), 256):
            final[start : start + 256] = array[retained[start : start + 256]]
        final.flush()
        del final, array
        pending.unlink()
    info = dict(dataset.info)
    info.update(
        selected_image_examples=len(records),
        selected_groups=len({x["episode_directory_group"] for x in records}),
        excluded_duplicate_images=len(dataset) - len(records),
        image_hashes_checked=True,
    )
    meta = {
        "schema_version": 1,
        "complete": True,
        "format": "robo2vlm",
        "data_root": info["data_root"],
        "split": info["split"],
        "image_size": dataset.image_size,
        "seed": info["seed"],
        "validation_fraction": info["validation_fraction"],
        "requested_limit": info["requested_limit"],
        "info": info,
        "records": records,
    }
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(meta, indent=2))
    temporary.replace(output.with_suffix(".json"))
    dataset.manifest, dataset.samples, dataset.info = records, records, info
    dataset.image_hashes = {x["image_sha256"] for x in records}
    dataset.image_cache_file, dataset._cached_images = str(output), None
    return str(output)


def build_vl_dataset(
    data_root,
    split,
    *,
    format="robo2vlm",
    limit=None,
    image_size=224,
    cache_dir=None,
    seed=42,
    validation_fraction=0.05,
    excluded_image_hashes=None,
):
    """Load an explicit real VL source, using metadata+memmap for prepared Robo2VLM."""
    root = Path(data_root).resolve()
    if (root / "data").is_dir():
        root = root / "data"
    if format == "robo2vlm":
        if cache_dir is not None:
            return Robo2VLMDataset.from_cache(
                Path(cache_dir) / f"{split}.npy",
                root,
                split,
                limit,
                image_size,
                seed,
                validation_fraction,
                excluded_image_hashes,
            )
        return Robo2VLMDataset(
            root,
            split,
            limit,
            image_size,
            seed,
            validation_fraction,
            excluded_image_hashes,
        )
    if format == "scienceqa":
        if cache_dir is not None:
            raise ValueError(
                "Prepared VL cache currently supports the selected Robo2VLM source"
            )
        return ScienceQADataset(root, split, limit, image_size)
    raise ValueError(f"Expected robo2vlm or scienceqa, got {format!r}")


def _prepare_vl_cache():
    from rlinf.utils.logging import get_logger

    parser = argparse.ArgumentParser(
        description="Prepare selected group-disjoint Robo2VLM image caches on CPU"
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-limit", type=int, default=8192)
    parser.add_argument("--val-limit", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    val = build_vl_dataset(
        args.data_path,
        "validation",
        limit=args.val_limit,
        image_size=args.image_size,
        seed=args.seed,
    )
    cache_vl_images(val, Path(args.output) / "validation.npy", args.workers)
    train = build_vl_dataset(
        args.data_path,
        "train",
        limit=args.train_limit,
        image_size=args.image_size,
        seed=args.seed,
    )
    cache_vl_images(
        train, Path(args.output) / "train.npy", args.workers, val.image_hashes
    )
    if train.image_hashes & val.image_hashes or {
        x["episode_directory_group"] for x in train.manifest
    } & {x["episode_directory_group"] for x in val.manifest}:
        raise RuntimeError("Selected Robo2VLM train/validation cache overlaps")
    get_logger().info(
        json.dumps({"train": train.info, "validation": val.info}, indent=2)
    )


if __name__ == "__main__":
    _prepare_vl_cache()
