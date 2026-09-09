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

"""Read instance-ID frames from recorded B1K challenge demonstrations."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from rlinf.data.b1k_grounded import (
    CameraID,
    EntityResolver,
    parse_instance_id_mapping,
)

_UNIQUE_ID_KEYS = {
    CameraID.HEAD: "robot_r1::robot_r1:zed_link:Camera:0::unique_ins_ids",
    CameraID.LEFT_WRIST: (
        "robot_r1::robot_r1:left_realsense_link:Camera:0::unique_ins_ids"
    ),
    CameraID.RIGHT_WRIST: (
        "robot_r1::robot_r1:right_realsense_link:Camera:0::unique_ins_ids"
    ),
}


@dataclasses.dataclass(frozen=True)
class DecodeDiagnostics:
    """Nearest-palette diagnostics for one decoded segmentation frame."""

    unique_encoded_colors: int
    mean_color_error: float
    p99_color_error: float
    ambiguous_pixel_fraction: float


@dataclasses.dataclass(frozen=True)
class EpisodeSegmentationMetadata:
    """Instance mapping and per-camera palettes for one recorded episode."""

    resolver: EntityResolver
    unique_instance_ids: dict[CameraID, tuple[int, ...]]
    n_steps: int

    @classmethod
    def from_path(cls, path: str | Path) -> EpisodeSegmentationMetadata:
        """Load segmentation metadata from an episode JSON file."""
        with Path(path).open() as file:
            metadata = json.load(file)
        try:
            mapping = parse_instance_id_mapping(metadata["ins_id_mapping"])
            n_steps = int(metadata["n_steps"])
            unique_instance_ids = {
                camera: tuple(int(value) for value in metadata[key])
                for camera, key in _UNIQUE_ID_KEYS.items()
            }
        except KeyError as error:
            raise ValueError(
                f"Missing segmentation metadata key {error.args[0]!r}."
            ) from error
        return cls(
            resolver=EntityResolver(mapping),
            unique_instance_ids=unique_instance_ids,
            n_steps=n_steps,
        )


def generate_segmentation_palette(num_ids: int) -> np.ndarray:
    """Generate the palette used by the B1K segmentation video writer."""
    if num_ids <= 0:
        raise ValueError("num_ids must be positive.")
    bins = int(np.ceil(num_ids ** (1 / 3)))
    first = np.linspace(16, 235, bins)
    second = np.linspace(16, 240, bins)
    third = np.linspace(16, 240, bins)
    colors = [
        (channel_0, channel_1, channel_2)
        for channel_0 in first
        for channel_1 in second
        for channel_2 in third
    ]
    return np.asarray(colors[:num_ids], dtype=np.uint8)


def _import_cv2():
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "Recorded-video probing requires OpenCV (import name: cv2)."
        ) from error
    return cv2


def read_rgb_video_frame(path: str | Path, frame_index: int) -> np.ndarray:
    """Read an exact zero-based video frame as an RGB uint8 array."""
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative.")
    cv2 = _import_cv2()
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise OSError(f"Could not open video {path}.")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_index >= frame_count:
            raise IndexError(
                f"Frame {frame_index} is outside video with {frame_count} frames."
            )
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, bgr = capture.read()
        if not success:
            raise OSError(f"Could not decode frame {frame_index} from {path}.")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def read_rgb_video_frames(
    path: str | Path, frame_indices: list[int] | tuple[int, ...]
) -> dict[int, np.ndarray]:
    """Read several zero-based frames while reusing one video decoder."""
    requested = sorted(set(frame_indices))
    if not requested:
        return {}
    if requested[0] < 0:
        raise ValueError("frame indices must be non-negative.")

    cv2 = _import_cv2()
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise OSError(f"Could not open video {path}.")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if requested[-1] >= frame_count:
            raise IndexError(
                f"Frame {requested[-1]} is outside video with {frame_count} frames."
            )

        result = {}
        current_position = 0
        for frame_index in requested:
            gap = frame_index - current_position
            if gap < 0 or gap > 64:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                current_position = frame_index
            else:
                for _ in range(gap):
                    if not capture.grab():
                        raise OSError(
                            f"Could not advance to frame {frame_index} in {path}."
                        )
                    current_position += 1
            success, bgr = capture.read()
            if not success:
                raise OSError(f"Could not decode frame {frame_index} from {path}.")
            current_position += 1
            result[frame_index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return result
    finally:
        capture.release()


def decode_segmentation_rgb(
    encoded_rgb: np.ndarray, instance_ids: tuple[int, ...]
) -> tuple[np.ndarray, DecodeDiagnostics]:
    """Recover instance IDs using the dataset writer's nearest-palette rule."""
    encoded_rgb = np.asarray(encoded_rgb)
    if encoded_rgb.ndim != 3 or encoded_rgb.shape[-1] != 3:
        raise ValueError(
            f"encoded_rgb must have shape (H, W, 3), got {encoded_rgb.shape}."
        )
    if not instance_ids:
        raise ValueError("instance_ids must not be empty.")

    palette = generate_segmentation_palette(len(instance_ids))
    colors, inverse, counts = np.unique(
        encoded_rgb.reshape(-1, 3),
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    deltas = colors[:, None, :].astype(np.int32) - palette[None, :, :].astype(np.int32)
    squared_distances = np.sum(deltas * deltas, axis=-1)
    nearest_indices = np.argmin(squared_distances, axis=1)
    nearest_squared = squared_distances[
        np.arange(squared_distances.shape[0]), nearest_indices
    ]
    if len(instance_ids) > 1:
        two_nearest = np.partition(squared_distances, 1, axis=1)[:, :2]
        ambiguous = two_nearest[:, 0] == two_nearest[:, 1]
    else:
        ambiguous = np.zeros(colors.shape[0], dtype=bool)

    decoded_values = np.asarray(instance_ids, dtype=np.int64)[nearest_indices]
    segmentation = decoded_values[inverse].reshape(encoded_rgb.shape[:2])
    errors = np.sqrt(nearest_squared)
    pixel_count = int(counts.sum())
    diagnostics = DecodeDiagnostics(
        unique_encoded_colors=len(colors),
        mean_color_error=float(np.dot(errors, counts) / pixel_count),
        p99_color_error=float(
            np.quantile(np.repeat(errors, counts), 0.99, method="higher")
        ),
        ambiguous_pixel_fraction=float(np.dot(ambiguous, counts) / pixel_count),
    )
    return segmentation, diagnostics


def decode_segmentation_rgb_fast(
    encoded_rgb: np.ndarray, instance_ids: tuple[int, ...]
) -> np.ndarray:
    """Decode a frame exactly using the palette's Cartesian-grid structure.

    The official nearest-palette decoder compares every pixel with every color.
    Palette colors form a Cartesian grid, so the closest full-grid color is the
    combination of the closest value in each channel. Only combinations beyond
    the truncated final palette need a small exhaustive fallback.
    """
    encoded_rgb = np.asarray(encoded_rgb)
    if encoded_rgb.ndim != 3 or encoded_rgb.shape[-1] != 3:
        raise ValueError(
            f"encoded_rgb must have shape (H, W, 3), got {encoded_rgb.shape}."
        )
    if not instance_ids:
        raise ValueError("instance_ids must not be empty.")

    num_ids = len(instance_ids)
    bins = int(np.ceil(num_ids ** (1 / 3)))
    channel_levels = (
        np.linspace(16, 235, bins).astype(np.uint8),
        np.linspace(16, 240, bins).astype(np.uint8),
        np.linspace(16, 240, bins).astype(np.uint8),
    )
    nearest_channels = [
        np.argmin(
            np.abs(
                encoded_rgb[..., channel, None].astype(np.int16)
                - levels.astype(np.int16)
            ),
            axis=-1,
        )
        for channel, levels in enumerate(channel_levels)
    ]
    palette_indices = (
        nearest_channels[0] * bins + nearest_channels[1]
    ) * bins + nearest_channels[2]

    invalid = palette_indices >= num_ids
    if invalid.any():
        palette = generate_segmentation_palette(num_ids)
        pixels = encoded_rgb[invalid].astype(np.int32)
        deltas = pixels[:, None, :] - palette[None, :, :].astype(np.int32)
        palette_indices[invalid] = np.argmin(np.sum(deltas * deltas, axis=-1), axis=1)

    return np.asarray(instance_ids, dtype=np.int64)[palette_indices]


def read_segmentation_video_frame(
    path: str | Path,
    frame_index: int,
    instance_ids: tuple[int, ...],
) -> tuple[np.ndarray, DecodeDiagnostics]:
    """Read and decode one recorded segmentation frame."""
    return decode_segmentation_rgb(
        read_rgb_video_frame(path, frame_index), instance_ids
    )


def read_segmentation_video_frames(
    path: str | Path,
    frame_indices: list[int] | tuple[int, ...],
    instance_ids: tuple[int, ...],
) -> dict[int, np.ndarray]:
    """Read and efficiently decode several recorded segmentation frames."""
    encoded_frames = read_rgb_video_frames(path, frame_indices)
    return {
        frame_index: decode_segmentation_rgb_fast(encoded_rgb, instance_ids)
        for frame_index, encoded_rgb in encoded_frames.items()
    }


def episode_metadata_path(
    dataset_root: str | Path, task_index: int, episode_index: int
) -> Path:
    """Return the standard metadata path for an episode."""
    return (
        Path(dataset_root)
        / "meta"
        / "episodes"
        / f"task-{task_index:04d}"
        / f"episode_{episode_index:08d}.json"
    )


def episode_video_path(
    dataset_root: str | Path,
    task_index: int,
    episode_index: int,
    modality: str,
    camera: CameraID,
) -> Path:
    """Return the standard path for one episode video stream."""
    return (
        Path(dataset_root)
        / "videos"
        / f"task-{task_index:04d}"
        / f"observation.images.{modality}.{camera.value}"
        / f"episode_{episode_index:08d}.mp4"
    )
