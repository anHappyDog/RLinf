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

"""Pure NumPy geometry for B1K mask-based 2D grounding."""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Mapping

import numpy as np

from .entity_resolver import EntityResolver
from .schema import (
    CameraID,
    EntityArgument,
    GroundedControlSpec,
    Grounding2D,
    PartArgument,
    Role,
)


@dataclasses.dataclass(frozen=True)
class GroundingConfig:
    """Visibility thresholds used when converting masks to groundings."""

    min_visible_pixels: int = 1
    min_visible_fraction: float = 0.0
    min_component_pixels: int = 1
    min_component_fraction_of_largest: float = 0.0

    def __post_init__(self) -> None:
        if self.min_visible_pixels <= 0:
            raise ValueError("min_visible_pixels must be positive.")
        if not 0.0 <= self.min_visible_fraction <= 1.0:
            raise ValueError("min_visible_fraction must be in [0, 1].")
        if self.min_component_pixels <= 0:
            raise ValueError("min_component_pixels must be positive.")
        if not 0.0 <= self.min_component_fraction_of_largest <= 1.0:
            raise ValueError("min_component_fraction_of_largest must be in [0, 1].")


DEFAULT_CAMERA_PRIORITY = (
    CameraID.HEAD,
    CameraID.LEFT_WRIST,
    CameraID.RIGHT_WRIST,
)

_BUTTON_PART_SKILLS = frozenset(("press", "turn off switch", "turn on switch"))
_BUTTON_PART_CANDIDATES = ("toggle button", "button")
_MIN_BUTTON_BBOX_FILL_FRACTION = 0.05


def mask_for_instance_ids(
    segmentation: np.ndarray, instance_ids: Collection[int]
) -> np.ndarray:
    """Return the union mask for a collection of visual-mesh IDs."""
    segmentation = np.asarray(segmentation)
    if segmentation.ndim != 2:
        raise ValueError(
            f"segmentation must have shape (H, W), got {segmentation.shape}."
        )
    if not instance_ids:
        return np.zeros(segmentation.shape, dtype=bool)
    return np.isin(segmentation, tuple(instance_ids))


def filter_mask_components(
    mask: np.ndarray,
    *,
    min_component_pixels: int,
    min_component_fraction_of_largest: float,
) -> np.ndarray:
    """Remove tiny disconnected components while always retaining the largest.

    This is useful for masks recovered from lossy videos, where a few decoded
    pixels can be assigned to the wrong palette entry. It is disabled by the
    default :class:`GroundingConfig` and therefore does not alter simulator
    masks unless explicitly requested.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any() or (
        min_component_pixels <= 1 and min_component_fraction_of_largest <= 0.0
    ):
        return mask
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "Connected-component filtering requires OpenCV (import name: cv2)."
        ) from error

    component_count, labels, statistics, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if component_count <= 1:
        return mask
    areas = statistics[1:, cv2.CC_STAT_AREA]
    largest_area = int(areas.max())
    threshold = max(
        min_component_pixels,
        int(np.ceil(largest_area * min_component_fraction_of_largest)),
    )
    retained_labels = np.flatnonzero(areas >= threshold) + 1
    largest_label = int(np.argmax(areas)) + 1
    if largest_label not in retained_labels:
        retained_labels = np.append(retained_labels, largest_label)
    return np.isin(labels, retained_labels)


def grounding_from_mask(
    mask: np.ndarray,
    camera: CameraID,
    *,
    config: GroundingConfig = GroundingConfig(),
) -> Grounding2D | None:
    """Convert a binary mask to a tight bbox and an in-mask point.

    The normalized bbox uses half-open image-edge coordinates. A mask spanning
    pixels ``x_min..x_max`` becomes ``[x_min / W, (x_max + 1) / W]``. This
    convention gives a one-pixel object nonzero area and maps a full-image mask
    exactly to ``[0, 0, 1, 1]``.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask must have shape (H, W), got {mask.shape}.")
    mask = filter_mask_components(
        mask.astype(bool, copy=False),
        min_component_pixels=config.min_component_pixels,
        min_component_fraction_of_largest=(config.min_component_fraction_of_largest),
    )
    height, width = mask.shape
    if height == 0 or width == 0:
        raise ValueError("mask dimensions must be nonzero.")

    y_coordinates, x_coordinates = np.nonzero(mask)
    visible_pixels = int(x_coordinates.size)
    visible_fraction = visible_pixels / (height * width)
    if (
        visible_pixels < config.min_visible_pixels
        or visible_fraction < config.min_visible_fraction
    ):
        return None

    x_min = int(x_coordinates.min())
    x_max = int(x_coordinates.max())
    y_min = int(y_coordinates.min())
    y_max = int(y_coordinates.max())

    centroid_x = float(x_coordinates.mean())
    centroid_y = float(y_coordinates.mean())
    nearest_index = np.argmin(
        (x_coordinates - centroid_x) ** 2 + (y_coordinates - centroid_y) ** 2
    )
    point_x = int(x_coordinates[nearest_index])
    point_y = int(y_coordinates[nearest_index])

    return Grounding2D(
        camera=camera,
        bbox_xyxy=(
            x_min / width,
            y_min / height,
            (x_max + 1) / width,
            (y_max + 1) / height,
        ),
        visible_pixels=visible_pixels,
        visible_fraction=visible_fraction,
        point_xy=((point_x + 0.5) / width, (point_y + 0.5) / height),
    )


def ground_instance_ids(
    segmentations: Mapping[CameraID, np.ndarray],
    instance_ids: Collection[int],
    *,
    config: GroundingConfig = GroundingConfig(),
) -> dict[CameraID, Grounding2D]:
    """Ground the union of several mesh IDs in every available camera."""
    groundings = {}
    for camera, segmentation in segmentations.items():
        grounding = grounding_from_mask(
            mask_for_instance_ids(segmentation, instance_ids),
            camera,
            config=config,
        )
        if grounding is not None:
            groundings[camera] = grounding
    return groundings


def ground_button_instance_ids(
    segmentations: Mapping[CameraID, np.ndarray],
    button_instance_ids: Collection[int],
    parent_instance_ids: Collection[int],
    *,
    config: GroundingConfig = GroundingConfig(),
) -> dict[CameraID, Grounding2D]:
    """Ground a button inside the main visible component of its parent object.

    Recorded B1K instance masks are stored in lossy videos.  Palette decoding
    can therefore produce disconnected false-positive pixels for small parts.
    The true button must be on or immediately beside the parent object's main
    visible component, which provides a deterministic spatial filter.
    """
    groundings = {}
    component_config = dataclasses.replace(
        config,
        min_component_pixels=1,
        min_component_fraction_of_largest=1.0,
    )
    for camera, segmentation in segmentations.items():
        parent_mask = filter_mask_components(
            mask_for_instance_ids(segmentation, parent_instance_ids),
            min_component_pixels=1,
            min_component_fraction_of_largest=1.0,
        )
        if not parent_mask.any():
            continue

        height, width = parent_mask.shape
        ys, xs = np.nonzero(parent_mask)
        padding_y = max(1, int(np.ceil(height * 0.02)))
        padding_x = max(1, int(np.ceil(width * 0.02)))
        y_min = max(0, int(ys.min()) - padding_y)
        y_max = min(height, int(ys.max()) + 1 + padding_y)
        x_min = max(0, int(xs.min()) - padding_x)
        x_max = min(width, int(xs.max()) + 1 + padding_x)
        parent_region = np.zeros(parent_mask.shape, dtype=bool)
        parent_region[y_min:y_max, x_min:x_max] = True
        button_mask = (
            mask_for_instance_ids(segmentation, button_instance_ids) & parent_region
        )
        button_mask = filter_mask_components(
            button_mask,
            min_component_pixels=1,
            min_component_fraction_of_largest=1.0,
        )
        button_ys, button_xs = np.nonzero(button_mask)
        if button_xs.size == 0:
            continue
        bbox_pixels = (int(button_xs.max()) - int(button_xs.min()) + 1) * (
            int(button_ys.max()) - int(button_ys.min()) + 1
        )
        if button_xs.size / bbox_pixels < _MIN_BUTTON_BBOX_FILL_FRACTION:
            continue
        grounding = grounding_from_mask(button_mask, camera, config=component_config)
        if grounding is not None:
            groundings[camera] = grounding
    return groundings


def select_primary_grounding(
    groundings: Mapping[CameraID, Grounding2D],
    *,
    camera_priority: tuple[CameraID, ...] = DEFAULT_CAMERA_PRIORITY,
) -> Grounding2D | None:
    """Select the camera with the largest normalized visible mask area."""
    if not groundings:
        return None
    priority = {camera: rank for rank, camera in enumerate(camera_priority)}
    return max(
        groundings.values(),
        key=lambda grounding: (
            grounding.visible_fraction,
            grounding.visible_pixels,
            -priority.get(grounding.camera, len(priority)),
        ),
    )


def ground_entity_argument(
    argument: EntityArgument,
    segmentations: Mapping[CameraID, np.ndarray],
    resolver: EntityResolver,
    *,
    config: GroundingConfig = GroundingConfig(),
) -> EntityArgument:
    """Attach object and, when resolvable, object-part groundings."""
    raw_object_id = argument.raw_object_id
    if raw_object_id is None:
        return argument
    object_ids = resolver.resolve(raw_object_id)
    groundings = ground_instance_ids(segmentations, object_ids, config=config)

    part = argument.part
    if part is not None:
        part_ids = resolver.resolve_part(raw_object_id, part.name)
        parent_ids = tuple(sorted(set(object_ids).difference(part_ids)))
        is_button = "button" in "".join(part.name.lower().split())
        if is_button and part_ids and parent_ids:
            clean_object_config = dataclasses.replace(
                config,
                min_component_pixels=1,
                min_component_fraction_of_largest=1.0,
            )
            groundings = ground_instance_ids(
                segmentations,
                parent_ids,
                config=clean_object_config,
            )
            part_groundings = ground_button_instance_ids(
                segmentations,
                part_ids,
                parent_ids,
                config=config,
            )
        else:
            part_groundings = ground_instance_ids(
                segmentations, part_ids, config=config
            )
        part = dataclasses.replace(
            part,
            groundings=part_groundings,
        )
    return dataclasses.replace(argument, groundings=groundings, part=part)


def infer_functional_parts(
    control: GroundedControlSpec,
    resolver: EntityResolver,
) -> GroundedControlSpec:
    """Add simulator-verifiable functional parts omitted by B1K annotations.

    BEHAVIOR's button-like annotations name the parent object (for example,
    ``radio_89`` or ``microwave_hjjxmi_0``), while the instance registry may
    expose its toggle button as a separate visual mesh. Only infer a part when
    the registry proves that the candidate exists, so objects without an
    independently segmented button keep their original annotation.
    """
    if control.skill not in _BUTTON_PART_SKILLS:
        return control

    arguments = []
    for argument in control.arguments:
        if (
            argument.role is not Role.TARGET
            or argument.part is not None
            or argument.raw_object_id is None
        ):
            arguments.append(argument)
            continue

        part_name = next(
            (
                candidate
                for candidate in _BUTTON_PART_CANDIDATES
                if resolver.resolve_part(argument.raw_object_id, candidate)
            ),
            None,
        )
        if part_name is None:
            arguments.append(argument)
            continue
        arguments.append(
            dataclasses.replace(argument, part=PartArgument(name=part_name))
        )
    return dataclasses.replace(control, arguments=tuple(arguments))


def ground_control_spec(
    control: GroundedControlSpec,
    segmentations: Mapping[CameraID, np.ndarray],
    resolver: EntityResolver,
    *,
    config: GroundingConfig = GroundingConfig(),
    timestep: int | None = None,
    infer_missing_parts: bool = True,
) -> GroundedControlSpec:
    """Attach per-camera mask groundings to every argument in a control spec."""
    if infer_missing_parts:
        control = infer_functional_parts(control, resolver)
    return dataclasses.replace(
        control,
        arguments=tuple(
            ground_entity_argument(argument, segmentations, resolver, config=config)
            for argument in control.arguments
        ),
        timestep=control.timestep if timestep is None else timestep,
    )
