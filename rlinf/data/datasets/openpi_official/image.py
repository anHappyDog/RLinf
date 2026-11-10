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

"""Image geometry used by the published MolmoBot policy."""

import numpy as np
from PIL import Image


# Crop primitive from allenai/MolmoBot, commit 33c0ca77bf6062a23d60ffd4a6859334c4a46d30,
# molmobot_pi0/eval/utils.py. Keep its integer rounding and PIL interpolation.
def preprocess_molmo_image(image: np.ndarray) -> np.ndarray:
    """Apply the official 360x640 center crop and 224x224 letterboxing."""
    from openpi_client.image_tools import resize_with_pad

    return resize_with_pad(_resize_with_crop(image, 360, 640), 224, 224)


def _resize_with_crop(images: np.ndarray, height: int, width: int) -> np.ndarray:
    original_shape = images.shape
    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack(
        [
            _resize_with_crop_pil(Image.fromarray(im), height, width, Image.BILINEAR)
            for im in images
        ]
    )
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_crop_pil(
    image: Image.Image, height: int, width: int, method: int
) -> Image.Image:
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image
    scale_factor = max(height / cur_height, width / cur_width)
    scaled_h = int(cur_height * scale_factor)
    scaled_w = int(cur_width * scale_factor)
    resized = image.resize((scaled_w, scaled_h), resample=method)
    left = (scaled_w - width) // 2
    top = (scaled_h - height) // 2
    cropped = resized.crop((left, top, left + width, top + height))
    assert cropped.size == (width, height)
    return cropped
