# Copyright 2025 The RLinf Authors.
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

import numbers
import os
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Optional

import gymnasium as gym
import imageio
import numpy as np
from PIL import Image

try:
    import torch
except ImportError:
    torch = None

from rlinf.envs.utils import put_info_on_image, tile_images


class RecordVideo(gym.Wrapper):
    """
    A general video recording wrapper that owns the recording logic.

    ``RecordVideo`` centralizes frame collection and MP4 writing for both regular
    stepping and chunked stepping APIs. Frames are buffered in memory and flushed
    asynchronously to avoid blocking environment interaction.

    The wrapper supports multiple observation image layouts (single frame, batched
    frames, and temporal batches). For ``chunk_step()``, it correctly handles the
    terminal-to-reset transition by recording terminal observations (for the last
    step in the chunk) and then appending the corresponding reset observations.

    When ``video_cfg.info_on_video`` is enabled, per-frame text metadata is drawn
    through ``put_info_on_image()``. The overlay always includes reward and
    termination when available, and can include extra fields from environment
    ``info`` via ``video_cfg.extra_info_on_video``. Nested keys are supported with
    dot notation, for example
    ``["env_id", "episode.success_once", "episode.episode_len"]``.

    Args:
        env: Wrapped environment. It must expose a ``seed`` attribute and may
            optionally provide ``num_envs`` and metadata for FPS inference.
        video_cfg: Video configuration object/dict. Common fields:
            ``video_base_dir`` (output directory root),
            ``fps`` (optional FPS override),
            ``info_on_video`` (whether to render overlay text),
            ``extra_info_on_video`` (list of ``info`` keys to render),
            ``include_wrist_views`` (compose left and right wrist views beside
            the main image), ``async_save`` (do not wait after every encode),
            and ``max_pending_videos`` (bound asynchronous encode memory).
        fps: Explicit FPS override. If ``None``, FPS is resolved from
            ``video_cfg.fps``, environment config/metadata, then fallback ``30``.
    """

    def __init__(self, env: gym.Env, video_cfg, fps: Optional[int] = None):
        """Initialize the wrapper and set FPS/config."""
        if isinstance(env, gym.Env):
            super().__init__(env)
        else:
            self.env = env

        if not hasattr(env, "seed"):
            raise AttributeError("Environment must have 'seed' attribute")

        self.video_cfg = video_cfg
        self.render_images: list[np.ndarray] = []
        self.video_cnt = 0
        self._num_envs = getattr(env, "num_envs", 1)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._save_futures: list[Future] = []
        self._include_wrist_views = bool(
            self.video_cfg.get("include_wrist_views", False)
        )
        self._async_save = bool(self.video_cfg.get("async_save", False))
        self._max_pending_videos = int(self.video_cfg.get("max_pending_videos", 1))
        if self._max_pending_videos <= 0:
            raise ValueError("video_cfg.max_pending_videos must be positive.")

        if fps is not None:
            self._fps = fps
        else:
            self._fps = self._get_fps_from_env(env)

    @property
    def is_start(self):
        return getattr(self.env, "is_start")

    @is_start.setter
    def is_start(self, value):
        setattr(self.env, "is_start", value)

    def _get_fps_from_env(self, env: gym.Env) -> int:
        """Resolve FPS from config/env metadata with fallback."""
        if hasattr(self.video_cfg, "fps") and self.video_cfg.fps is not None:
            return int(self.video_cfg.fps)
        if hasattr(env, "cfg") and hasattr(env.cfg, "init_params"):
            if hasattr(env.cfg.init_params, "sim_config"):
                if hasattr(env.cfg.init_params.sim_config, "control_freq"):
                    return int(env.cfg.init_params.sim_config.control_freq)
        metadata = getattr(env, "metadata", None)
        if isinstance(metadata, dict) and "render_fps" in metadata:
            return int(metadata["render_fps"])
        return 30

    def _to_numpy(self, value: Any) -> np.ndarray:
        """Convert tensors/arrays to numpy."""
        if torch is not None and isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value
        return np.array(value)

    def _get_image_from_dict(self, obs: dict) -> Optional[Any]:
        """Pick the best image field from an observation dict."""
        for key in ("main_images", "images", "rgb", "full_image", "main_image"):
            if key in obs and obs[key] is not None:
                return obs[key]
        if hasattr(self.env, "capture_image"):
            return self.env.capture_image()
        return None

    @staticmethod
    def _as_uint8_hwc(image: Any) -> np.ndarray:
        """Normalize one image to an uint8 HWC array."""
        if torch is not None and isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        image = np.asarray(image)
        if image.ndim != 3:
            raise ValueError(f"Expected one HWC image, got shape {image.shape}.")
        if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
            image = np.transpose(image, (1, 2, 0))
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        return image[..., :3]

    @staticmethod
    def _resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
        """Resize an image for video composition."""
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        return np.asarray(Image.fromarray(image).resize((width, height), resampling))

    @classmethod
    def _compose_robot_views(
        cls,
        main_image: Any,
        left_wrist_image: Any,
        right_wrist_image: Any,
    ) -> np.ndarray:
        """Place two wrist views in a column beside the main camera."""
        main = cls._as_uint8_hwc(main_image)
        left = cls._as_uint8_hwc(left_wrist_image)
        right = cls._as_uint8_hwc(right_wrist_image)
        main_height, main_width = main.shape[:2]
        wrist_width = max(main_width // 2, 1)
        top_height = max(main_height // 2, 1)
        bottom_height = main_height - top_height
        left = cls._resize_image(left, wrist_width, top_height)
        right = cls._resize_image(right, wrist_width, bottom_height)
        return np.concatenate((main, np.concatenate((left, right), axis=0)), axis=1)

    def _extract_robot_view_batches(self, obs: dict) -> list[list[np.ndarray]]:
        """Extract per-env composites when main and wrist views are available."""
        if not self._include_wrist_views or "wrist_images" not in obs:
            return []

        main = self._to_numpy(obs["main_images"])
        wrists = self._to_numpy(obs["wrist_images"])
        if main.ndim == 3:
            main = main[None]
        if wrists.ndim == 4:
            wrists = wrists[None]
        if main.ndim != 4 or wrists.ndim != 5:
            raise ValueError(
                "Multi-view video expects main_images [N,H,W,C] and "
                f"wrist_images [N,2,H,W,C], got {main.shape} and {wrists.shape}."
            )
        if main.shape[0] != wrists.shape[0] or wrists.shape[1] != 2:
            raise ValueError(
                "Main/wrist video batches must have matching environments and "
                f"exactly two wrist views, got {main.shape} and {wrists.shape}."
            )
        return [
            [
                self._compose_robot_views(
                    main[env_id], wrists[env_id, 0], wrists[env_id, 1]
                )
                for env_id in range(main.shape[0])
            ]
        ]

    def _extract_frame_batches(self, obs: Any) -> list[list[np.ndarray]]:
        """Extract a list of per-step image batches from obs."""
        if obs is None:
            return []

        if isinstance(obs, dict):
            robot_views = self._extract_robot_view_batches(obs)
            if robot_views:
                return robot_views
            image_src = self._get_image_from_dict(obs)
            if image_src is None:
                return []
            return self._split_image_source(image_src)

        if isinstance(obs, (list, tuple)):
            if len(obs) == 0:
                return []
            if isinstance(obs[0], dict):
                frames = []
                for item in obs:
                    robot_views = self._extract_robot_view_batches(item)
                    if robot_views:
                        frames.extend(robot_views)
                        continue
                    image_src = self._get_image_from_dict(item)
                    if image_src is None:
                        continue
                    batches = self._split_image_source(image_src)
                    if batches:
                        frames.append(batches[0])
                return frames
            images = []
            for item in obs:
                img = self._to_numpy(item)
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                images.append(img)
            return [images] if images else []

        if torch is not None and isinstance(obs, torch.Tensor):
            return self._split_image_source(obs)
        if isinstance(obs, np.ndarray):
            return self._split_image_source(obs)
        return []

    def _split_image_source(self, image_src: Any) -> list[list[np.ndarray]]:
        """Normalize common image tensor layouts into frame batches."""
        img = self._to_numpy(image_src)

        if img.ndim == 3:
            if img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
                img = np.transpose(img, (1, 2, 0))
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            return [[img]]

        if img.ndim == 4:
            if img.shape[1] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
                img = np.transpose(img, (0, 2, 3, 1))
            images = []
            for i in range(img.shape[0]):
                single = img[i]
                if single.dtype != np.uint8:
                    single = single.astype(np.uint8)
                images.append(single)
            return [images]

        if img.ndim == 5:
            if img.shape[2] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
                img = np.transpose(img, (0, 1, 3, 4, 2))
            frames = []
            for t in range(img.shape[1]):
                images = []
                for i in range(img.shape[0]):
                    single = img[i, t]
                    if single.dtype != np.uint8:
                        single = single.astype(np.uint8)
                    images.append(single)
                frames.append(images)
            return frames

        return []

    def _value_for_env(self, value: Any, env_id: int):
        """Select a scalar/value for a specific env from batched inputs."""
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            if value.shape == ():
                return value.item()
            if value.size == 1:
                return value.reshape(-1)[0].item()
            if value.shape[0] > env_id:
                return value[env_id]
            return value.reshape(-1)[0]
        if isinstance(value, (list, tuple)):
            if len(value) > env_id:
                return value[env_id]
            if len(value) > 0:
                return value[0]
        return value

    def _get_task_description(self, obs: Any, env_id: int):
        """Get task description from obs or env attribute."""
        if isinstance(obs, dict) and "task_descriptions" in obs:
            task_desc = obs["task_descriptions"]
            if isinstance(task_desc, (list, tuple)) and len(task_desc) > env_id:
                return task_desc[env_id]
            return task_desc[0] if isinstance(task_desc, (list, tuple)) else task_desc
        if hasattr(self.env, "task_descriptions"):
            task_desc = self.env.task_descriptions
            if isinstance(task_desc, (list, tuple)) and len(task_desc) > env_id:
                return task_desc[env_id]
            return task_desc[0] if isinstance(task_desc, (list, tuple)) else task_desc
        return None

    def _get_video_info_keys(self) -> list[str]:
        """Get configured info keys to overlay on video frames."""
        if hasattr(self.video_cfg, "extra_info_on_video"):
            keys = getattr(self.video_cfg, "extra_info_on_video")
        else:
            keys = None

        if keys:
            if isinstance(keys, str):
                return [keys]
            return list(keys)
        return []

    def _lookup_info_value(self, info: Any, key: str) -> Any:
        """Read a key from info, supporting dotted access for nested dicts."""
        if not isinstance(info, dict):
            return None
        if key in info:
            return info[key]

        value = info
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                return None
            value = value[part]
        return value

    def _build_info_item(
        self,
        infos: Optional[Any],
        rewards: Optional[Any],
        terminations: Optional[Any],
        env_id: int,
        time_idx: Optional[int] = None,
    ) -> dict:
        """Build a per-env info dict for overlay."""
        info_item: dict[str, Any] = {}

        if rewards is not None:
            value = self._value_for_env(rewards, env_id)
            if time_idx is not None and isinstance(value, (np.ndarray, list, tuple)):
                if len(value) > time_idx:
                    value = value[time_idx]
            info_item["reward"] = float(value) if value is not None else value

        if terminations is not None:
            value = self._value_for_env(terminations, env_id)
            if time_idx is not None and isinstance(value, (np.ndarray, list, tuple)):
                if len(value) > time_idx:
                    value = value[time_idx]
            info_item["termination"] = bool(value) if value is not None else value

        if infos is not None:
            for key in self._get_video_info_keys():
                value = self._lookup_info_value(infos, key)
                if value is None:
                    continue
                value = self._value_for_env(value, env_id)
                if isinstance(value, np.ndarray):
                    if value.shape == ():
                        value = value.item()
                    elif value.size == 1:
                        value = value.reshape(-1)[0].item()
                elif isinstance(value, numbers.Number):
                    pass
                else:
                    warnings.warn(f"Unsupported value type {type(value)} for key {key}")
                    continue
                info_item[key] = value

        return info_item

    def _append_frame(
        self,
        images: list[np.ndarray],
        infos: Optional[Any],
        rewards: Optional[Any],
        terminations: Optional[Any],
        time_idx: Optional[int] = None,
    ) -> None:
        """Overlay info (optional) and append a tiled frame."""
        if not images:
            return
        if self.video_cfg.get("info_on_video", True):
            images = [
                put_info_on_image(
                    img,
                    self._build_info_item(
                        infos, rewards, terminations, env_id, time_idx
                    ),
                )
                for env_id, img in enumerate(images)
            ]
        if len(images) > 1:
            nrows = int(np.sqrt(len(images)))
            full_image = tile_images(images, nrows=nrows)
            self.render_images.append(full_image)
        else:
            self.render_images.append(images[0])

    def add_new_frames(
        self,
        obs: Any,
        infos: Optional[Any] = None,
        rewards: Optional[Any] = None,
        terminations: Optional[Any] = None,
    ):
        """Extract frames from obs and append to the buffer."""
        frames = self._extract_frame_batches(obs)
        if not frames:
            warnings.warn(
                f"Failed to extract images from obs, obs type: {type(obs)}, obs keys: "
                f"{list(obs.keys()) if isinstance(obs, dict) else 'N/A'}"
            )
            return

        if isinstance(infos, (list, tuple)):
            for time_idx, images in enumerate(frames):
                step_info = infos[time_idx] if len(infos) > time_idx else None
                self._append_frame(images, step_info, rewards, terminations, time_idx)
            return

        for time_idx, images in enumerate(frames):
            self._append_frame(images, infos, rewards, terminations, time_idx)

    def reset(self, *args, **kwargs):
        """Reset env and record the initial frame."""
        obs, info = self.env.reset(*args, **kwargs)
        self.add_new_frames(obs, info)
        return obs, info

    def step(self, action):
        """Step env and record the resulting frame."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        terminations = (
            info.get("terminations", terminated)
            if isinstance(info, dict)
            else terminated
        )
        self.add_new_frames(obs, info, reward, terminations)
        return obs, reward, terminated, truncated, info

    def record_video_in_result(self, result) -> None:
        """Record video frames from a chunk_step / async_chunk_step result tuple."""
        if isinstance(result, tuple) and len(result) >= 5:
            obs_list, rewards, terminations, _truncations, infos_list = result[:5]

            # Some envs may skip intermediate observations for performance and return
            # None entries. Filter them out for video collection.
            if isinstance(obs_list, (list, tuple)):
                valid_indices = [i for i, obs in enumerate(obs_list) if obs is not None]
                if len(valid_indices) == 0:
                    return
                if len(valid_indices) != len(obs_list):
                    obs_list = [obs_list[i] for i in valid_indices]
                    if isinstance(infos_list, (list, tuple)):
                        infos_list = [infos_list[i] for i in valid_indices]
                    if (
                        torch is not None
                        and isinstance(rewards, torch.Tensor)
                        and rewards.ndim == 2
                    ):
                        rewards = rewards[:, valid_indices]
                    if (
                        torch is not None
                        and isinstance(terminations, torch.Tensor)
                        and terminations.ndim == 2
                    ):
                        terminations = terminations[:, valid_indices]

            final_obs = None
            last_info = None
            if isinstance(infos_list, (list, tuple)) and len(infos_list) > 0:
                last_info = infos_list[-1]
                if isinstance(last_info, dict):
                    if last_info.get("final_obs") is not None:
                        final_obs = last_info["final_obs"]
                    elif last_info.get("final_observation") is not None:
                        final_obs = last_info["final_observation"]

            if (
                final_obs is not None
                and isinstance(obs_list, (list, tuple))
                and len(obs_list) > 0
            ):
                reset_obs = obs_list[-1]
                obs_main = list(obs_list)
                obs_main[-1] = final_obs
                infos_main = (
                    list(infos_list)
                    if isinstance(infos_list, (list, tuple))
                    else infos_list
                )
                self.add_new_frames(obs_main, infos_main, rewards, terminations)
                self.add_new_frames(reset_obs, None)
            else:
                self.add_new_frames(obs_list, infos_list, rewards, terminations)

    def chunk_step(self, *args, **kwargs):
        """Step a chunk and record all frames from the chunk."""
        result = self.env.chunk_step(*args, **kwargs)
        self.record_video_in_result(result)
        return result

    def flush_video(self, video_sub_dir: Optional[str] = None):
        """Write buffered frames to an MP4 file.

        With ``async_save`` disabled, wait for the just-submitted encode as in
        the historical implementation. Asynchronous mode keeps a bounded queue;
        :meth:`close` must be called during orderly worker shutdown to drain it.
        """
        if not self.render_images:
            return

        output_dir = os.path.join(
            self.video_cfg.video_base_dir, f"seed_{self.env.seed}"
        )
        if video_sub_dir is not None:
            output_dir = os.path.join(output_dir, f"{video_sub_dir}")

        os.makedirs(output_dir, exist_ok=True)
        mp4_path = os.path.join(output_dir, f"{self.video_cnt}.mp4")
        frames = list(self.render_images)
        self.render_images = []
        self.video_cnt += 1
        future = self._submit_save(frames, mp4_path)
        if self._async_save:
            self._bound_pending_saves()
        else:
            future.result()

    def _submit_save(self, frames: list[np.ndarray], mp4_path: str) -> Future:
        """Submit a background job to save the video, return its Future."""
        self._prune_futures()
        future = self._executor.submit(self._save_video, frames, mp4_path)
        self._save_futures.append(future)
        return future

    def _save_video(self, frames: list[np.ndarray], mp4_path: str) -> None:
        """Save frames to disk (runs in background)."""
        video_writer = None
        temporary_path = f"{mp4_path}.partial.mp4"
        try:
            video_writer = imageio.get_writer(
                temporary_path,
                fps=self._fps,
                macro_block_size=1,
            )
            for img in frames:
                video_writer.append_data(img)
            video_writer.close()
            video_writer = None
            os.replace(temporary_path, mp4_path)
        except Exception as exc:
            warnings.warn(f"Failed to save video {mp4_path}: {exc}")
        finally:
            if video_writer is not None:
                video_writer.close()

    def _bound_pending_saves(self) -> None:
        """Apply backpressure before pending frame buffers grow without bound."""
        self._prune_futures()
        while len(self._save_futures) > self._max_pending_videos:
            self._save_futures.pop(0).result()

    def _prune_futures(self) -> None:
        """Remove finished futures to avoid unbounded growth."""
        self._save_futures = [f for f in self._save_futures if not f.done()]

    def close(self):
        """Wait for pending video writes before closing."""
        self._executor.shutdown(wait=True)
        self._save_futures = []
        return super().close()

    def update_reset_state_ids(self):
        if hasattr(self.env, "update_reset_state_ids"):
            self.env.update_reset_state_ids()
