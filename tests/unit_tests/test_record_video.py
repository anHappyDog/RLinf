import imageio
import numpy as np
from omegaconf import OmegaConf

from rlinf.envs.utils import get_env_attr
from rlinf.envs.wrappers.record_video import RecordVideo


class _DummyEnv:
    seed = 0

    def prepare_outcome_group_reset(self, collection_index):
        return collection_index

    def close(self):
        pass


def _wrapper(tmp_path, **overrides):
    config = {
        "video_base_dir": str(tmp_path),
        "fps": 15,
        "info_on_video": False,
        "include_wrist_views": True,
        "async_save": False,
        "max_pending_videos": 1,
    }
    config.update(overrides)
    return RecordVideo(_DummyEnv(), OmegaConf.create(config))


def test_extracts_head_and_two_wrist_views_as_one_frame(tmp_path):
    wrapper = _wrapper(tmp_path)
    main = np.full((1, 8, 8, 3), 10, dtype=np.uint8)
    wrists = np.stack(
        (
            np.full((4, 4, 3), 20, dtype=np.uint8),
            np.full((4, 4, 3), 30, dtype=np.uint8),
        ),
        axis=0,
    )[None]

    batches = wrapper._extract_frame_batches(
        {"main_images": main, "wrist_images": wrists}
    )
    wrapper.close()

    assert len(batches) == 1
    assert len(batches[0]) == 1
    frame = batches[0][0]
    assert frame.shape == (8, 12, 3)
    assert np.all(frame[:, :8] == 10)
    assert np.all(frame[:4, 8:] == 20)
    assert np.all(frame[4:, 8:] == 30)


def test_main_view_remains_default(tmp_path):
    wrapper = _wrapper(tmp_path, include_wrist_views=False)
    main = np.full((1, 8, 8, 3), 10, dtype=np.uint8)
    wrists = np.zeros((1, 2, 4, 4, 3), dtype=np.uint8)

    batches = wrapper._extract_frame_batches(
        {"main_images": main, "wrist_images": wrists}
    )
    wrapper.close()

    assert batches[0][0].shape == (8, 8, 3)


def test_get_env_attr_reaches_non_gym_base_env(tmp_path):
    wrapper = _wrapper(tmp_path)

    prepare_reset = get_env_attr(wrapper, "prepare_outcome_group_reset")

    assert prepare_reset(7) == 7
    wrapper.close()


def test_async_save_writes_complete_multiview_mp4(tmp_path):
    wrapper = _wrapper(tmp_path, async_save=True)
    main = np.full((1, 8, 8, 3), 10, dtype=np.uint8)
    wrists = np.stack(
        (
            np.full((4, 4, 3), 20, dtype=np.uint8),
            np.full((4, 4, 3), 30, dtype=np.uint8),
        ),
        axis=0,
    )[None]

    wrapper.add_new_frames({"main_images": main, "wrist_images": wrists})
    wrapper.flush_video()
    wrapper.close()

    video_path = tmp_path / "seed_0" / "0.mp4"
    assert video_path.is_file()
    assert not (tmp_path / "seed_0" / "0.mp4.partial.mp4").exists()
    reader = imageio.get_reader(video_path)
    try:
        frame = reader.get_data(0)
    finally:
        reader.close()
    assert frame.shape == (8, 12, 3)
