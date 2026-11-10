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

"""SFT scheduling across update and wall-clock boundaries."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import rlinf.runners.sft_runner as sft


class ActorResult:
    def __init__(self, value=None):
        self.value = value

    def wait(self):
        return self.value

    def consume_duration(self):
        return 10.0


class Actor:
    """Synchronous stand-in for the remote actor group."""

    def __init__(self, clock, fail_save=False, save_seconds=0):
        self.clock = clock
        self.steps = 0
        self.saved = []
        self.fail_save = fail_save
        self.save_seconds = save_seconds

    def get_max_steps_per_epoch(self):
        return ActorResult([10])

    def run_training(self):
        self.clock[0] += 10
        self.steps += 1
        return ActorResult([{"loss": 1.0}])

    def save_checkpoint(self, path, step):
        self.saved.append(step)
        if self.fail_save:
            raise OSError("checkpoint write failed")
        self.clock[0] += self.save_seconds
        Path(path, "weights.pt").write_text("complete actor checkpoint")
        return ActorResult()


@pytest.fixture
def runner_factory(tmp_path, monkeypatch):
    clock = [0.0]
    start = datetime(2026, 11, 9, 23, 19, tzinfo=timezone.utc)

    class ClockDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return (start + timedelta(seconds=clock[0])).astimezone(tz)

    monkeypatch.setattr(sft, "datetime", ClockDatetime)
    monkeypatch.setattr(sft.time, "monotonic", lambda: clock[0])

    def make(*, stop_seconds=None, fail_save=False, save_seconds=0, **settings):
        config = OmegaConf.create(
            {
                "runner": {
                    "max_epochs": -1,
                    "max_steps": 5,
                    "val_check_interval": -1,
                    "save_interval": -1,
                    "save_interval_seconds": 20,
                    "stop_at": (start + timedelta(seconds=stop_seconds)).isoformat()
                    if stop_seconds is not None
                    else None,
                    "logger": {
                        "log_path": str(tmp_path),
                        "experiment_name": "test",
                        "logger_backends": [],
                    },
                    **settings,
                }
            }
        )
        actor = Actor(clock, fail_save=fail_save, save_seconds=save_seconds)
        return sft.SFTRunner(config, actor), actor

    return make


def test_sft_periodic_checkpoint_marks_completed_steps(runner_factory, tmp_path):
    runner, actor = runner_factory()
    runner.run()
    assert actor.steps == 5
    assert actor.saved == [2, 4]
    for step in actor.saved:
        path = tmp_path / "test" / "checkpoints" / f"global_step_{step}"
        marker = json.loads((path / "CHECKPOINT_COMPLETE.json").read_text())
        assert marker["global_step"] == step
        assert Path(marker["actor_path"], "weights.pt").is_file()


def test_sft_deadline_coalesces_step_and_timed_saves(runner_factory):
    runner, actor = runner_factory(stop_seconds=20, save_interval=2)
    runner.run()
    assert actor.steps == 2
    assert actor.saved == [2]


def test_sft_deadline_during_save_does_not_save_same_step_twice(runner_factory):
    runner, actor = runner_factory(stop_seconds=25, save_seconds=10)
    runner.run()
    assert actor.steps == 2
    assert actor.saved == [2]


def test_sft_failed_save_has_no_completion_marker(runner_factory, tmp_path):
    runner, _ = runner_factory(stop_seconds=10, fail_save=True)
    with pytest.raises(OSError, match="checkpoint write failed"):
        runner.run()
    assert not list(tmp_path.rglob("CHECKPOINT_COMPLETE.json"))


def test_sft_legacy_step_schedule_is_unchanged(runner_factory):
    runner, actor = runner_factory(save_interval_seconds=None, save_interval=2)
    runner.run()
    assert actor.saved == [2, 4, 5]
