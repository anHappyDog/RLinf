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

"""Asynchronous, batched TensorBoard events for local KI training metrics."""

import queue
import threading
import time
from pathlib import Path


class AsyncTensorBoard:
    """Queue small metric records and write one Summary event per training step.

    The background thread owns the writer. Records are never dropped, and close
    drains the queue before flushing. Writer failures surface on metric or close.
    """

    def __init__(self, local_dir: str | Path) -> None:
        """Start a writer in the supplied local event directory."""
        self.log_dir = Path(local_dir)
        self._records = queue.SimpleQueue()
        self._closed = False
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def metric(self, record: dict) -> None:
        """Enqueue a scalar training record without waiting for filesystem I/O."""
        if self._closed:
            raise RuntimeError("TensorBoard writer is already closed")
        self._raise_if_failed()
        self._records.put((time.time(), dict(record)))

    def close(self) -> None:
        """Drain queued records, flush events, and close the background writer."""
        if not self._closed:
            self._closed = True
            self._records.put(None)
            self._thread.join()
        self._raise_if_failed()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError(
                "Asynchronous TensorBoard writing failed"
            ) from self._error

    def _run(self) -> None:
        writer = None
        try:
            from tensorboard.compat.proto.event_pb2 import Event
            from tensorboard.compat.proto.summary_pb2 import Summary
            from tensorboard.summary.writer.event_file_writer import EventFileWriter

            writer = EventFileWriter(str(self.log_dir), flush_secs=30)
            while True:
                item = self._records.get()
                if item is None:
                    break
                wall_time, record = item
                domain = record["domain"]
                loss_names = (
                    ("vl_ce_loss",) if domain == "vl" else ("fm_loss", "action_ce_loss")
                )
                tags = {
                    f"train/{name}": record[name]
                    for name in loss_names
                    if name in record
                }
                for name in (
                    "vlm_gradient_norm",
                    "expert_gradient_norm",
                    "vlm_lr",
                    "expert_lr",
                ):
                    if domain == "vl" and name in ("expert_gradient_norm", "expert_lr"):
                        continue
                    if name in record:
                        tags[f"train/{domain}/{name}"] = record[name]
                for name in (
                    "step_seconds",
                    "data_wait_seconds",
                    "checkpoint_seconds",
                    "logging_seconds",
                ):
                    if name in record:
                        tags[f"time/{name}"] = record[name]
                if "end_to_end_updates_per_second" in record:
                    tags["throughput/end_to_end_updates_per_second"] = record[
                        "end_to_end_updates_per_second"
                    ]
                summary = Summary(
                    value=[
                        Summary.Value(tag=tag, simple_value=float(value))
                        for tag, value in tags.items()
                    ]
                )
                writer.add_event(
                    Event(
                        step=int(record["step"]), wall_time=wall_time, summary=summary
                    )
                )
            writer.flush()
        except Exception as error:
            self._error = error
        finally:
            if writer is not None:
                try:
                    writer.close()
                except Exception as error:
                    if self._error is None:
                        self._error = error
