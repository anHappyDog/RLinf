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

import asyncio
import threading
import time
from typing import Any

import ray
from torch.futures import Future

from rlinf.scheduler.collective.async_work import AsyncWork


class AsyncPublishWork(AsyncWork):
    """Completion handle for a trajectory publish operation."""

    def __init__(self, publish_ref: ray.ObjectRef, send_work: AsyncWork):
        """Track the control RPC and P2P payload transfer."""
        super().__init__()
        self._publish_ref = publish_ref
        self._send_work = send_work
        self._completed = False

    async def async_wait(self) -> None:
        """Wait asynchronously for the publish operation."""
        if self._completed:
            return

        results = await asyncio.gather(
            self._send_work.async_wait(), self._publish_ref, return_exceptions=True
        )
        self._completed = True
        for result in results:
            if isinstance(result, BaseException):
                raise result

    def wait(self) -> Any:
        """Wait synchronously for the publish operation."""
        if self._completed:
            return
        publish_done: bool = False
        send_done: bool = False
        errors: list[BaseException] = []
        while not (publish_done and send_done):
            if not publish_done:
                ready, _ = ray.wait([self._publish_ref], timeout=0.01)
                if ready:
                    try:
                        ray.get(self._publish_ref)
                    except BaseException as error:
                        errors.append(error)
                    finally:
                        publish_done = True
            if not send_done and self._send_work.done():
                try:
                    self._send_work.wait()
                except BaseException as error:
                    errors.append(error)
                finally:
                    send_done = True
            if not (publish_done and send_done):
                time.sleep(0.001)
        self._completed = True
        if errors:
            raise errors[0]

    def done(self) -> bool:
        """Return whether the publish operation has completed."""
        if self._completed:
            return True
        ready, _ = ray.wait([self._publish_ref], timeout=0)
        return self._send_work.done() and bool(ready)


class AsyncSubscribeWork(AsyncWork):
    """Completion handle for a trajectory subscription."""

    _data_store: dict[int, Future] = {}
    _store_lock = threading.Lock()

    def __init__(
        self, subscribe_ref: ray.ObjectRef, recv_work: AsyncWork, query_id: int
    ):
        """Track the control RPC and P2P receive operation."""
        self._subscribe_ref = subscribe_ref
        self._recv_work = recv_work
        self._data_future = Future()
        self._query_id = query_id
        self._completed = False
        self._result = None
        with self._store_lock:
            self._data_store[self._query_id] = self._data_future

    async def async_wait(self) -> Any:
        """Wait asynchronously and return the subscribed item."""
        if self._completed:
            return self._result
        try:
            recv_result, _ = await asyncio.gather(
                self._recv_work.async_wait(),
                self._subscribe_ref,
            )
            received_data, query_id = recv_result
            self._handle_received_data(query_id, received_data)
            if not self._data_future.done():
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._data_future.wait)

            self._result = self._data_future.value()
            self._completed = True
            self._data_future = None
            return self._result
        finally:
            self._remove_future()

    def wait(self) -> Any:
        """Wait synchronously and return the subscribed item."""
        if self._completed:
            return self._result
        try:
            received_data, query_id = self._recv_work.wait()
            ray.get(self._subscribe_ref)
            self._handle_received_data(query_id, received_data)
            self._data_future.wait()
            self._result = self._data_future.value()
            self._completed = True
            self._data_future = None
            return self._result
        finally:
            self._remove_future()

    def done(self) -> bool:
        """Return whether the subscribed item is available."""
        if self._completed:
            return True
        ready, _ = ray.wait([self._subscribe_ref], timeout=0)
        return self._recv_work.done() and bool(ready)

    def _handle_received_data(self, query_id: int, data: Any) -> None:
        with self._store_lock:
            future = self._data_store.get(query_id)

        if future is None:
            raise ValueError(f"No future found for query_id {query_id}")
        future.set_result(data)

    def _remove_future(self) -> None:
        with self._store_lock:
            self._data_store.pop(self._query_id, None)
