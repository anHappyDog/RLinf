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

"""Reliable RPC transport for environments outside the Ray cluster.

The transport is deliberately small. It sends a safe, length-prefixed tree
encoding over TCP, supports a persistent SSH local-forward, and assigns every
state-mutating request a monotonically increasing id. The server retains the
last encoded response, so reconnecting after an ambiguous network failure does
not execute an action chunk twice.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import socket
import struct
import subprocess
import threading
import time
import traceback
import uuid
import zlib
from collections.abc import Callable
from math import prod
from typing import Any

import numpy as np
import torch

_PROTOCOL_VERSION = 1
_FORMAT_NAME = "rlinf-safe-tree-v1"
_LENGTH = struct.Struct("!Q")
_DEFAULT_MAX_MESSAGE_BYTES = 512 * 1024 * 1024
_DEFAULT_COMPRESSION_MIN_BYTES = 64 * 1024
_SUPPORTED_BLOB_COMPRESSION = ("none", "zlib")

_TORCH_DTYPES = {
    str(dtype).removeprefix("torch."): dtype
    for dtype in (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    )
}


class RemoteCollectorError(RuntimeError):
    """Base exception raised by the remote environment transport."""


class RemoteCollectorProtocolError(RemoteCollectorError):
    """The peer sent an invalid or incompatible protocol message."""


class RemoteCollectorConnectionError(RemoteCollectorError):
    """A collector transport could not be established."""


class RemoteCollectorStateLostError(RemoteCollectorError):
    """The remote daemon restarted and no longer owns the simulator state."""


class _TreeEncoder:
    def __init__(
        self,
        *,
        compression: str,
        compression_level: int,
        compression_min_bytes: int,
    ) -> None:
        self.blobs: list[bytes] = []
        self.blob_codecs: list[str] = []
        self.blob_raw_lengths: list[int] = []
        self.compression = compression
        self.compression_level = compression_level
        self.compression_min_bytes = compression_min_bytes

    def encode(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float, str)):
            return {"type": "scalar", "value": value}
        if isinstance(value, np.generic):
            return self.encode(value.item())
        if isinstance(value, bytes):
            return self._append_blob("bytes", value)
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            dtype_name = str(tensor.dtype).removeprefix("torch.")
            if dtype_name not in _TORCH_DTYPES:
                raise TypeError(f"Unsupported torch dtype: {tensor.dtype}.")
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            return self._append_blob(
                "torch",
                raw,
                dtype=dtype_name,
                shape=list(tensor.shape),
            )
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise TypeError("Object-dtype numpy arrays are not supported.")
            array = np.ascontiguousarray(value)
            return self._append_blob(
                "numpy",
                array.tobytes(),
                dtype=array.dtype.str,
                shape=list(array.shape),
            )
        if isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                raise TypeError("Remote collector dictionaries require string keys.")
            return {
                "type": "dict",
                "items": [[key, self.encode(item)] for key, item in value.items()],
            }
        if isinstance(value, list):
            return {"type": "list", "items": [self.encode(item) for item in value]}
        if isinstance(value, tuple):
            return {"type": "tuple", "items": [self.encode(item) for item in value]}
        raise TypeError(f"Unsupported remote collector value: {type(value).__name__}.")

    def _append_blob(self, node_type: str, value: bytes, **metadata: Any) -> dict:
        blob_index = len(self.blobs)
        raw_length = len(value)
        codec = "none"
        encoded = value
        if self.compression == "zlib" and raw_length >= self.compression_min_bytes:
            candidate = zlib.compress(value, self.compression_level)
            if len(candidate) < raw_length:
                codec = "zlib"
                encoded = candidate
        self.blobs.append(encoded)
        self.blob_codecs.append(codec)
        self.blob_raw_lengths.append(raw_length)
        return {"type": node_type, "blob": blob_index, **metadata}


def _decode_tree(node: Any, blobs: list[memoryview]) -> Any:
    if not isinstance(node, dict) or "type" not in node:
        raise RemoteCollectorProtocolError("Malformed tree node.")
    node_type = node["type"]
    if node_type == "scalar":
        return node.get("value")
    if node_type == "bytes":
        return bytes(blobs[_blob_index(node, blobs)])
    if node_type == "torch":
        dtype_name = node.get("dtype")
        if dtype_name not in _TORCH_DTYPES:
            raise RemoteCollectorProtocolError(
                f"Unsupported torch dtype {dtype_name!r}."
            )
        shape = _decode_shape(node)
        if prod(shape) == 0:
            return torch.empty(shape, dtype=_TORCH_DTYPES[dtype_name])
        raw = bytearray(blobs[_blob_index(node, blobs)])
        tensor = torch.frombuffer(raw, dtype=_TORCH_DTYPES[dtype_name]).clone()
        return tensor.reshape(shape)
    if node_type == "numpy":
        try:
            dtype = np.dtype(node["dtype"])
        except (KeyError, TypeError) as exc:
            raise RemoteCollectorProtocolError("Invalid numpy dtype.") from exc
        if dtype.hasobject:
            raise RemoteCollectorProtocolError("Object-dtype arrays are forbidden.")
        array = np.frombuffer(blobs[_blob_index(node, blobs)], dtype=dtype).copy()
        return array.reshape(_decode_shape(node))
    if node_type in ("list", "tuple"):
        items = node.get("items")
        if not isinstance(items, list):
            raise RemoteCollectorProtocolError("Sequence node is missing items.")
        decoded = [_decode_tree(item, blobs) for item in items]
        return tuple(decoded) if node_type == "tuple" else decoded
    if node_type == "dict":
        items = node.get("items")
        if not isinstance(items, list):
            raise RemoteCollectorProtocolError("Dictionary node is missing items.")
        result = {}
        for pair in items:
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not isinstance(pair[0], str)
            ):
                raise RemoteCollectorProtocolError("Invalid dictionary entry.")
            result[pair[0]] = _decode_tree(pair[1], blobs)
        return result
    raise RemoteCollectorProtocolError(f"Unknown tree node type {node_type!r}.")


def _blob_index(node: dict, blobs: list[memoryview]) -> int:
    try:
        index = int(node["blob"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RemoteCollectorProtocolError("Invalid blob index.") from exc
    if not 0 <= index < len(blobs):
        raise RemoteCollectorProtocolError(f"Blob index {index} is out of range.")
    return index


def _decode_shape(node: dict) -> tuple[int, ...]:
    shape = node.get("shape")
    if not isinstance(shape, list) or not all(
        isinstance(dimension, int) and dimension >= 0 for dimension in shape
    ):
        raise RemoteCollectorProtocolError("Invalid tensor shape.")
    return tuple(shape)


def encode_message(
    value: Any,
    *,
    compression: str = "none",
    compression_level: int = 1,
    compression_min_bytes: int = _DEFAULT_COMPRESSION_MIN_BYTES,
) -> bytes:
    """Encode a nested RPC value without pickle or executable objects.

    Large binary blobs may be compressed independently with zlib. Compression
    is lossless and self-describing, so :func:`decode_message` also accepts the
    original uncompressed wire format.
    """
    if compression not in _SUPPORTED_BLOB_COMPRESSION:
        raise ValueError(
            f"Unsupported blob compression {compression!r}; expected one of "
            f"{_SUPPORTED_BLOB_COMPRESSION}."
        )
    if not 0 <= compression_level <= 9:
        raise ValueError("compression_level must be in [0, 9].")
    if compression_min_bytes < 0:
        raise ValueError("compression_min_bytes must be non-negative.")
    encoder = _TreeEncoder(
        compression=compression,
        compression_level=compression_level,
        compression_min_bytes=compression_min_bytes,
    )
    tree = encoder.encode(value)
    header = json.dumps(
        {
            "format": _FORMAT_NAME,
            "tree": tree,
            "blob_lengths": [len(blob) for blob in encoder.blobs],
            "blob_codecs": encoder.blob_codecs,
            "blob_raw_lengths": encoder.blob_raw_lengths,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return _LENGTH.pack(len(header)) + header + b"".join(encoder.blobs)


def decode_message(payload: bytes) -> Any:
    """Decode a value emitted by :func:`encode_message`."""
    if len(payload) < _LENGTH.size:
        raise RemoteCollectorProtocolError("Message is shorter than its header.")
    (header_length,) = _LENGTH.unpack_from(payload)
    header_end = _LENGTH.size + header_length
    if header_end > len(payload):
        raise RemoteCollectorProtocolError("Truncated message header.")
    try:
        header = json.loads(payload[_LENGTH.size : header_end])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RemoteCollectorProtocolError("Invalid message header JSON.") from exc
    if header.get("format") != _FORMAT_NAME:
        raise RemoteCollectorProtocolError("Unsupported tree encoding format.")
    lengths = header.get("blob_lengths")
    if not isinstance(lengths, list) or not all(
        isinstance(length, int) and length >= 0 for length in lengths
    ):
        raise RemoteCollectorProtocolError("Invalid blob lengths.")
    codecs = header.get("blob_codecs", ["none"] * len(lengths))
    raw_lengths = header.get("blob_raw_lengths", lengths)
    if (
        not isinstance(codecs, list)
        or len(codecs) != len(lengths)
        or any(codec not in _SUPPORTED_BLOB_COMPRESSION for codec in codecs)
    ):
        raise RemoteCollectorProtocolError("Invalid blob compression codecs.")
    if (
        not isinstance(raw_lengths, list)
        or len(raw_lengths) != len(lengths)
        or not all(isinstance(length, int) and length >= 0 for length in raw_lengths)
    ):
        raise RemoteCollectorProtocolError("Invalid raw blob lengths.")
    if sum(raw_lengths) > _DEFAULT_MAX_MESSAGE_BYTES:
        raise RemoteCollectorProtocolError(
            "Decoded blob payload exceeds the maximum message size."
        )
    blobs = []
    offset = header_end
    view = memoryview(payload)
    for length, codec, raw_length in zip(lengths, codecs, raw_lengths, strict=True):
        next_offset = offset + length
        if next_offset > len(payload):
            raise RemoteCollectorProtocolError("Truncated message blob.")
        encoded_blob = view[offset:next_offset]
        if codec == "zlib":
            try:
                decompressor = zlib.decompressobj()
                blob = decompressor.decompress(encoded_blob, raw_length + 1)
            except zlib.error as exc:
                raise RemoteCollectorProtocolError(
                    "Invalid zlib-compressed message blob."
                ) from exc
            if (
                len(blob) != raw_length
                or not decompressor.eof
                or decompressor.unused_data
                or decompressor.unconsumed_tail
            ):
                raise RemoteCollectorProtocolError(
                    "Decompressed blob length does not match its metadata."
                )
            blobs.append(memoryview(blob))
        else:
            if length != raw_length:
                raise RemoteCollectorProtocolError(
                    "Uncompressed blob length does not match its metadata."
                )
            blobs.append(encoded_blob)
        offset = next_offset
    if offset != len(payload):
        raise RemoteCollectorProtocolError("Message contains trailing bytes.")
    return _decode_tree(header.get("tree"), blobs)


def _message_blob_sizes(payload: bytes) -> tuple[int, int]:
    """Return encoded and original blob bytes from one safe-tree message."""
    if len(payload) < _LENGTH.size:
        return 0, 0
    (header_length,) = _LENGTH.unpack_from(payload)
    header_end = _LENGTH.size + header_length
    if header_end > len(payload):
        return 0, 0
    try:
        header = json.loads(payload[_LENGTH.size : header_end])
    except (UnicodeDecodeError, json.JSONDecodeError):
        return 0, 0
    lengths = header.get("blob_lengths", [])
    raw_lengths = header.get("blob_raw_lengths", lengths)
    if not isinstance(lengths, list) or not isinstance(raw_lengths, list):
        return 0, 0
    return sum(lengths), sum(raw_lengths)


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        part = connection.recv(size - len(chunks))
        if not part:
            raise EOFError("Remote collector connection closed.")
        chunks.extend(part)
    return bytes(chunks)


def recv_frame(
    connection: socket.socket,
    *,
    max_message_bytes: int = _DEFAULT_MAX_MESSAGE_BYTES,
) -> bytes:
    """Read one encoded message payload from a stream socket."""
    (payload_size,) = _LENGTH.unpack(_recv_exact(connection, _LENGTH.size))
    if payload_size > max_message_bytes:
        raise RemoteCollectorProtocolError(
            f"Message size {payload_size} exceeds limit {max_message_bytes}."
        )
    return _recv_exact(connection, payload_size)


def send_frame(connection: socket.socket, payload: bytes) -> None:
    """Write one encoded message payload to a stream socket."""
    connection.sendall(_LENGTH.pack(len(payload)) + payload)


class SshLocalForward:
    """Own a reconnectable SSH local-forward to one collector daemon."""

    def __init__(
        self,
        ssh_host: str,
        remote_port: int,
        *,
        local_host: str = "127.0.0.1",
        local_port: int | None = None,
        connect_timeout: float = 30.0,
        server_alive_interval: int = 15,
        server_alive_count_max: int = 3,
        compression: bool = True,
        extra_options: list[str] | None = None,
    ) -> None:
        self.ssh_host = ssh_host
        self.remote_port = int(remote_port)
        self.local_host = local_host
        self.local_port = local_port or self._reserve_local_port(local_host)
        self.connect_timeout = float(connect_timeout)
        self.server_alive_interval = int(server_alive_interval)
        self.server_alive_count_max = int(server_alive_count_max)
        self.compression = bool(compression)
        self.extra_options = list(extra_options or [])
        self._process: subprocess.Popen | None = None

    @staticmethod
    def _reserve_local_port(host: str) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind((host, 0))
            return int(listener.getsockname()[1])

    def start(self) -> None:
        """Start the tunnel and wait until its local listener is reachable."""
        if self._process is not None and self._process.poll() is None:
            return
        command = [
            "ssh",
            "-N",
            *(["-C"] if self.compression else []),
            "-o",
            "BatchMode=yes",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            f"ConnectTimeout={max(1, int(self.connect_timeout))}",
            "-o",
            f"ServerAliveInterval={self.server_alive_interval}",
            "-o",
            f"ServerAliveCountMax={self.server_alive_count_max}",
            *self.extra_options,
            "-L",
            f"{self.local_host}:{self.local_port}:127.0.0.1:{self.remote_port}",
            self.ssh_host,
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + self.connect_timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                error = self._process.stderr.read().strip()
                raise RemoteCollectorConnectionError(
                    f"SSH tunnel to {self.ssh_host} exited early: {error}"
                )
            try:
                with socket.create_connection(
                    (self.local_host, self.local_port), timeout=0.2
                ):
                    return
            except OSError:
                time.sleep(0.05)
        self.close()
        raise TimeoutError(
            f"Timed out opening SSH tunnel to {self.ssh_host}:{self.remote_port}."
        )

    def restart(self) -> None:
        """Replace the current tunnel while preserving the local endpoint."""
        self.close()
        self.start()

    def close(self) -> None:
        """Stop the SSH subprocess without affecting the remote daemon."""
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


class RemoteCollectorClient:
    """Sequential, at-most-once RPC client for one remote environment."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        auth_token: str,
        request_timeout: float = 600.0,
        connect_timeout: float = 30.0,
        reconnect_attempts: int = 3,
        tunnel: SshLocalForward | None = None,
        response_compression: str = "none",
        response_compression_level: int = 1,
        response_compression_min_bytes: int = _DEFAULT_COMPRESSION_MIN_BYTES,
    ) -> None:
        if not auth_token:
            raise ValueError("Remote collector auth token must not be empty.")
        if reconnect_attempts < 0:
            raise ValueError("reconnect_attempts must be non-negative.")
        if response_compression not in _SUPPORTED_BLOB_COMPRESSION:
            raise ValueError(
                f"Unsupported response compression {response_compression!r}; "
                f"expected one of {_SUPPORTED_BLOB_COMPRESSION}."
            )
        if not 0 <= response_compression_level <= 9:
            raise ValueError("response_compression_level must be in [0, 9].")
        if response_compression_min_bytes < 0:
            raise ValueError("response_compression_min_bytes must be non-negative.")
        self.host = host
        self.port = int(port)
        self.auth_token = auth_token
        self.request_timeout = float(request_timeout)
        self.connect_timeout = float(connect_timeout)
        self.reconnect_attempts = int(reconnect_attempts)
        self.tunnel = tunnel
        self.response_compression = response_compression
        self.response_compression_level = int(response_compression_level)
        self.response_compression_min_bytes = int(response_compression_min_bytes)
        self.session_id = uuid.uuid4().hex
        self._next_request_id = 0
        self._server_instance_id: str | None = None
        self._connection: socket.socket | None = None
        self._last_call_metrics: dict[str, float | str] = {}

    def call(self, method: str, payload: Any = None) -> Any:
        """Execute one method, replaying the same id after transport failures."""
        request_id = self._next_request_id
        request_data = {
            "protocol_version": _PROTOCOL_VERSION,
            "auth_token": self.auth_token,
            "session_id": self.session_id,
            "request_id": request_id,
            "method": method,
            "payload": payload,
        }
        if self.response_compression != "none":
            # Requests remain uncompressed so a new client can still talk to an
            # older daemon. Old daemons ignore this optional response hint and
            # new clients transparently decode either response format.
            request_data["response_compression"] = {
                "codec": self.response_compression,
                "level": self.response_compression_level,
                "min_bytes": self.response_compression_min_bytes,
            }
        request = encode_message(request_data)
        started_at = time.monotonic()
        last_error: BaseException | None = None
        for attempt in range(self.reconnect_attempts + 1):
            try:
                connection = self._connect()
                send_frame(connection, request)
                response_payload = recv_frame(connection)
                response = decode_message(response_payload)
                result = self._validate_response(response, request_id)
                elapsed = time.monotonic() - started_at
                handler_seconds = float(response.get("handler_seconds", 0.0))
                encoded_blob_bytes, raw_blob_bytes = _message_blob_sizes(
                    response_payload
                )
                self._last_call_metrics = {
                    "method": method,
                    "rpc_seconds": elapsed,
                    "handler_seconds": handler_seconds,
                    "transport_seconds": max(elapsed - handler_seconds, 0.0),
                    "request_mib": (len(request) + _LENGTH.size) / (1024**2),
                    "response_mib": (len(response_payload) + _LENGTH.size) / (1024**2),
                    "response_raw_blob_mib": raw_blob_bytes / (1024**2),
                    "response_compression_ratio": (
                        encoded_blob_bytes / raw_blob_bytes if raw_blob_bytes else 1.0
                    ),
                    "reconnects": float(attempt),
                }
                self._next_request_id += 1
                return result
            except (OSError, EOFError, RemoteCollectorConnectionError) as exc:
                last_error = exc
                self._drop_connection()
                if attempt == self.reconnect_attempts:
                    break
                if self.tunnel is not None:
                    self.tunnel.close()
                time.sleep(min(0.25 * 2**attempt, 2.0))
        raise RemoteCollectorError(
            f"Remote collector request {request_id} ({method}) failed after "
            f"{self.reconnect_attempts + 1} transport attempts."
        ) from last_error

    def last_call_metrics(self) -> dict[str, float | str]:
        """Return transport metrics for the most recently completed RPC."""
        return dict(self._last_call_metrics)

    def _connect(self) -> socket.socket:
        if self._connection is not None:
            return self._connection
        if self.tunnel is not None:
            self.tunnel.start()
        connection = socket.create_connection(
            (self.host, self.port), timeout=self.connect_timeout
        )
        connection.settimeout(self.request_timeout)
        self._connection = connection
        return connection

    def _validate_response(self, response: Any, request_id: int) -> Any:
        if not isinstance(response, dict):
            raise RemoteCollectorProtocolError("RPC response is not a dictionary.")
        if response.get("protocol_version") != _PROTOCOL_VERSION:
            raise RemoteCollectorProtocolError("RPC protocol version mismatch.")
        if response.get("request_id") != request_id:
            raise RemoteCollectorProtocolError("RPC response request id mismatch.")
        server_instance_id = response.get("server_instance_id")
        if not isinstance(server_instance_id, str):
            raise RemoteCollectorProtocolError(
                "RPC response has no server instance id."
            )
        if self._server_instance_id is None:
            self._server_instance_id = server_instance_id
        elif server_instance_id != self._server_instance_id:
            raise RemoteCollectorStateLostError(
                "Remote collector daemon restarted; the simulator state was lost."
            )
        if not response.get("ok", False):
            error_type = response.get("error_type", "RemoteError")
            message = response.get("message", "Remote collector call failed.")
            remote_traceback = response.get("traceback", "")
            if error_type == "RemoteCollectorStateLostError":
                raise RemoteCollectorStateLostError(message)
            raise RemoteCollectorError(
                f"{error_type}: {message}\nRemote traceback:\n{remote_traceback}"
            )
        return response.get("result")

    def _drop_connection(self) -> None:
        if self._connection is None:
            return
        try:
            self._connection.close()
        finally:
            self._connection = None

    def close(self) -> None:
        """Close the client connection and its optional SSH tunnel."""
        self._drop_connection()
        if self.tunnel is not None:
            self.tunnel.close()


class RemoteCollectorServer:
    """Single-environment TCP server with duplicate-request suppression."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        auth_token: str,
        handler: Callable[[str, Any], Any],
        max_message_bytes: int = _DEFAULT_MAX_MESSAGE_BYTES,
    ) -> None:
        if not auth_token:
            raise ValueError("Remote collector auth token must not be empty.")
        self.host = host
        self.port = int(port)
        self.auth_token = auth_token
        self.handler = handler
        self.max_message_bytes = int(max_message_bytes)
        self.server_instance_id = uuid.uuid4().hex
        self._session_id: str | None = None
        self._session_initialize_digest: bytes | None = None
        self._session_closed = False
        self._last_request_id = -1
        self._last_request_digest: bytes | None = None
        self._last_response: bytes | None = None
        self._listener: socket.socket | None = None
        self._stop_event = threading.Event()

    @property
    def bound_port(self) -> int:
        """Return the actual bound port after the server starts."""
        if self._listener is None:
            return self.port
        return int(self._listener.getsockname()[1])

    def serve_forever(self, *, ready: threading.Event | None = None) -> None:
        """Accept sequential clients until :meth:`shutdown` is called."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((self.host, self.port))
            listener.listen(8)
            listener.settimeout(0.5)
            self._listener = listener
            if ready is not None:
                ready.set()
            while not self._stop_event.is_set():
                try:
                    connection, _ = listener.accept()
                except socket.timeout:
                    continue
                with connection:
                    self._serve_connection(connection)
            self._listener = None

    def _serve_connection(self, connection: socket.socket) -> None:
        while not self._stop_event.is_set():
            try:
                request_payload = recv_frame(
                    connection, max_message_bytes=self.max_message_bytes
                )
            except (EOFError, OSError):
                return
            try:
                response_payload = self._handle_request(request_payload)
            except Exception as exc:  # noqa: BLE001 - protect the daemon loop
                response_payload = encode_message(
                    self._error_response(-1, exc, traceback.format_exc())
                )
            try:
                send_frame(connection, response_payload)
            except OSError:
                return

    def _handle_request(self, request_payload: bytes) -> bytes:
        request = decode_message(request_payload)
        if not isinstance(request, dict):
            raise RemoteCollectorProtocolError("RPC request is not a dictionary.")
        if request.get("protocol_version") != _PROTOCOL_VERSION:
            raise RemoteCollectorProtocolError("RPC protocol version mismatch.")
        request_id = request.get("request_id")
        if not isinstance(request_id, int) or request_id < 0:
            raise RemoteCollectorProtocolError("Invalid RPC request id.")
        if not hmac.compare_digest(str(request.get("auth_token", "")), self.auth_token):
            return encode_message(
                self._error_response(
                    request_id, PermissionError("Authentication failed.")
                )
            )
        session_id = request.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            return encode_message(
                self._error_response(request_id, ValueError("Invalid session id."))
            )
        digest = hashlib.sha256(request_payload).digest()

        method = request.get("method")
        starts_session = request_id == 0 and method == "initialize"
        initialize_digest = None
        if starts_session:
            initialize_digest = hashlib.sha256(
                encode_message(
                    {
                        "method": method,
                        "payload": request.get("payload"),
                    }
                )
            ).digest()
        if self._session_id is None:
            if not starts_session:
                return encode_message(
                    self._error_response(
                        request_id,
                        RemoteCollectorStateLostError(
                            "Collector has no active session; initialize is required."
                        ),
                    )
                )
            self._session_id = session_id
            self._session_initialize_digest = initialize_digest
        elif session_id != self._session_id:
            resumes_initialized_session = (
                starts_session and initialize_digest == self._session_initialize_digest
            )
            if not starts_session or (
                not self._session_closed and not resumes_initialized_session
            ):
                return encode_message(
                    self._error_response(
                        request_id,
                        RemoteCollectorStateLostError(
                            "Collector is already owned by another session."
                        ),
                    )
                )
            self._session_id = session_id
            self._session_closed = False
            self._session_initialize_digest = initialize_digest
            self._last_request_id = -1
            self._last_request_digest = None
            self._last_response = None

        if request_id == self._last_request_id:
            if digest != self._last_request_digest:
                return encode_message(
                    self._error_response(
                        request_id,
                        RemoteCollectorProtocolError(
                            "A duplicate request id carried different content."
                        ),
                    )
                )
            if self._last_response is None:
                raise AssertionError("Cached request is missing its response.")
            return self._last_response
        if self._session_closed:
            return encode_message(
                self._error_response(
                    request_id,
                    RemoteCollectorStateLostError(
                        "Collector session is closed; initialize a new session."
                    ),
                )
            )
        if request_id != self._last_request_id + 1:
            return encode_message(
                self._error_response(
                    request_id,
                    RemoteCollectorProtocolError(
                        f"Expected request id {self._last_request_id + 1}, got "
                        f"{request_id}."
                    ),
                )
            )

        compression = request.get("response_compression", {})
        if not isinstance(compression, dict):
            return encode_message(
                self._error_response(
                    request_id,
                    RemoteCollectorProtocolError(
                        "response_compression must be a dictionary."
                    ),
                )
            )
        response_codec = compression.get("codec", "none")
        response_level = compression.get("level", 1)
        response_min_bytes = compression.get(
            "min_bytes", _DEFAULT_COMPRESSION_MIN_BYTES
        )
        if response_codec not in _SUPPORTED_BLOB_COMPRESSION:
            return encode_message(
                self._error_response(
                    request_id,
                    RemoteCollectorProtocolError(
                        f"Unsupported response compression {response_codec!r}."
                    ),
                )
            )
        if type(response_level) is not int or not 0 <= response_level <= 9:
            return encode_message(
                self._error_response(
                    request_id,
                    RemoteCollectorProtocolError(
                        "response compression level must be an integer in [0, 9]."
                    ),
                )
            )
        if type(response_min_bytes) is not int or response_min_bytes < 0:
            return encode_message(
                self._error_response(
                    request_id,
                    RemoteCollectorProtocolError(
                        "response compression min_bytes must be a non-negative integer."
                    ),
                )
            )

        try:
            handler_started_at = time.monotonic()
            result = self.handler(request["method"], request.get("payload"))
            response = {
                "protocol_version": _PROTOCOL_VERSION,
                "server_instance_id": self.server_instance_id,
                "request_id": request_id,
                "ok": True,
                "result": result,
                "handler_seconds": time.monotonic() - handler_started_at,
            }
        except Exception as exc:  # noqa: BLE001 - return remote exception details
            response = self._error_response(request_id, exc, traceback.format_exc())
        try:
            response_payload = encode_message(
                response,
                compression=response_codec,
                compression_level=response_level,
                compression_min_bytes=response_min_bytes,
            )
        except Exception as exc:  # noqa: BLE001 - state may already have advanced
            response_payload = encode_message(
                self._error_response(request_id, exc, traceback.format_exc())
            )
        self._last_request_id = request_id
        self._last_request_digest = digest
        self._last_response = response_payload
        if response["ok"] and method == "close":
            self._session_closed = True
        return response_payload

    def _error_response(
        self,
        request_id: int,
        error: BaseException,
        remote_traceback: str = "",
    ) -> dict[str, Any]:
        return {
            "protocol_version": _PROTOCOL_VERSION,
            "server_instance_id": self.server_instance_id,
            "request_id": request_id,
            "ok": False,
            "error_type": type(error).__name__,
            "message": str(error),
            "traceback": remote_traceback,
        }

    def shutdown(self) -> None:
        """Stop the accept loop and unblock a pending ``accept`` call."""
        self._stop_event.set()
        listener = self._listener
        if listener is None:
            return
        try:
            with socket.create_connection(listener.getsockname(), timeout=0.2):
                pass
        except OSError:
            pass


def auth_token_from_env(name: str) -> str:
    """Read a non-empty collector authentication token from the environment."""
    token = os.environ.get(name, "")
    if not token:
        raise ValueError(f"Environment variable {name!r} is empty or undefined.")
    return token


__all__ = [
    "RemoteCollectorClient",
    "RemoteCollectorConnectionError",
    "RemoteCollectorError",
    "RemoteCollectorProtocolError",
    "RemoteCollectorServer",
    "RemoteCollectorStateLostError",
    "SshLocalForward",
    "auth_token_from_env",
    "decode_message",
    "encode_message",
    "recv_frame",
    "send_frame",
]
