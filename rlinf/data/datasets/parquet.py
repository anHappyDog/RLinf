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

"""Bounded row-group reads for numeric episodes packed into Parquet files."""

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np


class ParquetEpisodeCache:
    """Cache selected numeric row groups and metadata without retaining file handles."""

    def __init__(self, max_files: int = 4) -> None:
        self.max_files = max_files
        self._metadata: OrderedDict[str, tuple[Any, np.ndarray]] = OrderedDict()
        self._groups: OrderedDict[tuple, dict[str, np.ndarray]] = OrderedDict()

    def read_columns(self, row: dict, columns: list[str]) -> dict[str, np.ndarray]:
        """Read an episode interval, decoding only intersecting row groups.

        ``row`` supplies ``data_file``, ``row_offset`` and ``frames``. Both caches
        contain at most ``max_files`` entries. Returned arrays retain the stored
        numeric dtypes and trailing list dimensions.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = str(Path(row["data_file"]))
        if path not in self._metadata:
            metadata = pq.read_metadata(path)
            ends = np.cumsum(
                [metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)]
            )
            self._metadata[path] = metadata, ends
            if len(self._metadata) > self.max_files:
                self._metadata.popitem(last=False)
        self._metadata.move_to_end(path)
        metadata, ends = self._metadata[path]
        start, stop = row["row_offset"], row["row_offset"] + row["frames"]
        if not 0 <= start < stop <= metadata.num_rows:
            raise ValueError(
                f"Expected a nonempty episode within {metadata.num_rows} rows; "
                f"found [{start}, {stop}) in {path}"
            )
        first = int(np.searchsorted(ends, start, side="right"))
        last = int(np.searchsorted(ends, stop - 1, side="right"))
        parts = {name: [] for name in columns}
        parquet = None
        try:
            for group in range(first, last + 1):
                key = path, group, tuple(columns)
                if key not in self._groups:
                    if parquet is None:
                        parquet = pq.ParquetFile(path, metadata=metadata)
                    table = parquet.read_row_group(group, columns=columns)
                    arrays = {}
                    for name in columns:
                        array = table[name].combine_chunks()
                        if pa.types.is_fixed_size_list(array.type) or pa.types.is_list(
                            array.type
                        ):
                            arrays[name] = (
                                array.flatten()
                                .to_numpy(zero_copy_only=False)
                                .reshape(len(array), -1)
                            )
                        else:
                            arrays[name] = array.to_numpy(zero_copy_only=False)
                    self._groups[key] = arrays
                    if len(self._groups) > self.max_files:
                        self._groups.popitem(last=False)
                self._groups.move_to_end(key)
                offset = int(ends[group - 1]) if group else 0
                lo = max(start - offset, 0)
                hi = min(stop, int(ends[group])) - offset
                for name in columns:
                    parts[name].append(self._groups[key][name][lo:hi])
        finally:
            if parquet is not None:
                parquet.close()
        return {
            name: values[0] if len(values) == 1 else np.concatenate(values, axis=0)
            for name, values in parts.items()
        }
