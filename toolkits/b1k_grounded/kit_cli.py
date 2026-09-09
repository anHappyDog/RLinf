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

"""Command-line helpers shared by grounded B1K evaluation entrypoints."""

from __future__ import annotations

from collections.abc import Sequence


def split_hydra_and_kit_args(args: Sequence[str]) -> tuple[list[str], list[str]]:
    """Separate Hydra overrides from arguments consumed by Omniverse Kit.

    Args:
        args: Command-line arguments without the program name.

    Returns:
        A pair containing Hydra overrides and Kit arguments, respectively.

    Raises:
        ValueError: If ``--portable-root`` is missing its directory argument.
    """
    hydra_args: list[str] = []
    kit_args: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--portable-root":
            if index + 1 >= len(args):
                raise ValueError("--portable-root requires a directory argument")
            kit_args.extend((arg, args[index + 1]))
            index += 2
        elif arg.startswith("--/app/"):
            kit_args.append(arg)
            index += 1
        else:
            hydra_args.append(arg)
            index += 1
    return hydra_args, kit_args
