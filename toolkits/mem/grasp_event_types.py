"""Lightweight data types shared by grasp-event selection and launchers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rlinf.models.embodiment.openpi_rlinf.pi0_model.model import Observation


@dataclass(frozen=True)
class GraspEventSample:
    """One history observation paired with its demonstrated action chunk."""

    phase: str
    episode_index: int
    frame_index: int
    valid_history_frames: int
    observation: Observation
    actions: torch.Tensor
