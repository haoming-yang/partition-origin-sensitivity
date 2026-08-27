"""Equal-information phase layouts for patch-origin evaluation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


def padded_length(context: int, patch: int, alignment: int | None = None) -> int:
    """Return the smallest patch-aligned length that supports every phase."""
    if context <= 0 or patch <= 1:
        raise ValueError("context must be positive and patch must exceed one")
    total = math.ceil((context + patch - 1) / patch) * patch
    if alignment is not None:
        if alignment <= 0 or alignment % patch:
            raise ValueError("alignment must be a positive multiple of patch")
        total = math.ceil(total / alignment) * alignment
    return total


def phase_pad(values: np.ndarray, phase: int, patch: int, total_length: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Place each context observation once in a patch-aligned padded sequence."""
    if values.ndim != 2:
        raise ValueError("values must have shape (context, channels)")
    context = values.shape[0]
    minimum_length = padded_length(context, patch)
    total_length = minimum_length if total_length is None else total_length
    if total_length < minimum_length or total_length % patch:
        raise ValueError("total_length must be patch-aligned and cover the full phase cycle")
    padding = total_length - context
    if not 0 <= phase < patch or phase > padding:
        raise ValueError("phase must fit in the available boundary padding")
    left = phase
    right = padding - phase
    padded = np.pad(values, ((left, right), (0, 0)), mode="constant")
    observed = np.zeros(total_length, dtype=bool)
    observed[left : left + context] = True
    return padded, observed


def audit_phase_layout(context: int, patch: int, total_length: int | None = None) -> list[dict[str, Any]]:
    """Return a per-phase proof that the original context is represented once."""
    values = np.arange(context, dtype=np.float32)[:, None]
    audit: list[dict[str, Any]] = []
    for phase in range(patch):
        padded, observed = phase_pad(values, phase, patch, total_length=total_length)
        recovered = padded[observed, 0].astype(int)
        audit.append(
            {
                "phase": phase,
                "padded_length": int(padded.shape[0]),
                "real_slots": int(observed.sum()),
                "unique_real_slots": int(np.unique(recovered).size),
                "exactly_once": bool(np.array_equal(recovered, np.arange(context))),
            }
        )
    return audit


def phase_pad_torch(values: torch.Tensor, phase: int, patch: int, total_length: int | None = None) -> torch.Tensor:
    """Batch version of :func:`phase_pad` for tensors shaped B x L x C."""
    if values.ndim != 3:
        raise ValueError("values must have shape (batch, context, channels)")
    context = values.shape[1]
    minimum_length = padded_length(context, patch)
    total_length = minimum_length if total_length is None else total_length
    if total_length < minimum_length or total_length % patch:
        raise ValueError("total_length must be patch-aligned and cover the full phase cycle")
    padding = total_length - context
    if not 0 <= phase < patch or phase > padding:
        raise ValueError("phase must fit in the available boundary padding")
    left = values.new_zeros((values.shape[0], phase, values.shape[2]))
    right = values.new_zeros((values.shape[0], padding - phase, values.shape[2]))
    return torch.cat((left, values, right), dim=1)
