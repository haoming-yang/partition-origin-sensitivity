"""Layerwise wrappers around the token-axis latent spectrum summary."""

from __future__ import annotations

import numpy as np
import torch

from .latent_spectral import summarize_latent_pair


def dense_token_tensor(hidden: torch.Tensor, valid_tokens: torch.Tensor | None = None) -> torch.Tensor:
    """Restore a Transformer fast-path NestedTensor to its original token coordinates."""
    if not hidden.is_nested:
        return hidden
    compact = hidden.to_padded_tensor(0.0)
    if valid_tokens is None:
        return compact
    if valid_tokens.ndim != 2 or valid_tokens.shape[0] != compact.shape[0]:
        raise ValueError("valid token mask must have shape [rows, tokens]")
    output = compact.new_zeros((compact.shape[0], valid_tokens.shape[1], compact.shape[2]))
    for row in range(compact.shape[0]):
        count = int(valid_tokens[row].sum())
        output[row, valid_tokens[row]] = compact[row, :count]
    return output


def summarize_layer_pairs(
    layer_pairs: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    """Compute the same per-sample spectral summary for each named layer."""
    if not layer_pairs:
        raise ValueError("at least one named layer pair is required")
    return {name: summarize_latent_pair(first, second) for name, (first, second) in layer_pairs.items()}
