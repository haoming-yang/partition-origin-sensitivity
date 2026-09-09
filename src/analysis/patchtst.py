"""Latent extraction helpers for the protocol-adapted PatchTST audit."""

from __future__ import annotations

import torch


def full_patchtst_token_latents(
    encoded: torch.Tensor,
    observed: torch.Tensor,
    patch_length: int,
    n_vars: int,
) -> torch.Tensor:
    """Average encoder latents over variates and retain only complete outer tokens.

    ``encoded`` is the official PatchTST encoder output shaped
    ``[batch * n_vars, inner_tokens, hidden]``.  The official patch embedding
    appends one replication-padded token; it is excluded together with every
    outer token that is partially observed or fully padded.
    """
    if encoded.ndim != 3:
        raise ValueError("encoded must have shape [batch * n_vars, tokens, hidden]")
    if observed.ndim != 2 or observed.shape[1] % patch_length:
        raise ValueError("observed must have a patch-aligned [batch, time] shape")
    batch = observed.shape[0]
    if encoded.shape[0] != batch * n_vars:
        raise ValueError("encoded first dimension must equal batch * n_vars")
    outer_tokens = observed.shape[1] // patch_length
    if encoded.shape[1] < outer_tokens:
        raise ValueError("encoded sequence is shorter than the outer patch sequence")

    full = observed.reshape(batch, outer_tokens, patch_length).all(dim=-1)
    counts = full.sum(dim=-1)
    if int(counts.min()) != int(counts.max()):
        raise RuntimeError("batch windows have unequal numbers of complete tokens")
    reshaped = encoded.reshape(batch, n_vars, encoded.shape[1], encoded.shape[2])
    rows = [
        reshaped[row, :, :outer_tokens, :][:, full[row], :].mean(dim=0)
        for row in range(batch)
    ]
    return torch.stack(rows)
