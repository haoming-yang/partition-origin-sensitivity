from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import nn


def linear_head_delta_contributions(
    head: nn.Linear,
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    feature_order: str = "token_feature",
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(head, nn.Linear):
        raise TypeError("head must be torch.nn.Linear")
    if features_a.shape != features_b.shape or features_a.ndim != 4:
        raise ValueError("features must have shape [batch, group, token, width] or [batch, group, width, token]")
    if feature_order not in {"token_feature", "feature_token"}:
        raise ValueError("feature_order must be token_feature or feature_token")
    batch, group, first, second = features_a.shape
    tokens, width = (first, second) if feature_order == "token_feature" else (second, first)
    if head.in_features != tokens * width:
        raise ValueError("head input width does not match features")
    delta = features_b - features_a
    if feature_order == "token_feature":
        weight = head.weight.reshape(head.out_features, tokens, width)
        contributions = torch.einsum("bgtd,otd->bgto", delta, weight)
    else:
        weight = head.weight.reshape(head.out_features, width, tokens)
        contributions = torch.einsum("bgdt,odt->bgto", delta, weight)
    output_delta = contributions.sum(dim=2)
    return contributions, output_delta


def summarize_token_contributions(contributions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    if contributions.ndim != 4:
        raise ValueError("contributions must have shape [batch, group, token, output]")
    count = contributions.shape[0] * contributions.shape[1] * contributions.shape[3]
    absolute = contributions.abs().sum(dim=(0, 1, 3)) / count
    signed = contributions.sum(dim=(0, 1, 3)) / count
    return absolute, signed, count


def summarize_position_gradient(gradient: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    if gradient.ndim != 3:
        raise ValueError("gradient must have shape [batch, position, channel]")
    count = gradient.shape[0] * gradient.shape[2]
    absolute = gradient.abs().sum(dim=(0, 2)) / count
    signed = gradient.sum(dim=(0, 2)) / count
    return absolute, signed, count


def masked_location_statistics(
    values: torch.Tensor, observed: torch.Tensor, variance_epsilon: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 3 or observed.ndim != 2 or values.shape[:2] != observed.shape:
        raise ValueError("values must have shape [batch, time, channel] and observed [batch, time]")
    mask = observed.unsqueeze(-1).to(values.dtype)
    count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (values * mask).sum(dim=1, keepdim=True) / count
    centered = (values - mean) * mask
    stdev = torch.sqrt((centered * centered).sum(dim=1, keepdim=True) / count + variance_epsilon)
    return mean[:, 0, :], stdev[:, 0, :]


def rademacher_jacobian_energy(
    forward_a: Callable[[torch.Tensor], torch.Tensor],
    forward_b: Callable[[torch.Tensor], torch.Tensor],
    raw: torch.Tensor,
    projections: int = 2,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if projections <= 0:
        raise ValueError("projections must be positive")
    values = raw.detach().clone().requires_grad_(True)
    prediction_a = forward_a(values)
    prediction_b = forward_b(values)
    flat_a = prediction_a.flatten(1)
    flat_b = prediction_b.flatten(1)
    flat_difference = flat_a - flat_b
    energy_a = torch.zeros_like(values)
    energy_b = torch.zeros_like(values)
    energy_difference = torch.zeros_like(values)
    for _ in range(projections):
        signs = torch.randint(0, 2, flat_a.shape, device=flat_a.device, dtype=flat_a.dtype, generator=generator).mul_(2).sub_(1)
        gradient_a = torch.autograd.grad((flat_a * signs).sum(), values, retain_graph=True)[0]
        gradient_b = torch.autograd.grad((flat_b * signs).sum(), values, retain_graph=True)[0]
        gradient_difference = torch.autograd.grad((flat_difference * signs).sum(), values, retain_graph=True)[0]
        energy_a += gradient_a.square()
        energy_b += gradient_b.square()
        energy_difference += gradient_difference.square()
    return energy_a / projections, energy_b / projections, energy_difference / projections


def disagreement_gradient(
    forward_a: Callable[[torch.Tensor], torch.Tensor],
    forward_b: Callable[[torch.Tensor], torch.Tensor],
    raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = raw.detach().clone().requires_grad_(True)
    prediction_a = forward_a(values)
    prediction_b = forward_b(values)
    objective = (prediction_a - prediction_b).square().flatten(1).mean(dim=1)
    gradient = torch.autograd.grad(objective.sum(), values)[0]
    return objective.detach(), gradient.detach()


def finite_difference_response(
    forward_a: Callable[[torch.Tensor], torch.Tensor],
    forward_b: Callable[[torch.Tensor], torch.Tensor],
    raw: torch.Tensor,
    indices: Sequence[int],
    delta: float,
) -> torch.Tensor:
    if delta <= 0:
        raise ValueError("delta must be positive")
    base_a = forward_a(raw)
    base_b = forward_b(raw)
    base = (base_a - base_b).square().flatten(1).mean(dim=1)
    values = []
    for index in indices:
        perturbed = raw.clone()
        if perturbed.ndim == 2:
            perturbed[:, int(index)] += delta
        else:
            perturbed[:, int(index), ...] += delta
        changed_a = forward_a(perturbed)
        changed_b = forward_b(perturbed)
        changed = (changed_a - changed_b).square().flatten(1).mean(dim=1)
        values.append(((changed - base) / delta).mean())
    return torch.stack(values)
