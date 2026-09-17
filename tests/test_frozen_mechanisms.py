import torch
from torch import nn

from src.analysis.frozen_mechanisms import (
    disagreement_gradient,
    finite_difference_response,
    linear_head_delta_contributions,
    masked_location_statistics,
    rademacher_jacobian_energy,
    summarize_position_gradient,
    summarize_token_contributions,
)
from tools.run_frozen_mechanism_matrix import discover_jobs


def test_linear_head_delta_contributions_sum_to_delta():
    head = nn.Linear(6, 2, bias=True)
    left = torch.arange(12, dtype=torch.float32).reshape(2, 1, 3, 2)
    right = left + 0.5
    contributions, delta = linear_head_delta_contributions(head, left, right)
    assert contributions.shape == (2, 1, 3, 2)
    expected = head(right.reshape(2, 1, -1)) - head(left.reshape(2, 1, -1))
    assert torch.allclose(contributions.sum(dim=2), delta)
    assert torch.allclose(delta, expected, atol=1e-6)


def test_feature_token_head_layout_matches_linear_head():
    head = nn.Linear(6, 2, bias=True)
    left = torch.arange(12, dtype=torch.float32).reshape(2, 1, 2, 3)
    right = left + 0.5
    contributions, delta = linear_head_delta_contributions(head, left, right, feature_order="feature_token")
    expected = head(right.reshape(2, 1, -1)) - head(left.reshape(2, 1, -1))
    assert torch.allclose(contributions.sum(dim=2), delta)
    assert torch.allclose(delta, expected, atol=1e-6)


def test_token_summary_is_weighted_by_actual_batch_elements():
    first = torch.ones(2, 1, 3, 4)
    second = torch.full((1, 1, 3, 4), 3.0)
    first_abs, first_signed, first_count = summarize_token_contributions(first)
    second_abs, second_signed, second_count = summarize_token_contributions(second)
    combined = torch.cat((first, second), dim=0)
    combined_abs, combined_signed, combined_count = summarize_token_contributions(combined)
    assert torch.allclose(
        (first_abs * first_count + second_abs * second_count) / (first_count + second_count),
        combined_abs,
    )
    assert torch.allclose(
        (first_signed * first_count + second_signed * second_count) / (first_count + second_count),
        combined_signed,
    )
    assert combined_count == first_count + second_count


def test_position_gradient_summary_uses_channel_weighting():
    gradient = torch.ones(2, 5, 3)
    absolute, signed, count = summarize_position_gradient(gradient)
    assert torch.allclose(absolute, torch.ones(5))
    assert torch.allclose(signed, torch.ones(5))
    assert count == 6


def test_masked_location_statistics_match_observed_values():
    values = torch.tensor([[[1.0], [3.0], [99.0], [5.0]]])
    observed = torch.tensor([[True, True, False, True]])
    mean, stdev = masked_location_statistics(values, observed)
    assert torch.allclose(mean, torch.tensor([[3.0]]))
    assert torch.allclose(stdev, torch.tensor([[torch.sqrt(torch.tensor(8.0 / 3.0 + 1e-5))]]))


def test_rademacher_jacobian_energy_returns_origin_and_difference_profiles():
    raw = torch.tensor([[[1.0], [2.0]]])

    def left(x):
        return x[:, :1]

    def right(x):
        return 2.0 * x[:, :1]

    energy_left, energy_right, energy_difference = rademacher_jacobian_energy(left, right, raw, projections=3)
    assert energy_left.shape == energy_right.shape == energy_difference.shape == raw.shape
    assert torch.allclose(energy_left, torch.tensor([[[1.0], [0.0]]]))
    assert torch.allclose(energy_right, torch.tensor([[[4.0], [0.0]]]))
    assert torch.allclose(energy_difference, torch.tensor([[[1.0], [0.0]]]))


def test_disagreement_gradient_uses_same_raw_observation():
    raw = torch.tensor([[1.0, 2.0]], requires_grad=True)

    def left(x):
        return x[:, :1]

    def right(x):
        return 2.0 * x[:, :1]

    value, gradient = disagreement_gradient(left, right, raw)
    assert torch.allclose(value, torch.tensor([1.0]))
    assert torch.allclose(gradient, torch.tensor([[2.0, 0.0]]))


def test_finite_difference_response_matches_linear_disagreement():
    raw = torch.tensor([[1.0, 2.0]])

    def left(x):
        return x[:, :1]

    def right(x):
        return 2.0 * x[:, :1]

    result = finite_difference_response(left, right, raw, [0, 1], 1e-4)
    assert result.shape == (2,)
    assert torch.allclose(result, torch.tensor([2.0003, 0.0]), atol=5e-4)


def test_matrix_discovery_maps_and_deduplicates_frozen_runs(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    for path, experiment in ((first, "CANONICAL"), (second, "DUPLICATE")):
        path.mkdir()
        (path / "checkpoint.pt").write_bytes(b"same")
        (path / "config.json").write_text(
            '{"experiment_id":"%s","model":"Controlled-Transformer-V2","dataset":"ETTh2","seed":42,"p":12,"stride":12,"context":512,"horizon":96}' % experiment,
            encoding="utf-8",
        )
    jobs = discover_jobs(tmp_path, "controlled", True)
    assert len(jobs) == 1
    assert jobs[0]["dataset"] == "ETTh2"
