import numpy as np
import pytest
import torch

from src.analysis.latent_spectral import permutation_spearman_test, summarize_latent_pair
from src.analysis.layerwise import summarize_layer_pairs
from src.analysis.layerwise import dense_token_tensor
from src.analysis.patchtst import full_patchtst_token_latents
from src.training import runner
from tools.analyze_patchtst_latent_spectrum import _encode_and_predict
from tools.analyze_latent_spectrum import dataset_path_for


def test_identical_latents_have_zero_spectral_distance():
    hidden = np.arange(2 * 4 * 3, dtype=np.float64).reshape(2, 4, 3)
    summary = summarize_latent_pair(hidden, hidden.copy())

    assert summary["spectral_l1"].shape == (2,)
    assert np.allclose(summary["spectral_l1"], 0.0)
    assert np.allclose(summary["low_band_l1"], 0.0)
    assert np.allclose(summary["mid_band_l1"], 0.0)
    assert np.allclose(summary["high_band_l1"], 0.0)


def test_spectral_summary_is_per_sample_and_detects_a_change():
    first = np.zeros((2, 8, 1), dtype=np.float64)
    second = first.copy()
    second[1, ::2, 0] = 1.0
    second[1, 1::2, 0] = -1.0

    summary = summarize_latent_pair(first, second)

    assert summary["spectral_l1"].shape == (2,)
    assert summary["spectral_l1"][0] == 0.0
    assert summary["spectral_l1"][1] > 0.0
    assert summary["high_band_l1"][1] > summary["low_band_l1"][1]


def test_dc_and_non_dc_low_components_are_separated():
    first = np.zeros((1, 8, 1), dtype=np.float64)
    second = np.ones((1, 8, 1), dtype=np.float64)

    summary = summarize_latent_pair(first, second)

    assert summary["dc_l1"][0] > 0.0
    assert summary["non_dc_low_l1"][0] == 0.0
    assert summary["mid_band_l1"][0] == 0.0
    assert summary["high_band_l1"][0] == 0.0


def test_permutation_spearman_reports_small_two_sided_p_for_perfect_pairing():
    values = np.arange(12, dtype=np.float64)

    result = permutation_spearman_test(values, values, permutations=200, seed=0)

    assert result["observed_rho"] == pytest.approx(1.0)
    assert result["exceedances"] == 0
    assert result["p_two_sided"] == pytest.approx(1.0 / 201.0)
    assert abs(result["permutation_mean"]) < 0.1


def test_dataset_path_uses_the_cli_data_root_not_module_import_defaults(tmp_path):
    expected = tmp_path / "ETT-small" / "ETTh1.csv"

    assert dataset_path_for(tmp_path) == expected


def test_layerwise_summary_retains_each_named_layer():
    first = np.zeros((1, 4, 2), dtype=np.float64)
    second = first.copy()
    second[:, 1, :] = 1.0

    summary = summarize_layer_pairs({"layer0": (first, second), "layer1": (first, first)})

    assert set(summary) == {"layer0", "layer1"}
    assert summary["layer0"]["spectral_l1"][0] > 0.0
    assert summary["layer1"]["spectral_l1"][0] == 0.0


def test_dense_token_tensor_materializes_nested_transformer_output():
    with pytest.warns(UserWarning, match="NestedTensor"):
        nested = torch.nested.nested_tensor([torch.ones(2, 3), torch.ones(1, 3)])

    dense = dense_token_tensor(nested)

    assert dense.shape == (2, 2, 3)
    assert torch.equal(dense[1, 1], torch.zeros(3))


def test_dense_token_tensor_restores_nested_rows_to_valid_token_coordinates():
    nested = torch.nested.nested_tensor([torch.tensor([[1.0], [2.0]]), torch.tensor([[3.0]])])
    valid = torch.tensor([[True, False, True], [False, True, False]])

    dense = dense_token_tensor(nested, valid)

    assert dense.tolist() == [[[1.0], [0.0], [2.0]], [[0.0], [3.0], [0.0]]]


def test_full_patchtst_token_latents_excludes_partial_and_inner_padding_tokens():
    observed = torch.tensor([[True, True, False, False, True, True]])
    encoded = torch.tensor(
        [
            [[1.0], [2.0], [3.0], [99.0]],
            [[5.0], [6.0], [7.0], [99.0]],
        ]
    )

    latent = full_patchtst_token_latents(encoded, observed, patch_length=2, n_vars=2)

    assert latent.shape == (1, 2, 1)
    assert latent.tolist() == [[[3.0], [5.0]]]


def test_patchtst_latent_path_preserves_adapter_prediction():
    torch.manual_seed(0)
    model = runner.OfficialPatchTSTAdapter(context=24, horizon=4, patch_len=4, stride=4, channels=2)
    model.eval()
    x = torch.randn(2, 24, 2)
    padded, observed = runner.partition(x, origin=0, context=24, patch_len=4, stride=4)

    latent, prediction = _encode_and_predict(model, padded, observed, patch_length=4)

    assert latent.shape == (2, 6, 512)
    assert torch.allclose(prediction, model(padded, observed))
