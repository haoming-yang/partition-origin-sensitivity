import numpy as np

from src.experiments.cross_model.result_schema import summarize_phase_mse
from src.training.runner import formal_metrics


def test_origin_gap_uses_the_minimum_mse_denominator():
    metrics = formal_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0]))
    assert np.isclose(metrics["G_origin"], 200.0)


def test_pairwise_dispersion_identity_matches_origin_variance():
    predictions = np.array([0.0, 1.0, 3.0, 6.0])
    pairwise = np.mean(
        [(predictions[i] - predictions[j]) ** 2
         for i in range(len(predictions))
         for j in range(len(predictions))
         if i != j]
    )
    origin_variance = np.var(predictions)
    assert np.isclose(pairwise, 2 * len(predictions) / (len(predictions) - 1) * origin_variance)


def test_phase_metric_schema_reports_the_same_formal_gap():
    result = summarize_phase_mse([1.0, 2.0, 3.0])
    assert np.isclose(result["G_phase_pct"], 200.0)

