import numpy as np

from src.patching.phase_protocol import audit_phase_layout
from src.training.runner import formal_metrics


def test_all_nonoverlap_origins_recover_each_observation_once():
    audit = audit_phase_layout(context=512, patch=12)
    assert len(audit) == 12
    assert all(row["exactly_once"] for row in audit)
    assert all(row["real_slots"] == 512 for row in audit)


def test_formal_gaps_use_minimum_mse_denominators():
    mse = np.array([1.0, 1.2, 1.1, 1.3], dtype=float)
    metrics = formal_metrics(mse, mse / 2.0)
    assert np.isclose(metrics["G_origin"], 30.0)
    assert np.isclose(metrics["G_interior"], (1.3 - 1.1) / 1.1 * 100.0)
