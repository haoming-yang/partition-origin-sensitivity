from __future__ import annotations

import math


TIE_TOLERANCE = 1e-12


def select_lambda(rows: list[dict], baseline_mse: float, eligibility_multiplier: float) -> float | None:
    if not math.isfinite(baseline_mse) or baseline_mse <= 0:
        raise ValueError("baseline_mse must be finite and positive")
    if eligibility_multiplier != 1.01:
        raise ValueError("The recorded historical eligibility_multiplier is 1.01; do not change it here")
    for row in rows:
        if any(not math.isfinite(row[field]) or row[field] < 0
               for field in ("lambda_value", "validation_mse", "validation_s")):
            raise ValueError("Candidate lambda and validation metrics must be finite and nonnegative")
    eligible = [row for row in rows if row["validation_mse"] <= baseline_mse * eligibility_multiplier]
    if not eligible:
        return None
    best = min(row["validation_s"] for row in eligible)
    return min(row["lambda_value"] for row in eligible
               if abs(row["validation_s"] - best) <= TIE_TOLERANCE)
