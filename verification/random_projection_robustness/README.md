# Independent Random-Projection Robustness Check

This anonymous supplementary package contains four unchanged archived seed-42
checkpoints, the exact diagnostic code used for the independent validation,
all 40 new profiles, sampled sign vectors, initial CUDA projection-generator
states, window start IDs, immutable input download sources, and paired statistics.
Historical profiles are supplied as comparison evidence and remain distinct.

Use Python 3.9.16 / PyTorch 2.5.1+cu121, CUDA 12.1 and the included environment
specification. The recorded device was an RTX 3060 Laptop GPU. Code byte hashes
in FILE_MANIFEST.json identify the exact packaged sources. Model builders
strictly load the fixed checkpoints and set eval mode with dropout disabled.
Only projection-generator seeds vary. No training is needed.

## Offline verification (no model/GPU computation)

    python -B verify_evidence.py
    python -B tests/test_projection_trace.py
    python -B reproduce.py

The last command prints the 40 run commands without executing them. Dataset CSVs
are distributed by immutable URLs and SHA-256 rather than redistributed here:

    python -B download_inputs.py

ETTh1/ETTh2 originate from zhouhaoyi/ETDataset at revision
1d16c8f4f943005d613b5bc962e9eeb06058cf07. Weather originates from
thuml/Time-Series-Library at revision 2b66e59ee19dac8f6f19fb5d4997f289fdfea357.
Exact file URLs, column order, scalers, split boundaries, timestamp anomalies,
and input-history row IDs are in provenance/datasets.json. Existing files with
a different hash are rejected. Consult each upstream dataset's license.

## Optional diagnostic replay

    python -B reproduce.py --run --output reruns

This explicitly starts new diagnostics. It preserves the archived results and
refuses existing output directories. Each estimate uses the first 512 fixed
chronological test windows, two output-shaped Rademacher projections per window,
shared signs across origins 0 and 6 and their difference, and batch sizes 4 for
controlled models or 2 for adapted PatchTST. Controlled inputs are training-
standardized; PatchTST inputs are raw. Output, temporary and cache paths stay
under this package. The model code uses the recorded protocol-adapted PatchTST
path, not an unmodified official training pipeline.

For each seed, normalized profile mass at positions 496–511 is computed after
aggregation. The reported quantity is origin-0 mass minus origin-6 mass in
percentage points. SD is computed from the ten paired differences with ddof=1.
statistics/paired_differences.csv contains every seed; ten_seed_metrics.json
includes means, sample SDs and observed ranges. These SDs quantify projection
randomness on fixed checkpoints and windows, not training or data uncertainty.

## Provenance boundaries

The recorded seed-42 checkpoint files match the archive manifest SHA-256.
configs/checkpoints.json records their archive-relative locations and hashes.
The independent CSV hashes match same-dataset historical inference artifacts,
whose data lineage is retained in provenance/. The original Jacobian execution
did not directly archive its dataset hash or input tensor, so this does not
establish exact historical Jacobian input provenance.

The new run's actual projection vectors and initial dedicated CUDA generator
states are included. The historical seed/state/vectors remain unavailable.
Global CUDA bitwise determinism was not enforced or verified. Independent
robustness does not constitute exact historical numerical reproduction.
The no-reversal finding covers only the origin-0 versus origin-6 normalized
Jacobian mass in the final 16 positions. It makes no robustness claim about
other parts of Figure 5 or disagreement-loss gradient diagnostics.

Paths in saved summary metadata were converted to package-relative paths for
anonymity. Checkpoints, numerical CSV profiles, sampled vectors, and RNG states
are copied byte-for-byte. Their SHA-256 values are in FILE_MANIFEST.json.
