"""Native-source metadata for the requested Tier1 models.

The harness runs each repository from its own root to avoid collisions among the
many local ``models`` and ``layers`` packages.  This module deliberately stores
metadata only; it never replaces an original architecture with a shared model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


TIER1_ROOT = REPO_ROOT / "third_party" / "patch_models" / "tier1"
TIER1_MODELS = (
    "PatchTST",
    "PatchMixer",
    "PatchMLP",
    "Pathformer",
    "HDMixer",
    "DeformableTST",
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    root: Path
    entrypoint: Path
    patch_size: int
    phase_count: int
    input_alignment: int
    protocol_note: str

    def input_length(self, context_len: int) -> int:
        """Return the smallest supported native input length for a context.

        Some cloned models require an input length divisible by all of their
        internal patch scales.  This is a model-contract property, not a
        change to the observation-equivalent protocol.
        """
        minimum = context_len + self.patch_size - 1
        return ((minimum + self.input_alignment - 1) // self.input_alignment) * self.input_alignment


def tier1_specs() -> dict[str, ModelSpec]:
    """Return the Tier1 sources and their native primary patch granularities."""
    roots = {
        "PatchTST": TIER1_ROOT / "patchtst" / "patchtst_supervised",
        "PatchMixer": TIER1_ROOT / "patchmixer",
        "PatchMLP": TIER1_ROOT / "patchmlp",
        "Pathformer": TIER1_ROOT / "pathformer",
        "HDMixer": TIER1_ROOT / "hdmixer",
        "DeformableTST": TIER1_ROOT / "deformabletst",
    }
    entries = {
        "PatchTST": "run_longexp.py",
        "PatchMixer": "run_longexp.py",
        "PatchMLP": "run.py",
        "Pathformer": "run.py",
        "HDMixer": "run_longexp.py",
        "DeformableTST": "run.py",
    }
    notes = {
        "PatchTST": "native patch_len=16, stride=8",
        "PatchMixer": "native patch_len=16, stride=8",
        "PatchMLP": "parallel native patch lengths 48, 24, 12, and 6; 48 covers every alignment",
        "Pathformer": "native 16/12/8/32/6/4/2 scales; 96 covers every alignment",
        "HDMixer": "native patch_len=16, stride=8",
        "DeformableTST": "phase is referenced to its 4-point convolutional stem",
    }
    sizes = {
        "PatchTST": 16,
        "PatchMixer": 16,
        "PatchMLP": 48,
        "Pathformer": 96,
        "HDMixer": 16,
        "DeformableTST": 4,
    }
    alignments = {name: sizes[name] for name in TIER1_MODELS}
    alignments["DeformableTST"] = 32
    return {
        name: ModelSpec(name, roots[name], roots[name] / entries[name], sizes[name], sizes[name], alignments[name], notes[name])
        for name in TIER1_MODELS
    }
