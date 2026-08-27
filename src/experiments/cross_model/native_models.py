"""Small constructors for native models used by the external benchmark harness."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[3]
TIER1_ROOT = REPO_ROOT / "third_party" / "patch_models" / "tier1"
PATCHTST_ROOT = TIER1_ROOT / "patchtst" / "patchtst_supervised"
PATCHMIXER_ROOT = TIER1_ROOT / "patchmixer"
PATCHMLP_ROOT = TIER1_ROOT / "patchmlp"
PATHFORMER_ROOT = TIER1_ROOT / "pathformer"
HDMIXER_ROOT = TIER1_ROOT / "hdmixer"
DEFORMABLETST_ROOT = TIER1_ROOT / "deformabletst"


def _load_native_module(root: Path, module_name: str):
    """Import one source tree without retaining another clone's package namespace."""
    for name in list(sys.modules):
        if (
            name == "models" or name.startswith("models.") or name == "model" or name.startswith("model.")
            or name == "layers" or name.startswith("layers.") or name == "utils" or name.startswith("utils.")
        ):
            del sys.modules[name]
    sys.path.insert(0, str(root))
    try:
        return importlib.import_module(module_name)
    finally:
        sys.path.pop(0)


def build_patchtst(seq_len: int, pred_len: int, channels: int):
    """Build the cloned PatchTST implementation with a compact smoke configuration."""
    module = _load_native_module(PATCHTST_ROOT, "models.patchtst")
    config = SimpleNamespace(
        enc_in=channels,
        seq_len=seq_len,
        pred_len=pred_len,
        e_layers=1,
        n_heads=4,
        d_model=16,
        d_ff=64,
        dropout=0.1,
        fc_dropout=0.1,
        head_dropout=0.0,
        individual=False,
        patch_len=16,
        stride=8,
        padding_patch="end",
        revin=True,
        affine=False,
        subtract_last=False,
        decomposition=False,
        kernel_size=25,
    )
    return module.Model(config)


def build_patchmixer(seq_len: int, pred_len: int, channels: int):
    """Build the cloned PatchMixer implementation for the shared input length."""
    module = _load_native_module(PATCHMIXER_ROOT, "models.patchmixer")
    config = SimpleNamespace(
        enc_in=channels,
        seq_len=seq_len,
        pred_len=pred_len,
        patch_len=16,
        stride=8,
        mixer_kernel_size=8,
        a=0,
        d_model=64,
        dropout=0.1,
        head_dropout=0.0,
        e_layers=1,
    )
    return module.Model(config)


def build_patchmlp(seq_len: int, pred_len: int, channels: int):
    """Build the cloned multiscale PatchMLP implementation."""
    module = _load_native_module(PATCHMLP_ROOT, "model.patchmlp")
    config = SimpleNamespace(
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=channels,
        d_model=512,
        e_layers=1,
        output_attention=False,
        use_norm=True,
    )
    return module.Model(config)


def build_pathformer(seq_len: int, pred_len: int, channels: int):
    """Build the cloned Pathformer with all native multiscale experts enabled."""
    module = _load_native_module(PATHFORMER_ROOT, "models.pathformer")
    config = SimpleNamespace(
        seq_len=seq_len,
        pred_len=pred_len,
        layer_nums=3,
        num_nodes=channels,
        k=2,
        num_experts_list=[4, 4, 4],
        patch_size_list=[[16, 12, 8, 32], [12, 8, 6, 4], [8, 6, 4, 2]],
        d_model=4,
        d_ff=16,
        residual_connection=1,
        revin=1,
        gpu=0,
        batch_norm=0,
    )
    return module.Model(config)


def build_hdmixer(seq_len: int, pred_len: int, channels: int):
    """Build the cloned HDMixer with its native deformable-patch branch enabled."""
    module = _load_native_module(HDMIXER_ROOT, "models.hdmixer")
    config = SimpleNamespace(
        enc_in=channels,
        seq_len=seq_len,
        pred_len=pred_len,
        e_layers=1,
        n_heads=4,
        d_model=16,
        d_ff=64,
        dropout=0.1,
        fc_dropout=0.1,
        head_dropout=0.0,
        individual=False,
        patch_len=16,
        stride=8,
        padding_patch="end",
        revin=True,
        affine=False,
        subtract_last=False,
        decomposition=False,
        kernel_size=25,
        deform_patch=True,
        deform_range=0.25,
        lambda_=0.1,
        r=0.01,
        mix_time=True,
        mix_variable=True,
        mix_channel=True,
    )
    return module.Model(config)


def build_deformabletst(seq_len: int, pred_len: int, channels: int):
    """Build the cloned DeformableTST with its native 4-point convolutional stem."""
    module = _load_native_module(DEFORMABLETST_ROOT, "models.deformabletst")
    config = SimpleNamespace(
        n_vars=channels,
        revin=True,
        revin_affine=False,
        revin_subtract_last=False,
        stem_ratio=4,
        down_ratio=2,
        fmap_size=seq_len,
        dims=[8, 16, 32, 64],
        depths=[1, 1, 1, 1],
        drop_path_rate=0.0,
        layer_scale_value=[-1.0, -1.0, -1.0, -1.0],
        use_pe=[0, 0, 0, 0],
        use_lpu=[1, 1, 1, 1],
        local_kernel_size=[3, 3, 3, 3],
        expansion=2,
        drop=0.0,
        use_dwc_mlp=[1, 1, 1, 1],
        heads=[2, 4, 8, 16],
        attn_drop=0.0,
        proj_drop=0.0,
        stage_spec=[["D"], ["D"], ["D"], ["D"]],
        window_size=[3, 3, 3, 3],
        nat_ksize=[3, 3, 3, 3],
        ksize=[3, 3, 3, 3],
        stride=[1, 1, 1, 1],
        n_groups=[1, 2, 4, 8],
        offset_range_factor=[-1.0, -1.0, -1.0, -1.0],
        no_off=[0, 0, 0, 0],
        dwc_pe=[0, 0, 0, 0],
        fixed_pe=[0, 0, 0, 0],
        log_cpb=[0, 0, 0, 0],
        seq_len=seq_len,
        pred_len=pred_len,
        head_dropout=0.0,
        head_type="Flatten",
        use_head_norm=True,
    )
    return module.Model(config)


def build_model(name: str, seq_len: int, pred_len: int, channels: int):
    """Construct a currently supported native architecture by benchmark name."""
    builders = {
        "PatchTST": build_patchtst,
        "PatchMixer": build_patchmixer,
        "PatchMLP": build_patchmlp,
        "Pathformer": build_pathformer,
        "HDMixer": build_hdmixer,
        "DeformableTST": build_deformabletst,
    }
    try:
        return builders[name](seq_len=seq_len, pred_len=pred_len, channels=channels)
    except KeyError as exc:
        raise ValueError(f"unsupported native model: {name}") from exc


def native_forecast(name: str, model, x):
    """Call a model through its native forecasting signature."""
    if name == "PatchMLP":
        return model(x, None, None, None)
    if name == "Pathformer" or name == "HDMixer":
        return model(x)[0]
    return model(x)
