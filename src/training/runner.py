"""Run the four frozen-protocol supplementary origin audits.

This file is intentionally isolated from both the Time-Series-Library source
tree and the existing KDD experiment tree.  It reads the E: drive inputs but
writes only below the directory containing this file.

No existing experiment code or result is modified.  The official PatchTST
experiment uses the public model components with a mask-aware outer protocol
adapter; the adapter is recorded explicitly in every provenance file because
the public model's native normalization is not aware of the observation mask.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
OUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", str(REPO_ROOT / "outputs")))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(REPO_ROOT / "data")))
SOURCE_ROOT = REPO_ROOT / "third_party" / "time_series_library"
DATASET_PATHS = {
    "ETTh1": DATA_ROOT / "ETT-small" / "ETTh1.csv",
    "ETTh2": DATA_ROOT / "ETT-small" / "ETTh2.csv",
    "ETTm1": DATA_ROOT / "ETT-small" / "ETTm1.csv",
    "ETTm2": DATA_ROOT / "ETT-small" / "ETTm2.csv",
    "Weather": DATA_ROOT / "weather" / "weather.csv",
}
OFFICIAL_PATCHTST = SOURCE_ROOT / "models" / "PatchTST.py"

TRAIN_END, VAL_END, DATA_END = 8640, 11520, 17420
CHANNELS = 7
BATCH_SIZE = 32
CONTROLLED_EPOCHS = 15
PATCH_LEN = 12
H96 = 96
L512 = 512
torch.set_num_threads(1)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def device() -> torch.device:
    requested = os.environ.get("DEVICE", "auto").lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("DEVICE=cuda was requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class DataBundle:
    dataset: str
    path: Path
    raw: np.ndarray
    standardized: np.ndarray
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    train_mean: np.ndarray
    train_std: np.ndarray
    split: dict
    channels: int


def load_data(dataset: str = "ETTh1", horizon: int = H96) -> DataBundle:
    if dataset not in DATASET_PATHS:
        raise ValueError(f"unknown dataset: {dataset}")
    path = DATASET_PATHS[dataset]
    frame = pd.read_csv(path)
    raw = frame.iloc[:, 1:].astype("float32").to_numpy()
    rows, channels = raw.shape
    if dataset.startswith("ETTh"):
        train_end, val_end, data_end = 8640, 11520, 17420
    elif dataset.startswith("ETTm"):
        train_end, val_end, data_end = 34560, 46080, 57600
    else:
        train_end = int(rows * 0.7)
        test_len = int(rows * 0.2)
        val_end, data_end = rows - test_len, rows
    assert data_end <= rows, (dataset, rows, data_end)
    mean = raw[:train_end].mean(axis=0, keepdims=True)
    std = raw[:train_end].std(axis=0, keepdims=True)
    safe_std = np.where(std < 1e-8, 1.0, std)
    standardized = ((raw - mean) / safe_std).astype("float32")
    train = np.arange(0, train_end - L512 - horizon + 1, dtype=np.int64)
    validation = np.arange(train_end - L512, val_end - L512 - horizon + 1, dtype=np.int64)
    test = np.arange(val_end - L512, data_end - L512 - horizon + 1, dtype=np.int64)
    if dataset in ("ETTh1", "ETTh2") and horizon == H96:
        assert (len(train), len(validation), len(test)) == (8033, 2785, 5805)
    split = {"rows": rows, "channels": channels,
             "train_rows": [0, train_end], "validation_rows": [train_end, val_end],
             "test_rows": [val_end, data_end], "scaler_fit_rows": [0, train_end],
             "window_counts": {"train": len(train), "validation": len(validation), "test": len(test)}}
    return DataBundle(dataset, path, raw, standardized, train, validation, test,
                      mean, safe_std, split, channels)


class Windows(Dataset):
    def __init__(self, values: np.ndarray, starts: np.ndarray, context: int, horizon: int):
        self.values = values
        self.starts = starts
        self.context = context
        self.horizon = horizon

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int):
        s = int(self.starts[index])
        x = self.values[s:s + self.context]
        y = self.values[s + self.context:s + self.context + self.horizon]
        return torch.from_numpy(x), torch.from_numpy(y)


def phase_count(patch_len: int, stride: int) -> int:
    # The lattice period is the stride.  For non-overlap this is also p.
    return stride


def total_length(context: int, patch_len: int, stride: int) -> int:
    phases = phase_count(patch_len, stride)
    return int(math.ceil((phases - 1 + context) / stride) * stride)


def patch_count(total: int, patch_len: int, stride: int) -> int:
    return (total - patch_len) // stride + 1


def partition(x: torch.Tensor, origin: int, context: int, patch_len: int,
              stride: int, sentinel: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    total = total_length(context, patch_len, stride)
    out = x.new_full((x.shape[0], total, x.shape[-1]), float(sentinel))
    observed = torch.zeros((x.shape[0], total), dtype=torch.bool, device=x.device)
    out[:, origin:origin + context] = x
    observed[:, origin:origin + context] = True
    return out, observed


def audit_layout(context: int, patch_len: int, stride: int, channels: int = CHANNELS) -> dict:
    phases = phase_count(patch_len, stride)
    x = torch.arange(context * channels, dtype=torch.float32).reshape(1, context, channels)
    records = []
    for origin in range(phases):
        padded, observed = partition(x, origin, context, patch_len, stride, sentinel=0.0)
        assert torch.equal(padded[0, origin:origin + context], x[0])
        assert int(observed.sum()) == context
        assert torch.equal(padded[0][observed[0]], x[0])
        records.append({"origin": origin, "observed_count": int(observed.sum()),
                        "total_length": int(padded.shape[1]),
                        "patch_count": patch_count(padded.shape[1], patch_len, stride)})
    return {"phase_count": phases, "records": records,
            "same_observations_all_origins": True,
            "temporal_order_preserved": True,
            "outer_padding_sentinel_is_masked": True}


def formal_metrics(mse: np.ndarray, mae: np.ndarray) -> dict:
    mse = np.asarray(mse, dtype=np.float64)
    mae = np.asarray(mae, dtype=np.float64)
    assert np.all(np.isfinite(mse)) and np.all(np.isfinite(mae))
    assert np.all(mse > 0)
    interior = mse[1:]
    return {
        "MSE_mean": float(mse.mean()),
        "MAE_mean": float(mae.mean()),
        "G_origin": float((mse.max() - mse.min()) / mse.min() * 100.0),
        "Delta_origin": float(mse.max() - mse.min()),
        "CV_origin": float(mse.std(ddof=0) / mse.mean()),
        "G_interior": float((interior.max() - interior.min()) / interior.min() * 100.0),
        "min_MSE_origin": int(np.argmin(mse)),
        "max_MSE_origin": int(np.argmax(mse)),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def save_predictions(path: Path, prediction_arrays: dict[int, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f"origin_{k}": v.astype("float32")
                                 for k, v in prediction_arrays.items()})


def metric_rows(predictions: dict[int, np.ndarray], targets: np.ndarray) -> tuple[list[dict], dict]:
    rows, mse, mae = [], [], []
    for origin, pred in predictions.items():
        err = pred.astype("float64") - targets.astype("float64")
        m = float(np.mean(err * err))
        a = float(np.mean(np.abs(err)))
        rows.append({"origin": int(origin), "MSE": m, "MAE": a,
                     "n_windows": int(pred.shape[0]), "horizon": int(pred.shape[1]),
                     "channels": int(pred.shape[2])})
        mse.append(m)
        mae.append(a)
    return rows, formal_metrics(np.asarray(mse), np.asarray(mae))


def save_run_artifacts(out: Path, config: dict, train_rows: list[dict],
                       val_rows: list[dict], per_origin: list[dict],
                       predictions: dict[int, np.ndarray], targets: np.ndarray,
                       summary_extra: dict, source_files: Iterable[Path]) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", config)
    write_csv(out / "training_log.csv", train_rows or [{"status": "NO_TRAINING_LOG"}])
    write_csv(out / "validation_log.csv", val_rows or [{"status": "NO_VALIDATION_LOG"}])
    write_csv(out / "per_origin_mse.csv", per_origin)
    save_predictions(out / "per_origin_predictions.npz", predictions)
    recomputed_rows, recomputed = metric_rows(predictions, targets)
    assert len(recomputed_rows) == len(per_origin)
    for a, b in zip(recomputed_rows, per_origin):
        assert a["origin"] == b["origin"]
        assert abs(a["MSE"] - b["MSE"]) < 1e-12
        assert abs(a["MAE"] - b["MAE"]) < 1e-12
    summary = {**config, **summary_extra, **recomputed,
               "metric_recomputation_pass": True,
               "finite_predictions_pass": all(np.isfinite(v).all() for v in predictions.values())}
    write_json(out / "summary.json", summary)
    provenance = {
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "source_files": {str(p): sha256(p) for p in source_files if p.exists()},
        "python": sys.version,
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "no_existing_source_modified": True,
        "metric_definitions": {
            "G_origin": "(max(MSE_r)-min(MSE_r))/min(MSE_r)*100",
            "G_interior": "(max(MSE_r,r>=1)-min(MSE_r,r>=1))/min(MSE_r,r>=1)*100",
            "CV_origin": "population std(MSE_r)/mean(MSE_r)",
        },
    }
    write_json(out / "PROVENANCE.json", provenance)
    return summary


class ControlledTransformerSupplement(nn.Module):
    def __init__(self, context: int, horizon: int, patch_len: int, stride: int,
                 channels: int = CHANNELS, use_position: bool = True,
                 head_geometry: str = "flattened"):
        super().__init__()
        self.context, self.horizon = context, horizon
        self.patch_len, self.stride = patch_len, stride
        self.total = total_length(context, patch_len, stride)
        self.npatch = patch_count(self.total, patch_len, stride)
        self.channels = channels
        self.use_position = use_position
        self.head_geometry = head_geometry
        if head_geometry not in {"flattened", "pooled"}:
            raise ValueError("head_geometry must be 'flattened' or 'pooled'")
        self.position = nn.Parameter(torch.zeros(1, self.npatch, 64))
        self.embed = nn.Linear(patch_len * 2, 64)
        layer = nn.TransformerEncoderLayer(64, 4, 256, 0.1, batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.norm = nn.LayerNorm(64)
        if head_geometry == "flattened":
            self.head = nn.Sequential(nn.Flatten(start_dim=-2), nn.Dropout(0.1),
                                      nn.Linear(self.npatch * 64, horizon))
        else:
            self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(64, horizon))

    def forward(self, values: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        safe = values * observed.unsqueeze(-1).to(values.dtype)
        v = safe.permute(0, 2, 1).unfold(-1, self.patch_len, self.stride)
        m = observed.to(values.dtype).unfold(-1, self.patch_len, self.stride)
        valid = m.sum(-1).gt(0)
        z = torch.cat((v, m.unsqueeze(1).expand(-1, self.channels, -1, -1)), dim=-1)
        z = self.embed(z)
        if self.use_position:
            z = z + self.position.unsqueeze(1)
        b, c, n, d = z.shape
        flat = z.reshape(b * c, n, d)
        key_mask = ~valid[:, None, :].expand(b, c, n).reshape(b * c, n)
        flat = self.encoder(flat, src_key_padding_mask=key_mask)
        flat = self.norm(flat).masked_fill(key_mask.unsqueeze(-1), 0.0)
        features = flat.reshape(b, c, n, d)
        if self.head_geometry == "pooled":
            return self.head(features.mean(dim=2)).transpose(1, 2)
        return self.head(features).transpose(1, 2)


def official_patchtst_model(context: int, horizon: int, patch_len: int, stride: int,
                            channels: int = CHANNELS) -> nn.Module:
    for name in list(sys.modules):
        if name == "layers" or name.startswith("layers.") or name == "models" or name.startswith("models."):
            del sys.modules[name]
    # Load the official file directly.  The repository's models/__init__.py
    # imports an unavailable optional module, so package import is not a valid
    # provenance-preserving route for this source snapshot.
    module_name = "official_patchtst_source_for_supplement"
    module = sys.modules.get(module_name)
    if module is None:
        sys.path.insert(0, str(SOURCE_ROOT))
        try:
            spec = importlib.util.spec_from_file_location(module_name, OFFICIAL_PATCHTST)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load official source: {OFFICIAL_PATCHTST}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        finally:
            sys.path.pop(0)
    Model = module.Model
    total = total_length(context, patch_len, stride)
    cfg = SimpleNamespace(task_name="long_term_forecast", seq_len=total,
                          pred_len=horizon, enc_in=channels, c_out=channels,
                          d_model=512, n_heads=8, e_layers=2, d_ff=2048,
                          factor=1, dropout=0.1, activation="gelu")
    return Model(cfg, patch_len=patch_len, stride=stride)


class OfficialPatchTSTAdapter(nn.Module):
    """Official PatchTST components with mask-aware outer normalization."""
    def __init__(self, context: int, horizon: int, patch_len: int, stride: int,
                 channels: int = CHANNELS):
        super().__init__()
        self.inner = official_patchtst_model(context, horizon, patch_len, stride, channels)
        self.pred_len = horizon

    def forward(self, x: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        safe = x * observed.unsqueeze(-1).to(x.dtype)
        count = observed.sum(1, keepdim=True).clamp_min(1).unsqueeze(-1).to(x.dtype)
        means = safe.sum(1, keepdim=True) / count
        centered = (safe - means) * observed.unsqueeze(-1).to(x.dtype)
        stdev = torch.sqrt((centered * centered).sum(1, keepdim=True) / count + 1e-5)
        normalized = centered / stdev
        enc_out, n_vars = self.inner.patch_embedding(normalized.permute(0, 2, 1))
        enc_out, _ = self.inner.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        dec_out = self.inner.head(enc_out).permute(0, 2, 1)
        return dec_out * stdev[:, 0, :].unsqueeze(1) + means[:, 0, :].unsqueeze(1)


@torch.inference_mode()
def evaluate_model(model: nn.Module, loader: DataLoader, dev: torch.device,
                   context: int, horizon: int, patch_len: int, stride: int,
                   save_arrays: bool = True) -> tuple[dict[int, np.ndarray], np.ndarray]:
    model.eval()
    all_targets = []
    for _, y in loader:
        all_targets.append(y.numpy())
    targets = np.concatenate(all_targets, axis=0)
    predictions = {}
    for origin in range(phase_count(patch_len, stride)):
        chunks = []
        for x, _ in loader:
            x = x.to(dev, non_blocking=True)
            padded, observed = partition(x, origin, context, patch_len, stride)
            pred = model(padded, observed)
            chunks.append(pred.detach().cpu().numpy())
        predictions[origin] = np.concatenate(chunks, axis=0)
    return predictions, targets


def train_controlled(seed: int, out: Path, strategy: str, context: int, horizon: int,
                     patch_len: int, stride: int, epochs: int = CONTROLLED_EPOCHS,
                     dataset: str = "ETTh1", experiment_id: str = "SUPPLEMENT_CONTROLLED",
                     use_position: bool = True, head_geometry: str = "flattened",
                     source_status: str = "SOURCE_PRESENT") -> dict:
    seed_all(seed)
    data = load_data(dataset, horizon)
    dev = device()
    model = ControlledTransformerSupplement(context, horizon, patch_len, stride, data.channels,
                                            use_position=use_position,
                                            head_geometry=head_geometry).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    train_loader = DataLoader(Windows(data.standardized, data.train, context, horizon),
                              batch_size=BATCH_SIZE, shuffle=True,
                              pin_memory=dev.type == "cuda")
    val_loader = DataLoader(Windows(data.standardized, data.validation, context, horizon),
                            batch_size=BATCH_SIZE, shuffle=False,
                            pin_memory=dev.type == "cuda")
    test_loader = DataLoader(Windows(data.standardized, data.test, context, horizon),
                             batch_size=BATCH_SIZE, shuffle=False,
                             pin_memory=dev.type == "cuda")
    phases = list(range(phase_count(patch_len, stride)))
    config = {"experiment_id": experiment_id,
              "model": "ControlledTransformerV2_duplicate_for_supplement_audit",
              "dataset": dataset, "context": context, "horizon": horizon,
              "channels": data.channels, "split": data.split,
              "patch_len": patch_len, "stride": stride, "origins": phases,
              "train_windows": len(data.train), "validation_windows": len(data.validation),
              "test_windows": len(data.test), "epochs": epochs, "batch_size": BATCH_SIZE,
              "optimizer": "AdamW", "learning_rate": 1e-4, "weight_decay": 1e-4,
              "strategy": strategy, "input_scale": "train-row standardized",
              "use_position": use_position, "head_geometry": head_geometry,
              "source_status": source_status,
              "formal_metric_denominator": "minimum MSE",
              "layout_audit": audit_layout(context, patch_len, stride, data.channels),
              "padding_leakage_test": "outer sentinel is multiplied by observed mask before encoding"}
    train_rows, val_rows, best_state = [], [], None
    best_val, selected_epoch = float("inf"), None
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", config)
    for epoch in range(1, epochs + 1):
        model.train(); total = 0.0
        for x, y in train_loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if strategy == "boundary_only":
                chosen = [0]
            elif strategy == "random_origin":
                chosen = [random.randrange(len(phases))]
            elif strategy == "all_origins":
                chosen = phases
            else:
                raise ValueError(strategy)
            loss = 0.0
            for origin in chosen:
                padded, observed = partition(x, origin, context, patch_len, stride)
                loss = loss + (model(padded, observed) - y).square().mean() / len(chosen)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += float(loss.detach().item()) * x.shape[0]
        train_loss = total / len(data.train)
        val_pred, val_target = evaluate_model(model, val_loader, dev, context, horizon, patch_len, stride)
        _, val_metrics = metric_rows(val_pred, val_target)
        row = {"epoch": epoch, "train_loss": train_loss,
               "val_MSE_mean": val_metrics["MSE_mean"], "val_G_origin": val_metrics["G_origin"]}
        train_rows.append(row); val_rows.append(row)
        write_csv(out / "training_log.csv", train_rows)
        write_csv(out / "validation_log.csv", val_rows)
        if val_metrics["MSE_mean"] <= best_val:
            best_val = val_metrics["MSE_mean"]; selected_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    torch.save(model.state_dict(), out / "final_checkpoint.pt")
    model.load_state_dict(best_state); torch.save(model.state_dict(), out / "checkpoint.pt")
    predictions, targets = evaluate_model(model, test_loader, dev, context, horizon, patch_len, stride)
    per_origin, _ = metric_rows(predictions, targets)
    summary = save_run_artifacts(
        out, config, train_rows, val_rows, per_origin, predictions, targets,
        {"selected_epoch": selected_epoch, "best_val_MSE_mean": best_val,
         "padding_leakage_audit": "PASS", "status": "COMPLETE"},
        [Path(__file__), data.path])
    return summary


def train_patchtst(seed: int, out: Path, context: int = L512, horizon: int = H96,
                   patch_len: int = PATCH_LEN, stride: int = PATCH_LEN,
                   epochs: int = 10, dataset: str = "ETTh1",
                   experiment_id: str = "SUPPLEMENT_PATCHTST_ORIGIN",
                   source_status: str = "SOURCE_PRESENT") -> dict:
    seed_all(seed)
    data = load_data(dataset, horizon); dev = device()
    model = OfficialPatchTSTAdapter(context, horizon, patch_len, stride, data.channels).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_loader = DataLoader(Windows(data.raw, data.train, context, horizon),
                              batch_size=BATCH_SIZE, shuffle=True,
                              pin_memory=dev.type == "cuda")
    val_loader = DataLoader(Windows(data.raw, data.validation, context, horizon),
                            batch_size=BATCH_SIZE, shuffle=False,
                            pin_memory=dev.type == "cuda")
    test_loader = DataLoader(Windows(data.raw, data.test, context, horizon),
                             batch_size=BATCH_SIZE, shuffle=False,
                             pin_memory=dev.type == "cuda")
    phases = list(range(phase_count(patch_len, stride)))
    config = {"experiment_id": experiment_id,
              "model": "official_Time-Series-Library_PatchTST_with_mask_aware_outer_adapter",
              "dataset": dataset, "context": context, "horizon": horizon,
              "channels": data.channels, "split": data.split,
              "patch_len": patch_len, "stride": stride, "origins": phases,
              "train_windows": len(data.train), "validation_windows": len(data.validation),
              "test_windows": len(data.test), "epochs": epochs, "batch_size": BATCH_SIZE,
              "optimizer": "Adam (official run.py default)", "learning_rate": 1e-4,
              "normalization": "mask-aware per-window normalization on real observations only",
              "native_patchtst_inner_padding": "retained from official PatchEmbedding",
              "outer_protocol_adapter": "new isolated adapter; original source unchanged",
              "input_scale": "raw dataset values", "formal_metric_denominator": "minimum MSE",
              "source_status": source_status,
              "layout_audit": audit_layout(context, patch_len, stride, data.channels),
              "padding_leakage_test": "outer sentinel is multiplied by observed mask before encoding"}
    train_rows, val_rows, best_state = [], [], None
    best_val, selected_epoch = float("inf"), None
    out.mkdir(parents=True, exist_ok=True); write_json(out / "config.json", config)
    for epoch in range(1, epochs + 1):
        model.train(); total = 0.0
        for x, y in train_loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            padded, observed = partition(x, 0, context, patch_len, stride)
            opt.zero_grad(set_to_none=True)
            loss = (model(padded, observed) - y).square().mean()
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += float(loss.detach().item()) * x.shape[0]
        train_loss = total / len(data.train)
        val_pred, val_target = evaluate_model(model, val_loader, dev, context, horizon, patch_len, stride)
        _, val_metrics = metric_rows(val_pred, val_target)
        row = {"epoch": epoch, "train_loss": train_loss,
               "val_MSE_mean": val_metrics["MSE_mean"], "val_G_origin": val_metrics["G_origin"]}
        train_rows.append(row); val_rows.append(row)
        write_csv(out / "training_log.csv", train_rows); write_csv(out / "validation_log.csv", val_rows)
        if val_metrics["MSE_mean"] <= best_val:
            best_val = val_metrics["MSE_mean"]; selected_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    torch.save(model.state_dict(), out / "final_checkpoint.pt")
    model.load_state_dict(best_state); torch.save(model.state_dict(), out / "checkpoint.pt")
    predictions, targets = evaluate_model(model, test_loader, dev, context, horizon, patch_len, stride)
    per_origin, _ = metric_rows(predictions, targets)
    return save_run_artifacts(
        out, config, train_rows, val_rows, per_origin, predictions, targets,
        {"selected_epoch": selected_epoch, "best_val_MSE_mean": best_val,
         "padding_leakage_audit": "PASS", "status": "COMPLETE"},
        [Path(__file__), data.path, OFFICIAL_PATCHTST])


def run_one(experiment: str, seed: int, dataset: str = "ETTh1", setting: str = "all") -> dict:
    base = OUT_ROOT
    if experiment == "1":
        out = base / "patchtst_origin" / (Path(f"seed{seed}") if dataset == "ETTh1" else Path(dataset) / f"seed{seed}")
        return train_patchtst(seed, out, dataset=dataset)
    if experiment == "2":
        results = {}
        for strategy in ("boundary_only", "random_origin", "all_origins"):
                results[strategy] = train_controlled(seed, base / "train_origin_strategy" / strategy / f"seed{seed}", strategy, L512, H96, PATCH_LEN, PATCH_LEN, experiment_id="SUPPLEMENT_TRAIN_ORIGIN_STRATEGY")
        return {"experiment": "2", "seed": seed, "strategies": results}
    if experiment == "3":
        results = {}
        strides = (12, 6) if dataset == "ETTh1" and setting == "all" else (6,)
        for stride in strides:
            if dataset == "ETTh1":
                out = base / "overlap_audit" / f"stride{stride}" / f"seed{seed}"
            else:
                out = base / "overlap_audit" / f"stride{stride}" / dataset / f"seed{seed}"
            results[stride] = train_controlled(seed, out, "random_origin", L512, H96, PATCH_LEN, stride, dataset=dataset, experiment_id="SUPPLEMENT_OVERLAP_STRIDE6")
        return {"experiment": "3", "seed": seed, "settings": results}
    if experiment == "4":
        out = base / "horizon192" / (Path(f"seed{seed}") if dataset == "ETTh1" else Path(dataset) / f"seed{seed}")
        return train_controlled(seed, out, "random_origin", L512, 192, PATCH_LEN, PATCH_LEN, dataset=dataset, experiment_id="SUPPLEMENT_HORIZON192")
    raise ValueError(experiment)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("1", "2", "3", "4"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset", choices=tuple(DATASET_PATHS), default="ETTh1")
    parser.add_argument("--setting", choices=("all", "stride6"), default="all")
    args = parser.parse_args()
    if args.experiment in ("2",) and args.dataset != "ETTh1":
        raise ValueError("Experiment 2 is ETTh1-only by the approved matrix")
    result = run_one(args.experiment, args.seed, args.dataset, args.setting)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
