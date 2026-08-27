from __future__ import annotations

import csv
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import stage4_inference as base  # noqa: E402


STAGE = base.STAGE
LAMBDA_DIR = STAGE / "batch2_poc" / "lambda_selection"
PROTOCOL = LAMBDA_DIR / "LAMBDA_SELECTION_PROTOCOL.json"
SCHEDULE_ROOT = base.E_STAGE3 / "attribution_abc"
B_ROOT = SCHEDULE_ROOT / "B_two_view_supervised"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def file_sha(path: Path) -> str:
    return base.sha256(path)


def state_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("ascii"))
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def json_dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def finite_or_stop(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise RuntimeError(f"NaN/Inf detected in {name}: {value}")
    return value


def load_protocol() -> dict:
    if not PROTOCOL.exists():
        raise FileNotFoundError(PROTOCOL)
    p = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if p.get("candidate_lambdas") != [0.01, 0.03, 0.1, 0.3, 1.0]:
        raise RuntimeError("Frozen candidate lambda set does not match protocol")
    if p.get("seed") != 42 or p.get("selection_data", {}).get("uses_test") is not False:
        raise RuntimeError("Protocol is not the validation-only seed42 pilot")
    return p


def load_schedule(seed: int) -> tuple[dict, Path, str]:
    path = SCHEDULE_ROOT / f"origin_pair_schedule_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    schedule = json.loads(raw)
    if schedule.get("seed") != seed or len(schedule.get("epochs", [])) != 15:
        raise RuntimeError(f"Unexpected schedule metadata: {path}")
    for epoch in schedule["epochs"]:
        for batch in epoch.get("batches", []):
            ia, ib = int(batch["origin_a"]), int(batch["origin_b"])
            if not (0 <= ia < 12 and 0 <= ib < 12 and ia != ib):
                raise RuntimeError(f"Invalid distinct origin pair in {path}")
            if not batch.get("train_indices"):
                raise RuntimeError(f"Empty scheduled batch in {path}")
    return schedule, path, hashlib.sha256(raw).hexdigest()


def make_model(seed: int, channels: int):
    base.seed_all(seed)
    model = base.ControlledTransformerV2Family(12, channels).to(DEVICE)
    return model, state_hash(model)


def validation_loader(values, val):
    return torch.utils.data.DataLoader(
        base.Windows(values, val), batch_size=base.BATCH, shuffle=False,
        pin_memory=DEVICE.type == "cuda"
    )


@torch.inference_mode()
def dispersion_metrics(model, loader, p=12):
    model.eval()
    total_s = 0.0
    total_v = 0.0
    total_n = 0
    for x, _ in loader:
        x = x.to(DEVICE)
        preds = []
        for origin in range(p):
            v, m = base.partition(x, origin, p)
            preds.append(model(v, m).detach().cpu().numpy())
        z = np.stack(preds, axis=0)  # p, batch, horizon, channels
        mu = z.mean(axis=0)
        var_per_window = ((z - mu) ** 2).mean(axis=(0, 2, 3))
        pair_per_window = ((z[:, None] - z[None, :]) ** 2).mean(axis=(3, 4))
        s_per_window = pair_per_window[
            ~np.eye(p, dtype=bool)
        ].reshape(p * (p - 1), z.shape[1]).mean(axis=0)
        total_s += float(s_per_window.sum())
        total_v += float(var_per_window.sum())
        total_n += int(z.shape[1])
    if total_n <= 0:
        raise RuntimeError("Validation dispersion has no windows")
    s = finite_or_stop("validation S_theta", total_s / total_n)
    var = finite_or_stop("validation origin variance", total_v / total_n)
    rhs = finite_or_stop("validation identity RHS", 2 * p / (p - 1) * var)
    abs_err = abs(s - rhs)
    rel_err = abs_err / max(abs(s), 1e-12)
    if abs_err > 1e-5:
        raise RuntimeError(f"Validation S_theta identity failed: {s} vs {rhs}")
    return {
        "S_theta": s,
        "origin_prediction_variance": var,
        "identity_rhs": rhs,
        "identity_absolute_error": abs_err,
        "identity_relative_error": rel_err,
    }


@torch.inference_mode()
def validation_metrics(model, loader):
    rows = base.origin_metrics(model, loader, DEVICE, 12)
    aggregate = base.metric_from_rows(rows)
    aggregate.update(dispersion_metrics(model, loader, 12))
    for k, v in list(aggregate.items()):
        if isinstance(v, (float, np.floating)):
            aggregate[k] = finite_or_stop(f"validation {k}", v)
    return rows, aggregate


def batch_from_indices(values: np.ndarray, starts: list[int]):
    xs = np.stack([values[int(s):int(s) + base.CONTEXT] for s in starts]).astype("float32")
    ys = np.stack([
        values[int(s) + base.CONTEXT:int(s) + base.CONTEXT + base.HORIZON]
        for s in starts
    ]).astype("float32")
    return torch.from_numpy(xs).to(DEVICE), torch.from_numpy(ys).to(DEVICE)


def write_per_origin(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["origin", "MSE", "MAE"])
        writer.writeheader()
        writer.writerows(rows)


def preflight_once(model, values, loader):
    model.eval()
    return base.preflight(model, values, loader, 12, DEVICE)


def source_b_validation(protocol: dict) -> dict:
    seed = 42
    values, _, val, _, split = base.load_data("ETTh1")
    schedule, schedule_path, schedule_hash = load_schedule(seed)
    ckpt = B_ROOT / f"seed{seed}" / "checkpoint.pt"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    model, initial_hash = make_model(seed, values.shape[1])
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    model.eval()
    loader = validation_loader(values, val)
    preflight = preflight_once(model, values, loader)
    rows, metrics = validation_metrics(model, loader)
    result = {
        "experiment_id": "STAGE4_BATCH2_B_VALIDATION_ONLY_V1",
        "condition": "B_two_view_supervised",
        "dataset": "ETTh1", "p": 12, "seed": seed,
        "test_evaluation_performed": False,
        "source_checkpoint": str(ckpt),
        "checkpoint_sha256": file_sha(ckpt),
        "origin_pair_schedule": str(schedule_path),
        "origin_pair_schedule_sha256": schedule_hash,
        "protocol_sha256": file_sha(PROTOCOL),
        "initial_parameter_hash_reconstructed": initial_hash,
        "split": split, "preflight": preflight,
        "validation": {"per_origin": rows, **metrics},
        "schedule_metadata": {
            "epochs": len(schedule["epochs"]),
            "batches_per_epoch": len(schedule["epochs"][0]["batches"]),
        },
    }
    out = LAMBDA_DIR / "B_seed42_validation.json"
    json_dump(out, result)
    return result


def train_candidate(lam: float, protocol: dict, schedule: dict, schedule_path: Path, schedule_hash: str):
    values, _, val, _, split = base.load_data("ETTh1")
    out = LAMBDA_DIR / f"lambda_{str(lam).replace('.', 'p')}"
    out.mkdir(parents=True, exist_ok=True)
    model, initial_hash = make_model(42, values.shape[1])
    initial_state = copy.deepcopy(model.state_dict())
    loader = validation_loader(values, val)
    preflight = preflight_once(model, values, loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_val = float("inf")
    best_epoch = None
    best_state = None
    train_rows = []
    val_rows = []

    for epoch_entry in schedule["epochs"]:
        epoch = int(epoch_entry.get("epoch", len(train_rows) + 1))
        model.train()
        loss_sum = 0.0
        loss_count = 0
        pred_sum = 0.0
        cons_sum = 0.0
        for batch in epoch_entry["batches"]:
            indices = [int(i) for i in batch["train_indices"]]
            ia, ib = int(batch["origin_a"]), int(batch["origin_b"])
            x, y = batch_from_indices(values, indices)
            va, ma = base.partition(x, ia, 12)
            vb, mb = base.partition(x, ib, 12)
            optimizer.zero_grad(set_to_none=True)
            pa = model(va, ma)
            pb = model(vb, mb)
            l_pred = 0.5 * (F.mse_loss(pa, y) + F.mse_loss(pb, y))
            l_cons = F.mse_loss(pa, pb)
            loss = l_pred + float(lam) * l_cons
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at lambda={lam}, epoch={epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n = len(indices)
            loss_sum += float(loss.detach().item()) * n
            pred_sum += float(l_pred.detach().item()) * n
            cons_sum += float(l_cons.detach().item()) * n
            loss_count += n
        train_row = {
            "epoch": epoch,
            "lambda": lam,
            "loss": finite_or_stop("train loss", loss_sum / loss_count),
            "L_pred": finite_or_stop("train L_pred", pred_sum / loss_count),
            "L_cons": finite_or_stop("train L_cons", cons_sum / loss_count),
            "scheduled_batches": len(epoch_entry["batches"]),
            "scheduled_examples": loss_count,
        }
        train_rows.append(train_row)

        model.eval()
        per_origin, metrics = validation_metrics(model, loader)
        val_row = {"epoch": epoch, **metrics}
        val_rows.append(val_row)
        avg = float(metrics["avg_mse"])
        # <= implements the frozen tie rule: later epoch wins exact ties.
        if avg <= best_val:
            best_val = avg
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, out / "checkpoint.pt")
            write_per_origin(out / "best_val_per_origin.csv", per_origin)

    if best_state is None:
        raise RuntimeError(f"No checkpoint produced for lambda={lam}")
    torch.save(model.state_dict(), out / "final_checkpoint.pt")
    model.load_state_dict(best_state)
    model.eval()
    best_per_origin, best_metrics = validation_metrics(model, loader)

    config = {
        "experiment_id": "STAGE4_BATCH2_LAMBDA_PILOT_V1",
        "condition": "C_two_view_supervised_plus_consistency",
        "dataset": "ETTh1", "p": 12, "seed": 42, "lambda": lam,
        "epochs": 15, "batch_size": 32, "optimizer": "AdamW",
        "learning_rate": 1e-4, "weight_decay": 1e-4,
        "gradient_clipping_max_norm": 1.0, "scheduler": "none",
        "test_evaluation_performed": False,
        "origin_pair_schedule": str(schedule_path),
        "origin_pair_schedule_sha256": schedule_hash,
        "protocol_sha256": file_sha(PROTOCOL),
        "initial_parameter_hash": initial_hash,
        "checkpoint_rule": "minimum mean validation MSE across all origins; ties later epoch",
        "loss": "L_pred + lambda * L_cons",
    }
    json_dump(out / "config.json", config)
    json_dump(out / "PROVENANCE.json", {
        **config,
        "runner": str(Path(__file__)),
        "runner_sha256": file_sha(Path(__file__)),
        "evaluator_sha256": file_sha(SCRIPT_DIR / "stage4_inference.py"),
        "checkpoint_sha256": file_sha(out / "checkpoint.pt"),
        "final_checkpoint_sha256": file_sha(out / "final_checkpoint.pt"),
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
    })
    with (out / "train_loss.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(train_rows[0].keys()))
        writer.writeheader(); writer.writerows(train_rows)
    with (out / "val_loss.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(val_rows[0].keys()))
        writer.writeheader(); writer.writerows(val_rows)
    summary = {
        **config,
        "selected_epoch": best_epoch,
        "best_val_avg_mse": best_val,
        "validation": {"per_origin": best_per_origin, **best_metrics},
        "preflight": preflight,
        "initial_parameter_hash": initial_hash,
        "test_evaluation_performed": False,
    }
    json_dump(out / "summary.json", summary)
    return summary


def main():
    protocol = load_protocol()
    schedule, schedule_path, schedule_hash = load_schedule(42)
    b = source_b_validation(protocol)
    print(json.dumps({"B_validation_avg_mse": b["validation"]["avg_mse"], "S_theta": b["validation"]["S_theta"]}), flush=True)
    for lam in protocol["candidate_lambdas"]:
        result = train_candidate(float(lam), protocol, schedule, schedule_path, schedule_hash)
        print(json.dumps({"lambda": lam, "best_epoch": result["selected_epoch"], "val_avg_mse": result["best_val_avg_mse"], "val_S_theta": result["validation"]["S_theta"]}), flush=True)


if __name__ == "__main__":
    main()
