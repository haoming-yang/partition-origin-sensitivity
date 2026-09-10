from __future__ import annotations

import argparse, csv, hashlib, json, os, random, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("PARTITION_ORIGIN_ETTH1_CSV", REPO_ROOT / "data" / "ETT-small" / "ETTh1.csv"))
CONTEXT, HORIZON, PATCH, TOTAL, CHANNELS = 512, 96, 12, 528, 7
TRAIN_END, VAL_END, DATA_END = 8640, 11520, 17420
EPOCHS, BATCH = 30, 32
torch.set_num_threads(1)


class ControlledTransformerV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.position = nn.Parameter(torch.zeros(1, 44, 64))
        self.embed = nn.Linear(24, 64)
        layer = nn.TransformerEncoderLayer(64, 4, 256, 0.1, batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.norm = nn.LayerNorm(64)
        self.head = nn.Sequential(nn.Flatten(start_dim=-2), nn.Dropout(0.1), nn.Linear(44 * 64, HORIZON))

    def forward(self, values, observed):
        safe = values * observed.unsqueeze(-1).to(values.dtype)
        v = safe.permute(0, 2, 1).unfold(-1, PATCH, PATCH)
        m = observed.to(values.dtype).unfold(-1, PATCH, PATCH)
        valid = m.sum(-1).gt(0)
        z = torch.cat((v, m.unsqueeze(1).expand(-1, CHANNELS, -1, -1)), dim=-1)
        z = self.embed(z) + self.position.unsqueeze(1)
        b, c, n, d = z.shape
        flat = z.reshape(b * c, n, d)
        key_mask = ~valid[:, None, :].expand(b, c, n).reshape(b * c, n)
        flat = self.encoder(flat, src_key_padding_mask=key_mask)
        flat = self.norm(flat).masked_fill(key_mask.unsqueeze(-1), 0.0)
        return self.head(flat.reshape(b, c, n, d)).transpose(1, 2)


class Windows(Dataset):
    def __init__(self, values, starts): self.values, self.starts = values, starts
    def __len__(self): return len(self.starts)
    def __getitem__(self, i):
        s = int(self.starts[i])
        return torch.from_numpy(self.values[s:s + CONTEXT]), torch.from_numpy(self.values[s + CONTEXT:s + CONTEXT + HORIZON])


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_data():
    raw = pd.read_csv(DATA).iloc[:, 1:].astype("float32").to_numpy()
    assert len(raw) == DATA_END and raw.shape[1] == CHANNELS, raw.shape
    mean, std = raw[:TRAIN_END].mean(0, keepdims=True), raw[:TRAIN_END].std(0, keepdims=True)
    values = ((raw - mean) / np.where(std < 1e-8, 1, std)).astype("float32")
    train = np.arange(0, TRAIN_END - CONTEXT - HORIZON + 1)
    val = np.arange(TRAIN_END - CONTEXT, VAL_END - CONTEXT - HORIZON + 1)
    test = np.arange(VAL_END - CONTEXT, DATA_END - CONTEXT - HORIZON + 1)
    assert (len(train), len(val), len(test)) == (8033, 2785, 5805)
    return values, train, val, test


def partition(x, origin, sentinel=0.0):
    out = x.new_full((x.shape[0], TOTAL, CHANNELS), sentinel)
    mask = torch.zeros((x.shape[0], TOTAL), dtype=torch.bool, device=x.device)
    out[:, origin:origin + CONTEXT] = x; mask[:, origin:origin + CONTEXT] = True
    return out, mask


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval(); rows = []
    for origin in range(PATCH):
        se = ae = n = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            v, m = partition(x, origin)
            e = model(v, m) - y
            se += e.square().sum().item(); ae += e.abs().sum().item(); n += e.numel()
        rows.append({"origin": origin, "MSE": se / n, "MAE": ae / n})
    a = np.array([r["MSE"] for r in rows]); best = a.min(); interior = a[1:]
    return rows, {"avg_mse": float(a.mean()), "avg_mae": float(np.mean([r["MAE"] for r in rows])),
                  "G_origin": float((a.max() - best) / best * 100), "CV_origin": float(a.std() / a.mean()),
                  "Delta_origin": float(a.max() - a.min()), "G_interior": float((interior.max() - interior.min()) / interior.min() * 100),
                  "min_mse_origin": int(a.argmin()), "max_mse_origin": int(a.argmax()),
                  "origin_0_mse": float(a[0]), "origin_11_mse": float(a[11])}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def run(seed, out):
    seed_all(seed); values, train, val, test = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ControlledTransformerV2().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    tl = DataLoader(Windows(values, train), batch_size=BATCH, shuffle=True, pin_memory=device.type == "cuda")
    vl = DataLoader(Windows(values, val), batch_size=BATCH, shuffle=False, pin_memory=device.type == "cuda")
    test_loader = DataLoader(Windows(values, test), batch_size=BATCH, shuffle=False, pin_memory=device.type == "cuda")
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    train_rows, val_rows, best_state, best_val, selected_epoch = [], [], None, float("inf"), EPOCHS
    for epoch in range(1, EPOCHS + 1):
        model.train(); total = 0.0
        for x, y in tl:
            x, y = x.to(device), y.to(device); origin = random.randrange(PATCH)
            v, m = partition(x, origin); opt.zero_grad(set_to_none=True)
            loss = (model(v, m) - y).square().mean(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); total += loss.item() * x.shape[0]
        train_loss = total / len(train)
        _, val_metrics = evaluate(model, vl, device)
        val_loss = val_metrics["avg_mse"]
        train_rows.append({"epoch": epoch, "train_loss": train_loss})
        val_rows.append({"epoch": epoch, "val_loss_avg_mse": val_loss})
        if val_loss <= best_val:
            best_val, selected_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(json.dumps({"seed": seed, "epoch": epoch, "train_loss": train_loss, "val_loss_avg_mse": val_loss}), flush=True)
    torch.save(model.state_dict(), out / "final_checkpoint.pt")
    model.load_state_dict(best_state); torch.save(model.state_dict(), out / "checkpoint.pt")
    test_rows, metrics = evaluate(model, test_loader, device)
    for name, rows in [("train_loss.csv", train_rows), ("val_loss.csv", val_rows), ("per_origin.csv", test_rows)]:
        with (out / name).open("w", newline="", encoding="utf8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary = {"experiment_id": "CANONICAL_STAGE3_LONGER_ETTH1_P12_V1", "model": "Controlled-Transformer-V2", "seed": seed,
               "dataset": "ETTh1", "context": CONTEXT, "horizon": HORIZON, "patch": PATCH, "stride": PATCH,
               "n_train": len(train), "n_validation": len(val), "n_test": len(test), "epochs": EPOCHS, "batch_size": BATCH,
               "optimizer": "AdamW", "learning_rate": 1e-4, "weight_decay": 1e-4, "scheduler": "none",
               "gradient_clipping_max_norm": 1.0, "origin_sampling": "one uniformly sampled origin per batch",
               "selected_epoch": selected_epoch, "best_val_avg_mse": best_val, **metrics}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    prov = {"experiment_id": summary["experiment_id"], "seed": seed, "python": sys.version,
            "pytorch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE",
            "code_sha256": sha256(Path(__file__)), "checkpoint_sha256": sha256(out / "checkpoint.pt"),
            "final_checkpoint_sha256": sha256(out / "final_checkpoint.pt")}
    (out / "PROVENANCE.json").write_text(json.dumps(prov, indent=2), encoding="utf8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args(); run(args.seed, args.out)
