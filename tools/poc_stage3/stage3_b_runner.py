from __future__ import annotations

import argparse, csv, json, random, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn

import stage3_longer_runner as base

EPOCHS, BATCH = 15, 32


def make_schedule(seed, path):
    values, train, _, _ = base.load_data()
    g = torch.Generator().manual_seed(seed + 700000)
    r = random.Random(seed + 800000)
    epochs = []
    for epoch in range(1, EPOCHS + 1):
        order = torch.randperm(len(train), generator=g).tolist()
        batches = []
        for batch_id, start in enumerate(range(0, len(order), BATCH)):
            idx = order[start:start + BATCH]
            a = r.randrange(base.PATCH); b = r.randrange(base.PATCH - 1)
            if b >= a: b += 1
            batches.append({"batch_id": batch_id, "train_indices": idx, "origin_a": a, "origin_b": b})
        epochs.append({"epoch": epoch, "batches": batches})
    obj = {"experiment_id": "CANONICAL_STAGE3_ABC_ATTRIBUTION_ETTH1_P12_V1", "seed": seed,
           "schedule_type": "pre_generated_two_distinct_origins_per_batch", "epochs": epochs}
    Path(path).write_text(json.dumps(obj), encoding="utf8")


def run(seed, out, schedule_path):
    base.seed_all(seed)
    values, train, val, test = base.load_data()
    schedule_path = Path(schedule_path)
    if not schedule_path.exists(): make_schedule(seed, schedule_path)
    schedule = json.loads(schedule_path.read_text(encoding="utf8"))
    assert schedule["seed"] == seed and len(schedule["epochs"]) == EPOCHS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = base.ControlledTransformerV2().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    vl = torch.utils.data.DataLoader(base.Windows(values, val), batch_size=BATCH, shuffle=False, pin_memory=device.type == "cuda")
    test_loader = torch.utils.data.DataLoader(base.Windows(values, test), batch_size=BATCH, shuffle=False, pin_memory=device.type == "cuda")
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    train_rows, val_rows, best_state, best_val, selected_epoch = [], [], None, float("inf"), EPOCHS
    for ep in schedule["epochs"]:
        model.train(); total = 0.0; count = 0
        for item in ep["batches"]:
            idx = np.asarray(item["train_indices"], dtype=np.int64)
            starts = train[idx]
            x = torch.from_numpy(np.stack([values[s:s + base.CONTEXT] for s in starts])).to(device)
            y = torch.from_numpy(np.stack([values[s + base.CONTEXT:s + base.CONTEXT + base.HORIZON] for s in starts])).to(device)
            v1, m1 = base.partition(x, int(item["origin_a"])); v2, m2 = base.partition(x, int(item["origin_b"]))
            opt.zero_grad(set_to_none=True)
            loss = ((model(v1, m1) - y).square().mean() + (model(v2, m2) - y).square().mean()) / 2
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += loss.item() * x.shape[0]; count += x.shape[0]
        _, vm = base.evaluate(model, vl, device); val_loss = vm["avg_mse"]
        train_loss = total / count
        train_rows.append({"epoch": ep["epoch"], "train_loss": train_loss}); val_rows.append({"epoch": ep["epoch"], "val_loss_avg_mse": val_loss})
        if val_loss <= best_val:
            best_val, selected_epoch = val_loss, ep["epoch"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(json.dumps({"seed": seed, "epoch": ep["epoch"], "train_loss": train_loss, "val_loss_avg_mse": val_loss}), flush=True)
    torch.save(model.state_dict(), out / "final_checkpoint.pt")
    model.load_state_dict(best_state); torch.save(model.state_dict(), out / "checkpoint.pt")
    test_rows, metrics = base.evaluate(model, test_loader, device)
    for name, rows in [("train_loss.csv", train_rows), ("val_loss.csv", val_rows), ("per_origin.csv", test_rows)]:
        with (out / name).open("w", newline="", encoding="utf8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary = {"experiment_id": "CANONICAL_STAGE3_ABC_ATTRIBUTION_ETTH1_P12_V1", "condition": "B_two_view_supervised", "model": "Controlled-Transformer-V2", "seed": seed,
               "dataset": "ETTh1", "context": base.CONTEXT, "horizon": base.HORIZON, "patch": base.PATCH, "stride": base.PATCH,
               "n_train": len(train), "n_validation": len(val), "n_test": len(test), "epochs": EPOCHS, "batch_size": BATCH,
               "optimizer": "AdamW", "learning_rate": 1e-4, "weight_decay": 1e-4, "scheduler": "none", "gradient_clipping_max_norm": 1.0,
               "training_views": 2, "origin_pair_schedule": str(schedule_path), "loss": "mean supervised MSE over two distinct origins; no consistency term",
               "selected_epoch": selected_epoch, "best_val_avg_mse": best_val, **metrics}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    prov = {"experiment_id": summary["experiment_id"], "condition": summary["condition"], "seed": seed, "python": sys.version,
            "pytorch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE",
            "runner": str(Path(__file__)), "schedule_sha256": base.sha256(schedule_path), "checkpoint_sha256": base.sha256(out / "checkpoint.pt"),
            "final_checkpoint_sha256": base.sha256(out / "final_checkpoint.pt")}
    (out / "PROVENANCE.json").write_text(json.dumps(prov, indent=2), encoding="utf8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--seed", type=int, required=True); p.add_argument("--out", required=True); p.add_argument("--schedule", required=True)
    a = p.parse_args(); run(a.seed, a.out, a.schedule)
