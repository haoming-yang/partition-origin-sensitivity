from __future__ import annotations
import csv, hashlib, json, os, random, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ...training.runner import seed_all  # reuse repository seed semantics

ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(REPO_ROOT / "data")))
CONTEXT, HORIZON, EPOCHS, BATCH, CHANNELS_EXPECTED = 512, 96, 15, 32, 7

DATASETS = {
    "ETTh1": {"path": DATA_ROOT / "ETT-small" / "ETTh1.csv", "kind": "ett", "fixed": (8640, 11520, 17420)},
    "Weather": {"path": DATA_ROOT / "weather" / "weather.csv", "kind": "custom"},
    "Electricity": {"path": DATA_ROOT / "electricity" / "electricity.csv", "kind": "custom"},
}


class ControlledTransformerV2Family(nn.Module):
    """Canonical V2 architecture parameterized only by p and channel count."""
    def __init__(self, p, channels):
        super().__init__(); self.p = p; self.channels = channels; self.total = ((CONTEXT + 2*p - 2) // p) * p; self.tokens = self.total // p
        self.position = nn.Parameter(torch.zeros(1, self.tokens, 64))
        self.embed = nn.Linear(2 * p, 64)
        layer = nn.TransformerEncoderLayer(64, 4, 256, 0.1, batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.norm = nn.LayerNorm(64)
        self.head = nn.Sequential(nn.Flatten(start_dim=-2), nn.Dropout(0.1), nn.Linear(self.tokens * 64, HORIZON))

    def forward(self, values, observed):
        safe = values * observed.unsqueeze(-1).to(values.dtype)
        v = safe.permute(0, 2, 1).unfold(-1, self.p, self.p)
        m = observed.to(values.dtype).unfold(-1, self.p, self.p)
        valid = m.sum(-1).gt(0)
        z = torch.cat((v, m.unsqueeze(1).expand(-1, self.channels, -1, -1)), dim=-1)
        z = self.embed(z) + self.position.unsqueeze(1)
        b, c, n, d = z.shape; flat = z.reshape(b*c, n, d)
        key_mask = ~valid[:, None, :].expand(b, c, n).reshape(b*c, n)
        flat = self.encoder(flat, src_key_padding_mask=key_mask)
        flat = self.norm(flat).masked_fill(key_mask.unsqueeze(-1), 0.0)
        return self.head(flat.reshape(b, c, n, d)).transpose(1, 2)


class Windows(Dataset):
    def __init__(self, values, starts): self.values, self.starts = values, starts
    def __len__(self): return len(self.starts)
    def __getitem__(self, i):
        s = int(self.starts[i]); return torch.from_numpy(self.values[s:s+CONTEXT]), torch.from_numpy(self.values[s+CONTEXT:s+CONTEXT+HORIZON])


def partition(x, origin, p, sentinel=0.0):
    total = ((CONTEXT + 2*p - 2) // p) * p
    out = x.new_full((x.shape[0], total, x.shape[-1]), sentinel)
    mask = torch.zeros((x.shape[0], total), dtype=torch.bool, device=x.device)
    out[:, origin:origin+CONTEXT] = x; mask[:, origin:origin+CONTEXT] = True
    return out, mask


def load_dataset(name):
    spec = DATASETS[name]; raw = pd.read_csv(spec["path"])
    values = raw.iloc[:, 1:].astype("float32").to_numpy()
    n, channels = values.shape
    if spec["kind"] == "ett": train_end, val_end, end = spec["fixed"]
    else:
        train_end, test_len = int(n * 0.7), int(n * 0.2); val_end, end = n - test_len, n
    mean = values[:train_end].mean(0, keepdims=True); std = values[:train_end].std(0, keepdims=True)
    values = ((values - mean) / np.where(std < 1e-8, 1, std)).astype("float32")
    train = np.arange(0, train_end - CONTEXT - HORIZON + 1)
    val = np.arange(train_end - CONTEXT, val_end - CONTEXT - HORIZON + 1)
    test = np.arange(val_end - CONTEXT, end - CONTEXT - HORIZON + 1)
    return values, train, val, test, {"rows": n, "channels": channels, "train_rows": [0, train_end], "validation_rows": [train_end, val_end], "test_rows": [val_end, end], "counts": {"train": len(train), "validation": len(val), "test": len(test)}, "scaler_fit_rows": [0, train_end]}


@torch.inference_mode()
def evaluate(model, loader, device, p, sentinel=0.0):
    model.eval(); rows = []
    for origin in range(p):
        se = ae = n = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device); v, m = partition(x, origin, p, sentinel); e = model(v, m) - y
            se += e.square().sum().item(); ae += e.abs().sum().item(); n += e.numel()
        rows.append({"origin": origin, "MSE": se/n, "MAE": ae/n})
    a = np.array([r["MSE"] for r in rows]); interior = a[1:]
    return rows, {"avg_mse": float(a.mean()), "avg_mae": float(np.mean([r["MAE"] for r in rows])), "G_origin": float((a.max()-a.min())/a.min()*100), "CV_origin": float(a.std()/a.mean()), "Delta_origin": float(a.max()-a.min()), "G_interior": float((interior.max()-interior.min())/interior.min()*100), "min_mse_origin": int(a.argmin()), "max_mse_origin": int(a.argmax()), "origin_0_mse": float(a[0]), "origin_last_mse": float(a[-1])}


def sha256(path):
    h = hashlib.sha256();
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""): h.update(b)
    return h.hexdigest()


def run(name, p, seed, out):
    seed_all(seed); values, train, val, test, split = load_dataset(name); channels = values.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model = ControlledTransformerV2Family(p, channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    tl = DataLoader(Windows(values, train), batch_size=BATCH, shuffle=True, pin_memory=device.type == "cuda")
    vl = DataLoader(Windows(values, val), batch_size=BATCH, shuffle=False, pin_memory=device.type == "cuda")
    test_loader = DataLoader(Windows(values, test), batch_size=BATCH, shuffle=False, pin_memory=device.type == "cuda")
    out = Path(out); out.mkdir(parents=True, exist_ok=True); train_rows=[]; val_rows=[]; best=None; best_val=float("inf"); selected=EPOCHS
    for epoch in range(1, EPOCHS+1):
        model.train(); total=0.0
        for x,y in tl:
            x,y=x.to(device),y.to(device); origin=random.randrange(p); v,m=partition(x,origin,p); opt.zero_grad(set_to_none=True)
            loss=(model(v,m)-y).square().mean(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); total += loss.item()*x.shape[0]
        _, vm=evaluate(model,vl,device,p); train_rows.append({"epoch":epoch,"train_loss":total/len(train)}); val_rows.append({"epoch":epoch,"val_loss_avg_mse":vm["avg_mse"]})
        if vm["avg_mse"] <= best_val: best_val=vm["avg_mse"]; selected=epoch; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    torch.save(model.state_dict(),out/"final_checkpoint.pt"); model.load_state_dict(best); torch.save(model.state_dict(),out/"checkpoint.pt")
    rows, metrics=evaluate(model,test_loader,device,p)
    for fn,data in [("train_loss.csv",train_rows),("val_loss.csv",val_rows),("per_origin.csv",rows)]:
        with (out/fn).open("w",newline="",encoding="utf8") as f: w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    cfg={"experiment_id":"CANONICAL_PHENOMENON_27_V1","dataset":name,"p":p,"stride":p,"seed":seed,"context":CONTEXT,"horizon":HORIZON,"epochs":EPOCHS,"batch_size":BATCH,"optimizer":"AdamW","learning_rate":1e-4,"weight_decay":1e-4,"scheduler":"none","gradient_clipping":1.0,"origin_sampling":"one uniformly sampled origin per batch","split":split,"model":"Controlled-Transformer-V2","source_status":"SOURCE_PRESENT"}
    (out/"config.json").write_text(json.dumps(cfg,indent=2),encoding="utf8")
    summary={"experiment_id":"CANONICAL_PHENOMENON_27_V1","dataset":name,"p":p,"stride":p,"seed":seed,"n_train":len(train),"n_validation":len(val),"n_test":len(test),"selected_epoch":selected,"best_val_avg_mse":best_val,**metrics}
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf8")
    total = ((CONTEXT + 2*p - 2) // p) * p
    prov={"experiment_id":"CANONICAL_PHENOMENON_27_V1","dataset":name,"p":p,"seed":seed,"python":sys.version,"pytorch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE","code_sha256":sha256(ROOT/"phenomenon_run.py"),"checkpoint_sha256":sha256(out/"checkpoint.pt"),"config_sha256":sha256(out/"config.json"),"tokens":total//p,"padded_length":total,"padding_slots":total-CONTEXT}
    (out/"PROVENANCE.json").write_text(json.dumps(prov,indent=2),encoding="utf8")
    print(json.dumps(summary),flush=True)


if __name__ == "__main__":
    import argparse
    a=argparse.ArgumentParser(); a.add_argument("--dataset",required=True,choices=DATASETS); a.add_argument("--p",required=True,type=int,choices=[8,12,16]); a.add_argument("--seed",required=True,type=int); a.add_argument("--out",required=True); q=a.parse_args(); run(q.dataset,q.p,q.seed,q.out)
