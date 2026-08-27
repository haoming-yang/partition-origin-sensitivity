from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from canonical_run import ControlledTransformerV2, Windows, load_data, partition, evaluate, PATCH, CONTEXT

ROOT = Path(__file__).resolve().parents[2] / "outputs" / "canonical_reference"

def load_model(ckpt):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = ControlledTransformerV2().to(dev); m.load_state_dict(torch.load(ckpt, map_location=dev)); m.eval()
    return m, dev

@torch.inference_mode()
def one_process_eval(ckpt, seed):
    _, _, _, test, _, _ = load_data(); values, _, _, _, _, _ = load_data()
    loader = DataLoader(Windows(values, test), batch_size=32, shuffle=False)
    m, dev = load_model(ckpt); rows, metrics = evaluate(m, loader, dev)
    return {"rows": rows, "metrics": metrics, "seed": seed}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--single", action="store_true"); args = ap.parse_args()
    run = ROOT / "runs" / f"seed{args.seed}"; ckpt = run / "checkpoint.pt"
    if args.single:
        print(json.dumps(one_process_eval(ckpt, args.seed)))
        return
    values, _, _, test, _, _ = load_data(); loader = DataLoader(Windows(values, test), batch_size=16, shuffle=False)
    m, dev = load_model(ckpt)
    reconstruction = []
    for origin in range(PATCH):
        x, _ = next(iter(loader)); v, mask = partition(x.to(dev), origin)
        real = v[:, origin:origin+CONTEXT]; original = x.to(dev)
        reconstruction.append({"origin":origin, "max_abs":float((real-original).abs().max()), "real_count":int(mask.sum().item()), "expected_count":x.shape[0]*CONTEXT})
    sentinel_rows = []
    x, _ = next(iter(loader)); x = x.to(dev)
    gen = torch.Generator(device=dev).manual_seed(20260816)
    for origin in range(PATCH):
        v0, mask = partition(x, origin, 0.0)
        vr = v0.clone()
        random_padding = torch.randn(vr.shape, generator=gen, device=dev)
        vr[~mask] = random_padding[~mask]
        vh = v0.clone(); vh[~mask] = 1000.0
        with torch.inference_mode():
            p0, pr, ph = m(v0, mask), m(vr, mask), m(vh, mask)
        sentinel_rows.append({"origin":origin, "max_abs_random":float((p0-pr).abs().max()), "max_abs_1000":float((p0-ph).abs().max())})
    first = one_process_eval(ckpt, args.seed); second = one_process_eval(ckpt, args.seed)
    same_process_max = max(abs(a["MSE"]-b["MSE"]) for a,b in zip(first["rows"], second["rows"]))
    if not args.single:
        child = subprocess.check_output([sys.executable, str(ROOT/"validate.py"), "--seed", str(args.seed), "--single"], text=True)
        external = json.loads(child); external_max = max(abs(a["MSE"]-b["MSE"]) for a,b in zip(first["rows"], external["rows"]))
    else:
        external_max = None
    result = {"experiment_id":"CANONICAL_CONTROLLED_V2", "seed":args.seed,
              "observation_reconstruction":reconstruction, "sentinel":sentinel_rows,
              "evaluator_same_process_max_abs_mse":same_process_max, "evaluator_new_process_max_abs_mse":external_max,
              "tolerance":1e-6,
              "observation_pass":all(r["max_abs"] == 0.0 for r in reconstruction),
              "sentinel_pass":all(r["max_abs_random"] <= 1e-6 and r["max_abs_1000"] <= 1e-6 for r in sentinel_rows),
              "deterministic_pass":same_process_max <= 1e-6 and (external_max is None or external_max <= 1e-6)}
    if not args.single:
        (run / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result))

if __name__ == "__main__": main()
