import csv,hashlib,json,statistics
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
for rel,digest in json.loads((ROOT/"FILE_MANIFEST.json").read_text(encoding="utf-8")).items():
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest,rel
config=json.loads((ROOT/"configs/protocol.json").read_text(encoding="utf-8"))
stored=json.loads((ROOT/"statistics/ten_seed_metrics.json").read_text(encoding="utf-8"))
for t in config["targets"]:
    windows=[]; differences=[]
    for seed in config["projection_seeds"]:
        d=ROOT/f'results/{t["dataset"].lower()}_{t["model"]}_seed{seed}'
        s=json.loads((d/"summary.json").read_text(encoding="utf-8"))
        assert s["checkpoint_sha256"]==t["checkpoint_sha256"]
        assert s["projections"]==2 and s["windows"]==512 and s["projection_seed"]==seed
        trace=ROOT/s["projection_trace"]
        assert hashlib.sha256(trace.read_bytes()).hexdigest()==s["projection_trace_sha256"]
        with np.load(trace,allow_pickle=False) as z:
            assert z["signs"].shape[:3]==(512,2,96)
            assert set(np.unique(z["signs"]).tolist())=={-1,1}
            assert hashlib.sha256(z["initial_generator_state"].tobytes()).hexdigest()==s["initial_generator_state_sha256"]
            windows.append(z["window_start_rows"].copy())
        q=np.genfromtxt(d/"jacobian_profile.csv",delimiter=",",names=True)
        def mass(k): return float(100*q[k][-16:].sum()/q[k].sum())
        differences.append(mass("origin_a_jacobian_energy")-mass("origin_b_jacobian_energy"))
    assert all(np.array_equal(windows[0],x) for x in windows)
    agg=next(x for x in stored["aggregate_summary"] if x["dataset"]==t["dataset"] and x["model"]==t["model"] and x["metric"]=="origin0_minus_origin6_pp")
    for value,key in [(statistics.mean(differences),"mean"),(statistics.stdev(differences),"sample_sd"),(min(differences),"min"),(max(differences),"max")]:
        assert abs(value-agg[key])<1e-10,(t["dataset"],key)
    assert len({x>0 for x in differences})==1
    print(t["dataset"],t["model"],"paired mean/SD/min/max",agg["mean"],agg["sample_sd"],agg["min"],agg["max"])
print("PASS: all file hashes, 40 saved traces, fixed input IDs, and paired summary statistics verified without model/GPU execution")
