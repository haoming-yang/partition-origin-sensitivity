import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument("--run",action="store_true",help="Explicitly execute new independent diagnostics")
p.add_argument("--seeds",nargs="+",type=int,default=list(range(10)))
p.add_argument("--output",type=Path,default=Path("reruns"))
a=p.parse_args()
out=(ROOT/a.output).resolve()
if not out.is_relative_to(ROOT) or out in (ROOT, ROOT/"results"):
    raise ValueError("Outputs must stay in a new directory below this package")
config=json.loads((ROOT/"configs/protocol.json").read_text(encoding="utf-8"))
env=os.environ.copy()
for name,rel in {"PYTHONPYCACHEPREFIX":"runtime/pycache","TEMP":"runtime/tmp","TMP":"runtime/tmp","TORCH_HOME":"runtime/torch","HF_HOME":"runtime/hf","XDG_CACHE_HOME":"runtime/cache","CUDA_CACHE_PATH":"runtime/cuda"}.items():
    env[name]=str(ROOT/rel)
env["DEVICE"]="cuda"
commands=[]
for seed in a.seeds:
    for t in config["targets"]:
        target=out/f'{t["dataset"].lower()}_{t["model"]}_seed{seed}'
        if target.exists(): raise FileExistsError(target)
        cmd=[sys.executable,"-B",str(ROOT/"source/tools/analyze_frozen_jacobian.py"),
             "--model",t["model"],"--dataset",t["dataset"],
             "--checkpoint",str(ROOT/t["checkpoint"]),"--data-root",str(ROOT/"data"),
             "--output",str(target),"--seed","42","--projection-seed",str(seed),
             "--origin-a","0","--origin-b","6","--context","512","--horizon","96",
             "--patch-length","12","--stride","12","--batch-size",str(t["batch_size"]),
             "--max-windows","512","--projections","2"]
        commands.append((target,cmd))
for target,cmd in commands:
    if not a.run:
        print(subprocess.list2cmdline(cmd))
    else:
        for name in ["PYTHONPYCACHEPREFIX","TEMP","TORCH_HOME","HF_HOME","XDG_CACHE_HOME","CUDA_CACHE_PATH"]:
            Path(env[name]).mkdir(parents=True,exist_ok=True)
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.with_suffix(".log").open("w",encoding="utf-8") as log:
            subprocess.run(cmd,cwd=ROOT/"source",env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
