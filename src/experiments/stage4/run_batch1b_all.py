from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path

STAGE=Path(os.environ.get("STAGE_ROOT", str(Path(__file__).resolve().parents[3] / "outputs" / "stage4")))
RUNNER=STAGE/"scripts"/"stage4_inference.py"
JOBS=[(d,s) for d in ("ETTh1","ETTh2","ETTm1","ETTm2","Weather") for s in (42,43,44)]

def main():
    log=STAGE/"logs"/"batch1b_all.log"
    with log.open('w',encoding='utf8') as lf:
        for d,s in JOBS:
            out=STAGE/"batch1_inference_and_fixed_origin"/"dispersion_analysis"/d/f"seed{s}.json"
            if out.exists(): continue
            cmd=[sys.executable,str(RUNNER),'--mode','dispersion','--dataset',d,'--seed',str(s),'--out',str(out)]
            lf.write(json.dumps({'status':'start','dataset':d,'seed':s})+'\n');lf.flush()
            p=subprocess.run(cmd,cwd=str(STAGE),capture_output=True,text=True)
            lf.write(p.stdout);lf.write(p.stderr);lf.write(json.dumps({'status':'complete' if p.returncode==0 else 'failed','dataset':d,'seed':s,'returncode':p.returncode})+'\n');lf.flush()
            if p.returncode!=0: raise SystemExit(p.returncode)
    print('BATCH1B_COMPLETE')
if __name__=='__main__':main()
