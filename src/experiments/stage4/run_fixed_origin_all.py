from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path
STAGE=Path(os.environ.get("STAGE_ROOT", str(Path(__file__).resolve().parents[3] / "outputs" / "poc_stage4")))
def main():
 log=STAGE/'logs'/'fixed_origin_all.log'; runner=STAGE/'scripts'/'fixed_origin_runner.py'
 with log.open('w',encoding='utf8') as lf:
  for seed in (42,43,44):
   out=STAGE/'batch1_inference_and_fixed_origin'/'fixed_origin'/f'seed{seed}';cmd=[sys.executable,str(runner),'--seed',str(seed),'--out',str(out)];lf.write(json.dumps({'status':'start','seed':seed})+'\n');lf.flush();p=subprocess.run(cmd,cwd=str(STAGE),capture_output=True,text=True);lf.write(p.stdout);lf.write(p.stderr);lf.write(json.dumps({'status':'complete' if p.returncode==0 else 'failed','seed':seed,'returncode':p.returncode})+'\n');lf.flush();
   if p.returncode!=0: raise SystemExit(p.returncode)
 print('FIXED_ORIGIN_COMPLETE')
if __name__=='__main__':main()
