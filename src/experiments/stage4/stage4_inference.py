from __future__ import annotations

import argparse, csv, hashlib, json, os, random, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

CONTEXT, HORIZON, PATCH, BATCH = 512, 96, 12, 32
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(REPO_ROOT / "data")))
STAGE = Path(os.environ.get("STAGE_ROOT", str(REPO_ROOT / "outputs" / "stage4")))
E_STAGE3 = Path(os.environ.get("STAGE3_ROOT", str(REPO_ROOT / "frozen_artifacts" / "stage3")))
E_KDD = Path(os.environ.get("KDD_ARTIFACT_ROOT", str(REPO_ROOT / "frozen_artifacts" / "canonical")))
C_KDD = E_KDD


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


class ControlledTransformerV2Family(nn.Module):
    def __init__(self, p: int, channels: int):
        super().__init__(); self.p=p; self.channels=channels
        self.total=((CONTEXT+2*p-2)//p)*p; self.tokens=self.total//p
        self.position=nn.Parameter(torch.zeros(1,self.tokens,64))
        self.embed=nn.Linear(2*p,64)
        layer=nn.TransformerEncoderLayer(64,4,256,0.1,batch_first=True,norm_first=False)
        self.encoder=nn.TransformerEncoder(layer,2)
        self.norm=nn.LayerNorm(64)
        self.head=nn.Sequential(nn.Flatten(start_dim=-2),nn.Dropout(0.1),nn.Linear(self.tokens*64,HORIZON))

    def forward(self, values, observed):
        safe=values*observed.unsqueeze(-1).to(values.dtype)
        v=safe.permute(0,2,1).unfold(-1,self.p,self.p)
        m=observed.to(values.dtype).unfold(-1,self.p,self.p)
        valid=m.sum(-1).gt(0)
        z=torch.cat((v,m.unsqueeze(1).expand(-1,self.channels,-1,-1)),dim=-1)
        z=self.embed(z)+self.position.unsqueeze(1)
        b,c,n,d=z.shape; flat=z.reshape(b*c,n,d)
        key_mask=~valid[:,None,:].expand(b,c,n).reshape(b*c,n)
        flat=self.encoder(flat,src_key_padding_mask=key_mask)
        flat=self.norm(flat).masked_fill(key_mask.unsqueeze(-1),0.0)
        return self.head(flat.reshape(b,c,n,d)).transpose(1,2)


class Windows(Dataset):
    def __init__(self, values, starts): self.values,self.starts=values,starts
    def __len__(self): return len(self.starts)
    def __getitem__(self,i):
        s=int(self.starts[i]); return torch.from_numpy(self.values[s:s+CONTEXT]),torch.from_numpy(self.values[s+CONTEXT:s+CONTEXT+HORIZON])


SPECS={
    "ETTh1":{"data":DATA_ROOT/"ETT-small"/"ETTh1.csv","kind":"ett_hour","split":(8640,11520,17420)},
    "ETTh2":{"data":DATA_ROOT/"ETT-small"/"ETTh2.csv","kind":"ett_hour","split":(8640,11520,17420)},
    "ETTm1":{"data":DATA_ROOT/"ETT-small"/"ETTm1.csv","kind":"ett_minute","split":(34560,46080,57600)},
    "ETTm2":{"data":DATA_ROOT/"ETT-small"/"ETTm2.csv","kind":"ett_minute","split":(34560,46080,57600)},
    "Weather":{"data":DATA_ROOT/"weather"/"weather.csv","kind":"custom"},
}


def source_run(dataset, seed, p=12):
    if dataset=="ETTh2": return C_KDD/"low_compute"/"ETTh2_p12"/f"seed{seed}"
    if dataset in ("ETTm1","ETTm2"): return E_STAGE3/dataset.lower()/f"seed{seed}"
    return C_KDD/"runs"/f"{dataset}_p{p}_seed{seed}"


def load_data(dataset):
    spec=SPECS[dataset]; raw=pd.read_csv(spec["data"]).iloc[:,1:].astype("float32").to_numpy()
    n,c=raw.shape
    if spec["kind"]=="custom":
        train_end=int(n*0.7); val_end=n-int(n*0.2); end=n
        borders=(0,train_end,val_end,end)
    else:
        train_end,val_end,end=spec["split"]; borders=(0,train_end,val_end,end)
    mean=raw[:train_end].mean(0,keepdims=True); std=raw[:train_end].std(0,keepdims=True)
    values=((raw-mean)/np.where(std<1e-8,1,std)).astype("float32")
    train=np.arange(0,train_end-CONTEXT-HORIZON+1)
    val=np.arange(train_end-CONTEXT,val_end-CONTEXT-HORIZON+1)
    test=np.arange(val_end-CONTEXT,end-CONTEXT-HORIZON+1)
    assert len(train)>0 and len(val)>0 and len(test)>0
    return values,train,val,test,{"rows":n,"channels":c,"borders":borders,"counts":{"train":len(train),"validation":len(val),"test":len(test)},"scaler_fit_rows":[0,train_end],"data_sha256":sha256(spec["data"])}


def partition(x,origin,p,sentinel=0.0):
    total=((CONTEXT+2*p-2)//p)*p
    out=x.new_full((x.shape[0],total,x.shape[-1]),sentinel); mask=torch.zeros((x.shape[0],total),dtype=torch.bool,device=x.device)
    out[:,origin:origin+CONTEXT]=x; mask[:,origin:origin+CONTEXT]=True
    return out,mask


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available(): torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False


def preflight(model, values, loader, p, device):
    x,y=next(iter(loader)); x=x.to(device)
    v,m=partition(x,0,p); recon=v[:,0:CONTEXT]
    reconstruction_error=float((recon-x).abs().max().item())
    geometry_pass=bool(v.shape[1]==((CONTEXT+2*p-2)//p)*p and m[:,0:CONTEXT].all() and (~m[:,CONTEXT:]).all())
    with torch.inference_mode():
        pred0=model(v,m)
        vr,mr=partition(x,0,p,0.0); vr[~mr]=torch.randn_like(vr[~mr]); pred_random=model(vr,mr)
        vz,mz=partition(x,0,p,1000.0); pred_1000=model(vz,mz)
    d_random=float((pred0-pred_random).abs().max().item()); d_1000=float((pred0-pred_1000).abs().max().item())
    out={"geometry_pass":geometry_pass,"reconstruction_pass":reconstruction_error==0.0,"reconstruction_max_abs":reconstruction_error,"sentinel_random_pass":d_random<=1e-6,"sentinel_plus1000_pass":d_1000<=1e-6,"sentinel_random_max_abs":d_random,"sentinel_plus1000_max_abs":d_1000}
    if not all(out[k] for k in ("geometry_pass","reconstruction_pass","sentinel_random_pass","sentinel_plus1000_pass")): raise RuntimeError(json.dumps(out))
    return out


@torch.inference_mode()
def origin_metrics(model, loader, device, p):
    model.eval(); rows=[]
    for origin in range(p):
        se=ae=n=0
        for x,y in loader:
            x,y=x.to(device),y.to(device); v,m=partition(x,origin,p); e=model(v,m)-y
            se+=e.square().sum().item(); ae+=e.abs().sum().item(); n+=e.numel()
        rows.append({"origin":origin,"MSE":se/n,"MAE":ae/n})
    return rows


def metric_from_rows(rows):
    a=np.array([r["MSE"] for r in rows]); interior=a[1:]; best=a.min()
    return {"avg_mse":float(a.mean()),"avg_mae":float(np.mean([r["MAE"] for r in rows])),"G_origin":float((a.max()-best)/best*100),"CV_origin":float(a.std()/a.mean()),"Delta_origin":float(a.max()-a.min()),"G_interior":float((interior.max()-interior.min())/interior.min()*100),"min_mse_origin":int(a.argmin()),"max_mse_origin":int(a.argmax())}


@torch.inference_mode()
def test_baselines(model, loader, device, p, n_test, channels):
    model.eval(); rows=[]; ensemble=np.zeros((n_test,HORIZON,channels),dtype=np.float64); target=np.zeros_like(ensemble); cursor_by_origin=[]
    for origin in range(p):
        se=ae=n=0; cursor=0
        for x,y in loader:
            x,y=x.to(device),y.to(device); v,m=partition(x,origin,p); pred=model(v,m); e=pred-y
            b=x.shape[0]; se+=e.square().sum().item(); ae+=e.abs().sum().item(); n+=e.numel()
            pn=pred.detach().cpu().numpy(); ensemble[cursor:cursor+b]+=pn
            if origin==0: target[cursor:cursor+b]=y.detach().cpu().numpy()
            cursor+=b
        rows.append({"origin":origin,"MSE":se/n,"MAE":ae/n})
    selected=int(np.argmin(np.array([r["MSE"] for r in rows])))
    mean_metrics=metric_from_rows(rows)
    ensemble/=p; ee=ensemble-target
    return rows,selected,mean_metrics,{"MSE":float(np.mean(ee**2)),"MAE":float(np.mean(np.abs(ee))),"forward_count":p}


def run_baseline(dataset,seed,out,p=12):
    seed_all(seed); values,train,val,test,split=load_data(dataset); src=source_run(dataset,seed,p); ckpt=src/"checkpoint.pt"; summary_path=src/"summary.json"
    if not ckpt.exists() or not summary_path.exists(): raise FileNotFoundError(f"missing trusted source: {ckpt}")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=ControlledTransformerV2Family(p,values.shape[1]).to(device); model.load_state_dict(torch.load(ckpt,map_location=device,weights_only=True)); model.eval()
    vl=DataLoader(Windows(values,val),batch_size=BATCH,shuffle=False,pin_memory=device.type=="cuda"); tl=DataLoader(Windows(values,test),batch_size=BATCH,shuffle=False,pin_memory=device.type=="cuda")
    pf=preflight(model,values,vl,p,device); val_rows=origin_metrics(model,vl,device,p); val_metrics=metric_from_rows(val_rows); rstar=int(np.argmin(np.array([r["MSE"] for r in val_rows])))
    test_rows,test_argmin,mean_metrics,ensemble=test_baselines(model,tl,device,p,len(test),values.shape[1])
    selected=test_rows[rstar]; result={"experiment_id":"CANONICAL_STAGE4_BATCH1_INFERENCE_V1","dataset":dataset,"p":p,"seed":seed,"source_run":str(src),"checkpoint_sha256":sha256(ckpt),"source_summary_sha256":sha256(summary_path),"evaluator_sha256":sha256(Path(__file__)),"split":split,"preflight":pf,"validation":{"r_star":rstar,"selected_origin":val_rows[rstar],"mean_origin":val_metrics,"per_origin":val_rows},"test":{"origin0":test_rows[0],"validation_selected_origin":selected,"test_argmin_origin_record_only":test_argmin,"mean_single_origin":mean_metrics,"ensemble":ensemble,"per_origin":test_rows,"single_origin_forward_count":1,"ensemble_forward_count":p}}
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2),encoding="utf8");print(json.dumps({"dataset":dataset,"seed":seed,"r_star":rstar,"origin0_mse":test_rows[0]["MSE"],"selected_mse":selected["MSE"],"mean_mse":mean_metrics["avg_mse"],"ensemble_mse":ensemble["MSE"]}),flush=True)


def run_dispersion(dataset,seed,out,p=12):
    seed_all(seed); values,train,val,test,split=load_data(dataset); src=source_run(dataset,seed,p); ckpt=src/"checkpoint.pt"; summary_path=src/"summary.json"
    if not ckpt.exists(): raise FileNotFoundError(ckpt)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=ControlledTransformerV2Family(p,values.shape[1]).to(device); model.load_state_dict(torch.load(ckpt,map_location=device,weights_only=True)); model.eval()
    vl=DataLoader(Windows(values,val),batch_size=BATCH,shuffle=False,pin_memory=device.type=="cuda"); tl=DataLoader(Windows(values,test),batch_size=BATCH,shuffle=False,pin_memory=device.type=="cuda"); pf=preflight(model,values,vl,p,device)
    total_s=total_v=total_identity=0.0; count=0; cursor=0
    for x,y in tl:
        x=x.to(device); preds=[]
        for origin in range(p):
            v,m=partition(x,origin,p); preds.append(model(v,m).detach().cpu().numpy())
        z=np.stack(preds,axis=0); mu=z.mean(axis=0); var=((z-mu)**2).mean(axis=(0,2,3)); pair=((z[:,None]-z[None,:])**2).mean(axis=(3,4)); s=pair[~np.eye(p,dtype=bool)].reshape(p*(p-1),-1).mean(axis=0)
        total_s+=float(s.sum()); total_v+=float(var.sum()); total_identity+=float((s-(2*p/(p-1))*var).sum()); count+=x.shape[0]
    smean=total_s/count; vmean=total_v/count; rhs=2*p/(p-1)*vmean; abs_err=abs(smean-rhs); rel_err=abs_err/max(abs(smean),1e-12)
    if abs_err>1e-5: raise RuntimeError(f"S_theta identity failed: {smean} vs {rhs}")
    srcs=json.loads(summary_path.read_text(encoding="utf8")); result={"experiment_id":"CANONICAL_STAGE4_BATCH1_DISPERSION_V1","dataset":dataset,"p":p,"seed":seed,"source_run":str(src),"checkpoint_sha256":sha256(ckpt),"source_summary_sha256":sha256(summary_path),"evaluator_sha256":sha256(Path(__file__)),"split":split,"preflight":pf,"S_theta":smean,"origin_prediction_variance":vmean,"identity_rhs":rhs,"identity_absolute_error":abs_err,"identity_relative_error":rel_err,"G_origin":srcs.get("G_origin",srcs.get("G_origin")),"CV_origin":srcs.get("CV_origin"),"Delta_origin":srcs.get("Delta_origin"),"G_interior":srcs.get("G_interior"),"Avg_MSE":srcs.get("avg_mse")}
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2),encoding="utf8");print(json.dumps({"dataset":dataset,"seed":seed,"S_theta":smean,"identity_abs_error":abs_err}),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--mode",choices=["baseline","dispersion"],required=True);ap.add_argument("--dataset",choices=sorted(SPECS),required=True);ap.add_argument("--seed",type=int,required=True);ap.add_argument("--out",required=True);ap.add_argument("--p",type=int,default=12);a=ap.parse_args()
    (run_baseline if a.mode=="baseline" else run_dispersion)(a.dataset,a.seed,a.out,a.p)
