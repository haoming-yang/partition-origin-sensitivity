from __future__ import annotations
import argparse, csv, json, os, random, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

CONTEXT,HORIZON,PATCH,STRIDE,TOTAL,TOKENS,D=512,96,12,12,528,44,64
_DATA_ROOT = Path(os.environ.get("PARTITION_ORIGIN_DATA_ROOT", Path(__file__).resolve().parents[2] / "data"))
DATA={"ETTh1":_DATA_ROOT / "ETT-small" / "ETTh1.csv", "Weather":_DATA_ROOT / "weather" / "weather.csv"}

class Windows(Dataset):
 def __init__(self,v,s): self.v,self.s=v,s
 def __len__(self): return len(self.s)
 def __getitem__(self,i):
  s=int(self.s[i]); return torch.from_numpy(self.v[s:s+CONTEXT]),torch.from_numpy(self.v[s+CONTEXT:s+CONTEXT+HORIZON])

class SharedTokenizer(nn.Module):
 def __init__(self,c):
  super().__init__(); self.c=c; self.position=nn.Parameter(torch.zeros(1,TOKENS,D)); self.embed=nn.Linear(2*PATCH,D)
 def forward(self,x,origin,sentinel=0.0):
  left,right=origin,TOTAL-CONTEXT-origin; vals=x.new_full((x.shape[0],TOTAL,x.shape[2]),sentinel); obs=torch.zeros((x.shape[0],TOTAL),dtype=torch.bool,device=x.device); vals[:,left:left+CONTEXT]=x; obs[:,left:left+CONTEXT]=True
  safe=vals*obs.unsqueeze(-1).to(vals.dtype); v=safe.permute(0,2,1).unfold(-1,PATCH,STRIDE); m=obs.to(vals.dtype).unfold(-1,PATCH,STRIDE); valid=m.sum(-1).gt(0); z=torch.cat((v,m.unsqueeze(1).expand(-1,self.c,-1,-1)),dim=-1); z=self.embed(z)+self.position.unsqueeze(1); return z,valid

class Transformer(nn.Module):
 def __init__(self,c):
  super().__init__(); self.tok=SharedTokenizer(c); layer=nn.TransformerEncoderLayer(D,4,256,0.1,batch_first=True,norm_first=False); self.back=nn.TransformerEncoder(layer,2); self.norm=nn.LayerNorm(D); self.head=nn.Linear(TOKENS*D,HORIZON)
 def forward(self,x,origin):
  z,valid=self.tok(x,origin); b,c,n,d=z.shape; km=~valid[:,None,:].expand(b,c,n).reshape(b*c,n); z=self.back(z.reshape(b*c,n,d),src_key_padding_mask=km); z=self.norm(z).masked_fill(km.unsqueeze(-1),0); return self.head(z.reshape(b,c,n,d).flatten(-2)).transpose(1,2)

class MLP(nn.Module):
 def __init__(self,c):
  super().__init__(); self.tok=SharedTokenizer(c); self.blocks=nn.ModuleList([nn.ModuleList([nn.LayerNorm(D),nn.Linear(TOKENS,TOKENS),nn.LayerNorm(D),nn.Linear(D,4*D),nn.GELU(),nn.Linear(4*D,D)]) for _ in range(2)]); self.head=nn.Linear(TOKENS*D,HORIZON)
 def forward(self,x,origin):
  z,valid=self.tok(x,origin); mask=valid[:,None,:,None];
  for ln1,tm,ln2,fc,g,proj in self.blocks:
   q=ln1(z).transpose(-1,-2); z=z+tm(q).transpose(-1,-2); z=z.masked_fill(~mask,0); z=z+proj(g(fc(ln2(z)))); z=z.masked_fill(~mask,0)
  return self.head(z.flatten(-2)).transpose(1,2)

class Conv(nn.Module):
 def __init__(self,c):
  super().__init__(); self.tok=SharedTokenizer(c); self.blocks=nn.ModuleList([nn.ModuleList([nn.LayerNorm(D),nn.Conv1d(D,D,3,padding=1),nn.Conv1d(D,D,1),nn.GELU(),nn.LayerNorm(D),nn.Linear(D,4*D),nn.GELU(),nn.Linear(4*D,D)]) for _ in range(2)]); self.head=nn.Linear(TOKENS*D,HORIZON)
 def forward(self,x,origin):
  z,valid=self.tok(x,origin); mask=valid[:,None,:,None]
  for ln,conv,pw,g,ln2,fc,g2,proj in self.blocks:
   q=ln(z).reshape(-1,TOKENS,D).transpose(-1,-2); q=conv(q).transpose(-1,-2); q=pw(g(q).transpose(-1,-2)).transpose(-1,-2).reshape(z.shape); z=z+q; z=z.masked_fill(~mask,0); z=z+proj(g2(fc(ln2(z)))); z=z.masked_fill(~mask,0)
  return self.head(z.flatten(-2)).transpose(1,2)

def load(name):
 raw=pd.read_csv(DATA[name]).iloc[:,1:].astype('float32').to_numpy(); n=len(raw); tr,va,end=(8640,11520,14400) if name=='ETTh1' else (int(n*.7),int(n*.8),n); mean=raw[:tr].mean(0,keepdims=True); std=raw[:tr].std(0,keepdims=True); v=((raw-mean)/np.where(std<1e-8,1,std)).astype('float32'); return v,np.arange(0,tr-CONTEXT-HORIZON+1)[:1024],np.arange(tr-CONTEXT,va-CONTEXT-HORIZON+1)[:16],np.arange(va-CONTEXT,end-CONTEXT-HORIZON+1)[:16]

@torch.inference_mode()
def evaluate(model,loader,device):
 model.eval(); rows=[]; preds=[]; target=None
 for r in range(PATCH):
  se=ae=n=0; pp=[]
  for x,y in loader:
   x,y=x.to(device),y.to(device); p=model(x,r); e=p-y; se+=e.square().sum().item(); ae+=e.abs().sum().item(); n+=e.numel(); pp.append(p.cpu().numpy()); target=y.cpu().numpy() if target is None else target
  preds.append(np.concatenate(pp)); rows.append({'origin':r,'MSE':se/n,'MAE':ae/n})
 p=np.stack(preds); a=np.array([x['MSE'] for x in rows]); ens=float(np.mean((p.mean(0)-target)**2)); return rows,{'MSEmean':float(a.mean()),'MSE0':float(a[0]),'MSEens':ens,'G_origin':float((a.max()-a.min())/a.min()*100),'CV_origin':float(a.std()/a.mean()),'Delta_origin':float(a.max()-a.min()),'G_interior':float((a[1:].max()-a[1:].min())/a[1:].min()*100),'S_theta':float(np.mean((p-p.mean(0))**2))},p,target

def run(model_name,dataset,seed,out):
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); v,tr,va,te=load(dataset); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model={'Transformer':Transformer,'MLP':MLP,'Conv':Conv}[model_name](v.shape[1]).to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4); tl=DataLoader(Windows(v,tr),32,shuffle=True,pin_memory=device.type=='cuda'); vl=DataLoader(Windows(v,va),32,False); tel=DataLoader(Windows(v,te),32,False); out=Path(out); out.mkdir(parents=True,exist_ok=True); hist=[]; best=None; bv=1e9; selected=0; start=time.perf_counter(); peak=0
 for ep in range(1,6):
  model.train(); total=0
  for x,y in tl:
   x,y=x.to(device),y.to(device); r=random.randrange(PATCH); opt.zero_grad(set_to_none=True); loss=(model(x,r)-y).square().mean(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); total+=loss.item()*x.shape[0]; peak=max(peak,torch.cuda.max_memory_allocated() if device.type=='cuda' else 0)
  _,vm,_,_=evaluate(model,vl,device); hist.append({'epoch':ep,'train_loss':total/len(tr),'val_MSEmean':vm['MSEmean']});
  if vm['MSEmean']<=bv: bv=vm['MSEmean']; selected=ep; best={k:x.detach().cpu().clone() for k,x in model.state_dict().items()}
 model.load_state_dict(best); vr,vm,vp,vt=evaluate(model,vl,device); rstar=int(np.argmin([x['MSE'] for x in vr])); trr,tm,tp,tt=evaluate(model,tel,device); tm['MSEsel']=float(trr[rstar]['MSE']); torch.save({'state_dict':model.state_dict(),'config':{'experiment_id':f'CONTROLLED_{model_name.upper()}_{dataset}_P12_S{seed}','model':model_name,'dataset':dataset,'L':CONTEXT,'H':HORIZON,'p':PATCH,'stride':STRIDE,'epochs':5,'seed':seed}},out/'checkpoint.pt');
 with (out/'training_curve.csv').open('w',newline='',encoding='utf8') as f: w=csv.DictWriter(f,fieldnames=hist[0].keys()); w.writeheader(); w.writerows(hist)
 np.savez_compressed(out/'validation_predictions.npz',predictions=vp,targets=vt); np.savez_compressed(out/'test_predictions.npz',predictions=tp,targets=tt); (out/'per_origin_validation.json').write_text(json.dumps(vr,indent=2)); (out/'per_origin_test.json').write_text(json.dumps(trr,indent=2)); (out/'metrics.json').write_text(json.dumps(tm,indent=2)); (out/'validation_metrics.json').write_text(json.dumps(vm,indent=2)); (out/'run_meta.json').write_text(json.dumps({'experiment_id':f'CONTROLLED_{model_name.upper()}_{dataset}_P12_S{seed}','model':model_name,'dataset':dataset,'seed':seed,'selected_epoch':selected,'best_validation_mean_origin_mse':bv,'parameter_count':sum(x.numel() for x in model.parameters()),'wall_seconds':time.perf_counter()-start,'peak_vram_bytes':peak,'device':str(device)},indent=2))
