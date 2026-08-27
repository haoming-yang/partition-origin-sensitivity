from __future__ import annotations
import json, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from phenomenon_run import DATASETS, ControlledTransformerV2Family, Windows, load_dataset, partition

def check(name, p):
    values, _, _, test, split = load_dataset(name); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=ControlledTransformerV2Family(p,values.shape[1]).to(dev).eval()
    loader=DataLoader(Windows(values,test),batch_size=2,shuffle=False); x,_=next(iter(loader)); x=x.to(dev); rows=[]
    for origin in range(p):
        v,m=partition(x,origin,p); rec=float((v[:,origin:origin+512]-x).abs().max()); token=m.unfold(-1,p,p); valid=token.sum(-1).gt(0); partial=((token.sum(-1)>0)&(token.sum(-1)<p)).sum().item(); full=(token.sum(-1)==0).sum().item()
        v0,_=partition(x,origin,p,0.0); vr=v0.clone(); vh=v0.clone(); rand=torch.randn_like(vr); vr[~m]=rand[~m]; vh[~m]=1000
        with torch.inference_mode(): p0=model(v0,m); pr=model(vr,m); ph=model(vh,m)
        total=((512+2*p-2)//p)*p; rows.append({'origin':origin,'reconstruction_max_abs':rec,'padded_length':total,'tokens':total//p,'padding_slots':total-512,'partial_tokens_per_batch':partial,'fully_masked_tokens_per_batch':full,'sentinel_random_max_abs':float((p0-pr).abs().max()),'sentinel_1000_max_abs':float((p0-ph).abs().max())})
    return {'dataset':name,'p':p,'channels':values.shape[1],'split':split,'rows':rows,'pass':all(r['reconstruction_max_abs']==0 and r['sentinel_random_max_abs']<=1e-6 and r['sentinel_1000_max_abs']<=1e-6 for r in rows)}

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser(); a.add_argument('--dataset',required=True,choices=DATASETS); a.add_argument('--p',required=True,type=int,choices=[8,12,16]); q=a.parse_args(); print(json.dumps(check(q.dataset,q.p)))
