from __future__ import annotations
import argparse, csv, hashlib, json, os, random, sys
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
import stage4_inference as base

EPOCHS,BATCH=15,32

def state_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items(): h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def run(seed,out):
    base.seed_all(seed); values,train,val,test,split=base.load_data('ETTh1'); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=base.ControlledTransformerV2Family(12,values.shape[1]).to(device); initial_hash=state_hash(model); model.eval()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    tl=DataLoader(base.Windows(values,train),batch_size=BATCH,shuffle=True,pin_memory=device.type=='cuda'); vl=DataLoader(base.Windows(values,val),batch_size=BATCH,shuffle=False,pin_memory=device.type=='cuda'); test_loader=DataLoader(base.Windows(values,test),batch_size=BATCH,shuffle=False,pin_memory=device.type=='cuda')
    pf=base.preflight(model,values,vl,12,device); out=Path(out);out.mkdir(parents=True,exist_ok=True); train_rows=[];val_rows=[];best=None;best_val=float('inf');selected=EPOCHS
    for epoch in range(1,EPOCHS+1):
        model.train();total=0.0
        for x,y in tl:
            x,y=x.to(device),y.to(device); v,m=base.partition(x,0,12);opt.zero_grad(set_to_none=True);loss=(model(v,m)-y).square().mean();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();total+=loss.item()*x.shape[0]
        rows=base.origin_metrics(model,vl,device,12);vm=base.metric_from_rows(rows);train_rows.append({'epoch':epoch,'train_loss':total/len(train)});val_rows.append({'epoch':epoch,'val_loss_avg_mse':vm['avg_mse']})
        if vm['avg_mse']<=best_val:best_val=vm['avg_mse'];selected=epoch;best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(json.dumps({'seed':seed,'epoch':epoch,'train_loss':train_rows[-1]['train_loss'],'val_loss_avg_mse':vm['avg_mse']}),flush=True)
    torch.save(model.state_dict(),out/'final_checkpoint.pt');model.load_state_dict(best);torch.save(model.state_dict(),out/'checkpoint.pt');test_rows=base.origin_metrics(model,test_loader,device,12);metrics=base.metric_from_rows(test_rows)
    for fn,rows in [('train_loss.csv',train_rows),('val_loss.csv',val_rows),('per_origin.csv',test_rows)]:
        with (out/fn).open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    cfg={'experiment_id':'CANONICAL_STAGE4_FIXED_ORIGIN_ETTH1_P12_V1','dataset':'ETTh1','p':12,'stride':12,'seed':seed,'context':512,'horizon':96,'epochs':15,'batch_size':32,'optimizer':'AdamW','learning_rate':1e-4,'weight_decay':1e-4,'scheduler':'none','gradient_clipping':1.0,'origin_sampling':'fixed origin r=0 for every training batch','validation':'all 12 origins','checkpoint_rule':'minimum mean validation MSE over all origins','split':split,'model':'Controlled-Transformer-V2'}
    (out/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf8'); summary={'experiment_id':cfg['experiment_id'],'dataset':'ETTh1','p':12,'seed':seed,'n_train':len(train),'n_validation':len(val),'n_test':len(test),'selected_epoch':selected,'best_val_avg_mse':best_val,'initial_parameter_hash':initial_hash,'preflight':pf,**metrics};(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    src=base.source_run('ETTh1',seed,12);prov={'experiment_id':cfg['experiment_id'],'seed':seed,'initial_parameter_hash':initial_hash,'source_random_origin_summary':str(src/'summary.json'),'source_random_origin_checkpoint_sha256':base.sha256(src/'checkpoint.pt'),'runner_sha256':base.sha256(Path(__file__)),'evaluator_sha256':base.sha256(Path(base.__file__)),'checkpoint_sha256':base.sha256(out/'checkpoint.pt'),'final_checkpoint_sha256':base.sha256(out/'final_checkpoint.pt'),'python':sys.version,'pytorch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'};(out/'PROVENANCE.json').write_text(json.dumps(prov,indent=2),encoding='utf8');print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.seed,a.out)
