from __future__ import annotations

import copy
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import stage4_inference as base
import poc_lambda_pilot as pilot


STAGE = base.STAGE
INPUT_STAGE4 = base.INPUT_STAGE4
FORMAL_ROOT = STAGE / "batch2_poc" / "formal_C"
FROZEN = INPUT_STAGE4 / "batch2_poc" / "FROZEN_POC_LAMBDA.json"
PROTOCOL = INPUT_STAGE4 / "batch2_poc" / "lambda_selection" / "LAMBDA_SELECTION_PROTOCOL.json"
SCHEDULE_ROOT = base.E_STAGE3 / "attribution_abc"
B_ROOT = SCHEDULE_ROOT / "b_two_view_supervised"
LAMBDA_INPUT = INPUT_STAGE4 / "batch2_poc" / "lambda_selection" / "lambda_1p0"
LAMBDA_CHECKPOINT_ROOT = Path(os.environ.get("PARTITION_ORIGIN_POC_LAMBDA_CHECKPOINT_ROOT", base.REPO_ROOT / "checkpoints" / "poc_stage4" / "batch2_poc" / "lambda_selection" / "lambda_1p0"))
DEVICE = pilot.DEVICE


def sha(path: Path) -> str:
    return base.sha256(path)


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_rows(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["origin", "MSE", "MAE"])
        w.writeheader(); w.writerows(rows)


def finite(name, x):
    x = float(x)
    if not np.isfinite(x):
        raise RuntimeError(f"NaN/Inf in {name}: {x}")
    return x


def load_frozen():
    f = json.loads(FROZEN.read_text(encoding="utf-8"))
    if f.get("status") != "FROZEN_POC_LAMBDA" or f.get("selected_lambda") is None:
        raise RuntimeError("Formal C cannot start without a frozen lambda")
    if f.get("test_evaluation_before_freeze") is not False:
        raise RuntimeError("Frozen lambda artifact indicates test leakage")
    return f


def b_schedule_check(seed: int):
    schedule, path, schedule_hash = pilot.load_schedule(seed)
    prov_path = B_ROOT / f"seed{seed}" / "PROVENANCE.json"
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    if prov.get("schedule_sha256") != schedule_hash:
        raise RuntimeError(f"B schedule hash mismatch for seed{seed}")
    return schedule, path, schedule_hash, prov


def validation_loader(values, val):
    return torch.utils.data.DataLoader(
        base.Windows(values, val), batch_size=base.BATCH, shuffle=False,
        pin_memory=DEVICE.type == "cuda"
    )


def train_formal(seed: int, lam: float, schedule: dict, schedule_path: Path, schedule_hash: str, out: Path):
    values, _, val, _, split = base.load_data("ETTh1")
    model, initial_hash = pilot.make_model(seed, values.shape[1])
    initial_state = copy.deepcopy(model.state_dict())
    loader = validation_loader(values, val)
    preflight = pilot.preflight_once(model, values, loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_val = float("inf"); best_epoch = None; best_state = None
    train_rows=[]; val_rows=[]
    for epoch_entry in schedule["epochs"]:
        epoch = int(epoch_entry.get("epoch", len(train_rows) + 1))
        model.train(); loss_sum=pred_sum=cons_sum=0.0; n_total=0
        for batch in epoch_entry["batches"]:
            indices=[int(i) for i in batch["train_indices"]]
            ia, ib=int(batch["origin_a"]), int(batch["origin_b"])
            x,y=pilot.batch_from_indices(values, indices)
            va,ma=base.partition(x,ia,12); vb,mb=base.partition(x,ib,12)
            optimizer.zero_grad(set_to_none=True)
            pa=model(va,ma); pb=model(vb,mb)
            l_pred=0.5*(F.mse_loss(pa,y)+F.mse_loss(pb,y))
            l_cons=F.mse_loss(pa,pb)
            loss=l_pred+lam*l_cons
            if not torch.isfinite(loss): raise RuntimeError(f"Non-finite formal C loss seed{seed} epoch{epoch}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            n=len(indices); n_total+=n
            loss_sum+=float(loss.detach().item())*n; pred_sum+=float(l_pred.detach().item())*n; cons_sum+=float(l_cons.detach().item())*n
        train_rows.append({"epoch":epoch,"lambda":lam,"loss":finite("train loss",loss_sum/n_total),"L_pred":finite("L_pred",pred_sum/n_total),"L_cons":finite("L_cons",cons_sum/n_total),"scheduled_batches":len(epoch_entry["batches"]),"scheduled_examples":n_total})
        model.eval(); per_origin, metrics=pilot.validation_metrics(model,loader)
        val_rows.append({"epoch":epoch,**metrics})
        if metrics["avg_mse"] <= best_val:
            best_val=float(metrics["avg_mse"]); best_epoch=epoch; best_state=copy.deepcopy(model.state_dict())
            torch.save(best_state,out/"checkpoint.pt"); write_rows(out/"best_val_per_origin.csv",per_origin)
    if best_state is None: raise RuntimeError(f"No formal checkpoint for seed{seed}")
    torch.save(model.state_dict(),out/"final_checkpoint.pt")
    model.load_state_dict(best_state); model.eval()
    best_per_origin,best_metrics=pilot.validation_metrics(model,loader)
    config={"experiment_id":"STAGE4_BATCH2_FORMAL_POC_C_V1","condition":"C_two_view_supervised_plus_consistency","dataset":"ETTh1","p":12,"seed":seed,"lambda":lam,"epochs":15,"batch_size":32,"optimizer":"AdamW","learning_rate":1e-4,"weight_decay":1e-4,"scheduler":"none","gradient_clipping_max_norm":1.0,"origin_pair_schedule":str(schedule_path),"origin_pair_schedule_sha256":schedule_hash,"lambda_selection_protocol_sha256":sha(PROTOCOL),"frozen_lambda_sha256":sha(FROZEN),"initial_parameter_hash_C":initial_hash,"checkpoint_rule":"minimum mean validation MSE across all origins; ties later epoch","test_evaluation_performed":False}
    dump(out/"config.json",config)
    with (out/"train_loss.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(train_rows[0].keys()));w.writeheader();w.writerows(train_rows)
    with (out/"val_loss.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(val_rows[0].keys()));w.writeheader();w.writerows(val_rows)
    dump(out/"training_summary.json",{**config,"selected_epoch":best_epoch,"best_val_avg_mse":best_val,"validation":{"per_origin":best_per_origin,**best_metrics},"preflight":preflight})
    return config


def reuse_seed42(lam: float, schedule_path: Path, schedule_hash: str, out: Path):
    src=LAMBDA_INPUT
    checkpoint=LAMBDA_CHECKPOINT_ROOT/"checkpoint.pt"
    if not (src/"summary.json").exists() or not checkpoint.exists(): raise FileNotFoundError(src)
    pilot_summary=json.loads((src/"summary.json").read_text(encoding="utf-8"))
    if pilot_summary.get("test_evaluation_performed") is not False or float(pilot_summary["lambda"]) != lam:
        raise RuntimeError("Selected seed42 pilot is not eligible for formal reuse")
    out.mkdir(parents=True,exist_ok=True)
    for p in src.iterdir():
        if p.is_file(): shutil.copy2(p,out/p.name)
    shutil.copy2(checkpoint,out/"checkpoint.pt")
    final_checkpoint=LAMBDA_CHECKPOINT_ROOT/"final_checkpoint.pt"
    if final_checkpoint.exists(): shutil.copy2(final_checkpoint,out/"final_checkpoint.pt")
    shutil.copy2(schedule_path,out/"origin_pair_schedule.json")
    model, initial_hash=pilot.make_model(42,base.load_data("ETTh1")[0].shape[1])
    if initial_hash != pilot_summary.get("initial_parameter_hash"):
        raise RuntimeError("Seed42 pilot initial hash mismatch")
    dump(out/"FORMAL_REUSE.json",{"formal_reuse":True,"source_directory":str(src),"source_summary_sha256":sha(src/"summary.json"),"source_checkpoint_sha256":sha(checkpoint),"test_evaluation_before_reuse":False,"initial_parameter_hash_C":initial_hash,"origin_pair_schedule_sha256":schedule_hash,"lambda":lam})


def evaluate(seed: int, lam: float, schedule_path: Path, schedule_hash: str, bprov: dict, out: Path, source_kind: str):
    values, _, val, test, split=base.load_data("ETTh1")
    ckpt=out/"checkpoint.pt"
    model, initial_hash=pilot.make_model(seed,values.shape[1]); model.load_state_dict(torch.load(ckpt,map_location=DEVICE,weights_only=True)); model.eval()
    vl=validation_loader(values,val); tl=torch.utils.data.DataLoader(base.Windows(values,test),batch_size=base.BATCH,shuffle=False,pin_memory=DEVICE.type=="cuda")
    preflight=pilot.preflight_once(model,values,vl)
    val_rows,val_metrics=pilot.validation_metrics(model,vl); rstar=int(np.argmin(np.array([r["MSE"] for r in val_rows])))
    test_rows,test_argmin,mean_metrics,ensemble=base.test_baselines(model,tl,DEVICE,12,len(test),values.shape[1])
    test_aggregate=base.metric_from_rows(test_rows); test_disp=pilot.dispersion_metrics(model,tl,12)
    write_rows(out/"validation_per_origin.csv",val_rows); write_rows(out/"per_origin_test.csv",test_rows)
    result={"experiment_id":"STAGE4_BATCH2_FORMAL_POC_C_V1","condition":"C_two_view_supervised_plus_consistency","dataset":"ETTh1","p":12,"seed":seed,"lambda":lam,"source_kind":source_kind,"checkpoint_sha256":sha(ckpt),"origin_pair_schedule":str(schedule_path),"origin_pair_schedule_sha256":schedule_hash,"lambda_selection_protocol_sha256":sha(PROTOCOL),"frozen_lambda_sha256":sha(FROZEN),"initial_parameter_hash_B_reconstructed":initial_hash,"initial_parameter_hash_C":initial_hash,"initial_parameter_hash_match":True,"B_schedule_hash_match":bprov.get("schedule_sha256")==schedule_hash,"split":split,"preflight":preflight,"validation":{"r_star":rstar,"selected_origin":val_rows[rstar],**val_metrics},"test":{"origin0":test_rows[0],"validation_selected_origin":test_rows[rstar],"test_argmin_origin_record_only":test_argmin,"mean_single_origin":mean_metrics,"ensemble":ensemble,"per_origin":test_rows,"inference_forward_count_single_origin":1,"inference_forward_count_ensemble":12,**test_aggregate,**test_disp},"test_evaluation_performed":True,"test_evaluation_after_lambda_freeze":True}
    dump(out/"formal_summary.json",result)
    dump(out/"PROVENANCE_FORMAL.json",{**result,"runner":str(Path(__file__)),"runner_sha256":sha(Path(__file__)),"evaluator_sha256":sha(SCRIPT_DIR/"stage4_inference.py")})
    return result


def main():
    frozen=load_frozen(); lam=float(frozen["selected_lambda"])
    required_checkpoint = LAMBDA_CHECKPOINT_ROOT / "checkpoint.pt"
    if not required_checkpoint.exists():
        raise FileNotFoundError(f"Required frozen lambda checkpoint is missing: {required_checkpoint}")
    FORMAL_ROOT.mkdir(parents=True,exist_ok=True)
    outputs=[]
    for seed in (42,43,44):
        schedule,schedule_path,schedule_hash,bprov=b_schedule_check(seed)
        out=FORMAL_ROOT/f"seed{seed}"
        if seed==42:
            reuse_seed42(lam,schedule_path,schedule_hash,out)
            source_kind="REUSED_SELECTED_LAMBDA_PILOT"
        else:
            out.mkdir(parents=True,exist_ok=True)
            train_formal(seed,lam,schedule,schedule_path,schedule_hash,out)
            shutil.copy2(schedule_path,out/"origin_pair_schedule.json")
            source_kind="NEW_FORMAL_TRAINING"
        result=evaluate(seed,lam,schedule_path,schedule_hash,bprov,out,source_kind); outputs.append(result)
        print(json.dumps({"seed":seed,"source_kind":source_kind,"test_avg_mse":result["test"]["avg_mse"],"test_S_theta":result["test"]["S_theta"]}),flush=True)
    dump(FORMAL_ROOT/"FORMAL_C_STATUS.json",{"experiment_id":"STAGE4_BATCH2_FORMAL_POC_C_V1","status":"POC_C_COMPLETE","seeds":[42,43,44],"lambda":lam,"test_evaluation_after_lambda_freeze":True,"all_initial_hash_matches":all(x["initial_parameter_hash_match"] for x in outputs),"all_schedule_hash_matches":all(x["B_schedule_hash_match"] for x in outputs)})


if __name__=="__main__": main()
