#!/usr/bin/env python3
"""Cheap controls and continuous FT30 confirmation for fixed shared-mask repair."""

import argparse
import copy
import json
import math
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import torch

import weight_compensation_worker as previous
from adapted_gradient_mask import restore_rng, rng_state
from scalar_weight_compensation import compensate_scalar
from task_loss_mask_search import recalibrate_bn

common=previous.common
core=previous.core
PROJECT=previous.PROJECT
OLD=previous.ROOT
ROOT=PROJECT/"runs/compensation_controls_ft30"
PLAN=PROJECT/"research_notes/2026-11-05_compensation_controls_plan.md"
PRIMARY=["A","BN","scalar","B","C","dense_extra"]
REFERENCES=["global_unstructured","quota_unstructured"]
VARIANTS=PRIMARY+REFERENCES
REUSED={"A":"original","B":"dense_input","C":"sequential"}


def event(directory,kind,**details):
    value=dict(time=common.stamp(),kind=kind,**details)
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/"journal.jsonl").open("a",encoding="utf-8") as stream:
        stream.write(json.dumps(value,ensure_ascii=False)+"\n")
    print(json.dumps(value,ensure_ascii=False),flush=True)


def old_cache(seed):
    return OLD/"calibration"/f"seed{seed}"


def prepared(seed):
    return ROOT/"prepared"/f"seed{seed}"


def source_files():
    return [Path(__file__).resolve(),PROJECT/"scripts/run_compensation_controls.py",PLAN,
        PROJECT/"scripts/scalar_weight_compensation.py",PROJECT/"tests/test_scalar_weight_compensation.py",
        PROJECT/"tests/test_compensation_controls.py",common.CORE,
        *[PROJECT/"scripts"/name for name in ["weight_compensation_worker.py","run_fair_criterion_r0p9.py",
        "adapted_gradient_mask.py","run_shared_mask_repair.py","task_loss_mask_search.py","fixed_support_ridge.py"]]]


def fingerprint(seed):
    cache=old_cache(seed)
    old_inputs=[cache/"configuration.json",cache/"ready.json",cache/"masks.pt",cache/"images.pt",cache/"data_indices.json"]
    old_inputs += [OLD/f"seed{seed}_{variant}"/name for variant in REUSED.values()
                   for name in ["configuration.json","initialization.pt","initial_validation.json","result.json","training_state.pt"]]
    return dict(seed=seed,model="resnet50",split_id="42fbb7440cc27c71",checkpoint_sha256=common.sha(common.checkpoint("resnet50",seed)),
        sources={str(p.relative_to(PROJECT)):common.sha(p) for p in source_files()},
        old_inputs={str(p.relative_to(PROJECT)):common.sha(p) for p in old_inputs},
        tm=8,nominal_sparsity=.9,ft_rng=seed+100000,lr=.001,momentum=.9,weight_decay=.0005,batch_size=128,
        bn_policy="normal training BN; fit512 reset only for BN branch",scalar="per-filter unregularized LS plus shared-cluster check fallback",
        final_primary_epochs=30,reference_epochs=10,test_evaluated=False)


def freeze(seed):
    directory=prepared(seed)
    directory.mkdir(parents=True,exist_ok=True)
    previous.freeze(directory/"configuration.json",fingerprint(seed))


def old_validate(seed,full_cache=False):
    cache=old_cache(seed)
    if json.loads((cache/"configuration.json").read_text())!=previous.fingerprint(SimpleNamespace(seed=seed)):
        raise ValueError("Original compensation source or checkpoint changed")
    ready=json.loads((cache/"ready.json").read_text())
    if common.sha(cache/"masks.pt")!=ready["mask_sha256"] or common.sha(cache/"images.pt")!=ready["image_sha256"]:
        raise ValueError("Original masks/images changed")
    split=json.loads((cache/"data_indices.json").read_text())
    selected=split["fit"]+split["check"]
    validation=set(torch.randperm(50000,generator=torch.Generator().manual_seed(0))[:5000].tolist())
    if len(set(selected))!=1024 or set(selected)&validation:
        raise ValueError("Calibration data isolation failed")
    if full_cache:
        for record in ready["cache_files"]:
            if common.sha(cache/record["file"])!=record["sha256"]:
                raise ValueError("Teacher patch cache changed: "+record["file"])
    return ready


def expand_shared(masks,specs):
    return {s.name:masks[s.name].repeat_interleave(8,0)[:s.module.out_channels].reshape_as(s.module.weight).bool().cpu() for s in specs}


@torch.no_grad()
def quota_mask(weights,counts):
    flat=weights.flatten(1)
    if len(counts)!=len(flat) or bool((counts<0).any()) or bool((counts>flat.shape[1]).any()):
        raise ValueError("Invalid per-filter quotas")
    order=torch.argsort(flat.abs(),dim=1,descending=True,stable=True)
    out=torch.zeros_like(flat,dtype=torch.bool)
    out.scatter_(1,order,torch.arange(flat.shape[1],device=weights.device)[None,:]<counts[:,None])
    return out.reshape_as(weights)


def install(model,specs,masks):
    if masks is None:
        return None
    stats=[]
    with torch.no_grad():
        for spec in specs:
            full=masks[spec.name].to(spec.module.weight.device)
            if full.shape!=spec.module.weight.shape or full.dtype!=torch.bool:
                raise ValueError("Invalid full mask")
            spec.module.weight.mul_(full)
            stats.append(SimpleNamespace(name=spec.name,full_mask=full,masks=[]))
    return stats


@torch.no_grad()
def prepare(seed,device):
    freeze(seed)
    directory=prepared(seed)
    if (directory/"ready.json").exists():
        return
    ready=old_validate(seed,full_cache=True)
    model,specs,payload=previous.load_model(seed,device)
    if payload["epochs"]!=100:
        raise ValueError("Expected independent Dense100 checkpoint")
    prepare_started=time.monotonic()
    grouped=torch.load(old_cache(seed)/"masks.pt",map_location="cpu",weights_only=False)
    shared=expand_shared(grouped,specs)
    quota={}
    for spec in specs:
        count=shared[spec.name].flatten(1).sum(1).to(device)
        quota[spec.name]=quota_mask(spec.module.weight,count).cpu()
    all_weights=torch.cat([s.module.weight.detach().abs().flatten() for s in specs])
    order=torch.argsort(all_weights,descending=True,stable=True)
    global_flat=torch.zeros_like(all_weights,dtype=torch.bool)
    global_flat[order[:ready["kept_weights"]]]=True
    global_masks={}
    offset=0
    for spec in specs:
        count=spec.module.weight.numel()
        global_masks[spec.name]=global_flat[offset:offset+count].reshape_as(spec.module.weight).cpu()
        offset+=count
    del all_weights,order,global_flat
    for name,masks in [("shared",shared),("quota",quota),("global",global_masks)]:
        torch.save(masks,directory/f"{name}_masks.pt")
    reference_mask_seconds=time.monotonic()-prepare_started
    rows,cluster_rows=[],[]
    for spec in specs:
        hw=core.output_hw(spec.input_hw,spec.module)
        row=dict(layer=spec.name,shape=list(spec.module.weight.shape),kernel=list(spec.module.kernel_size),
            stride=list(spec.module.stride),is_first=spec.name=="conv1",is_downsample="downsample" in spec.name,
            total=spec.module.weight.numel(),output_hw=list(hw),q_min=int(grouped[spec.name].sum(1).min()),q_max=int(grouped[spec.name].sum(1).max()))
        for name,masks in [("shared",shared),("quota",quota),("global",global_masks)]:
            row[name+"_kept"]=int(masks[spec.name].sum())
            row[name+"_macs"]=int(masks[spec.name].sum())*math.prod(hw)
        rows.append(row)
        for i,q in enumerate(grouped[spec.name].sum(1).tolist()):
            cluster_rows.append(dict(seed=seed,layer=spec.name,cluster=i,filters=8,k=grouped[spec.name].shape[1],q=q))
    common.write_csv(directory/"layers.csv",rows)
    common.write_csv(directory/"clusters.csv",cluster_rows)
    initial_a=torch.load(OLD/f"seed{seed}_original/initialization.pt",map_location=device,weights_only=False)["model"]
    model.load_state_dict(initial_a)
    scalar_rows=[]
    scalar_started=time.monotonic()
    for spec,record in zip(specs,ready["layers"]):
        if spec.name!=record["name"]:
            raise ValueError("Layer/cache alignment mismatch")
        data=torch.load(old_cache(seed)/record["file"],map_location="cpu",weights_only=False)
        fit_x,check_x=data["fit"]["x"].to(device),data["check"]["x"].to(device)
        fit_y,check_y=data["fit"]["y"].to(device),data["check"]["y"].to(device)
        dense=payload["model_state_dict"][spec.name+".weight"].to(device)
        for cluster,mask in enumerate(grouped[spec.name]):
            start,end=cluster*8,(cluster+1)*8
            value,report=compensate_scalar(dense[start:end].flatten(1),mask.to(device),fit_x,fit_y[start:end],check_x,check_y[start:end])
            spec.module.weight[start:end].copy_(value.reshape_as(spec.module.weight[start:end]))
            report["alpha_json"]=json.dumps(report.pop("alpha"))
            scalar_rows.append(dict(layer=spec.name,cluster=cluster,**report))
        event(directory,"scalar_layer",layer=spec.name)
    scalar_seconds=time.monotonic()-scalar_started
    conv_keys={s.name+".weight" for s in specs}
    for name,value in model.state_dict().items():
        if name not in conv_keys and not torch.equal(value,initial_a[name]):
            raise ValueError("Scalar changed BN or nonconv parameter")
    torch.save(dict(model=model.state_dict()),directory/"scalar_initialization.pt")
    common.write_csv(directory/"scalar_reconstruction.csv",scalar_rows)
    model.load_state_dict(initial_a)
    fit_images=torch.load(old_cache(seed)/"images.pt",map_location="cpu",weights_only=False)["fit"].to(device)
    bn_started=time.monotonic()
    recalibrate_bn(model,(fit_images,None))
    bn_seconds=time.monotonic()-bn_started
    allowed={n for n,_ in model.named_buffers() if n.endswith(("running_mean","running_var","num_batches_tracked"))}
    for name,value in model.state_dict().items():
        if name not in allowed and not torch.equal(value,initial_a[name]):
            raise ValueError("BN calibration changed weights/affine")
    torch.save(dict(model=model.state_dict()),directory/"bn_initialization.pt")
    shutil.copy2(old_cache(seed)/"data_indices.json",directory/"data_indices.json")
    files=["shared_masks.pt","quota_masks.pt","global_masks.pt","scalar_initialization.pt","bn_initialization.pt","scalar_reconstruction.csv","layers.csv","clusters.csv","data_indices.json"]
    totals={name:dict(kept=sum(r[name+"_kept"] for r in rows),macs=sum(r[name+"_macs"] for r in rows)) for name in ["shared","quota","global"]}
    total=sum(s.module.weight.numel() for s in specs)
    if any(v["kept"]!=ready["kept_weights"] for v in totals.values()) or totals["shared"]!=totals["quota"]:
        raise ValueError("Prepared budgets mismatch")
    for spec in specs:
        if not torch.equal(shared[spec.name].flatten(1).sum(1),quota[spec.name].flatten(1).sum(1)):
            raise ValueError("Independent-filter q mismatch")
    common.write_json(directory/"ready.json",dict(seed=seed,total_weights=total,totals=totals,
        scalar_seconds=scalar_seconds,bn_seconds=bn_seconds,scalar_fallback=sum(r["status"]=="fallback" for r in scalar_rows),
        reference_mask_seconds=reference_mask_seconds,checkpoint_id=payload["checkpoint_id"],dense_epochs=payload["epochs"],
        dense_validation_accuracy=payload["dense_val_acc"],dense_training_config=payload["training_config"],split_metadata=payload["split_metadata"],
        scalar_accepted=sum(r["status"]=="accepted" for r in scalar_rows),
        dense_macs=sum(r["total"]*math.prod(r["output_hw"]) for r in rows),
        files={name:common.sha(directory/name) for name in files},test_evaluated=False))
    event(directory,"prepare_complete",scalar_seconds=scalar_seconds,bn_seconds=bn_seconds,totals=totals)


def branch_masks(seed,variant):
    if variant=="dense_extra":
        return None
    name="global" if variant=="global_unstructured" else "quota" if variant=="quota_unstructured" else "shared"
    return torch.load(prepared(seed)/f"{name}_masks.pt",map_location="cpu",weights_only=False)


def load_initial(model,seed,variant,device):
    if variant in REUSED:
        path=OLD/f"seed{seed}_{REUSED[variant]}/initialization.pt"
    elif variant in ["BN","scalar"]:
        path=prepared(seed)/("bn_initialization.pt" if variant=="BN" else "scalar_initialization.pt")
    else:
        return
    model.load_state_dict(torch.load(path,map_location=device,weights_only=False)["model"])


def check_ready(seed):
    freeze(seed)
    value=json.loads((prepared(seed)/"ready.json").read_text())
    for name,digest in value["files"].items():
        if common.sha(prepared(seed)/name)!=digest:
            raise ValueError("Prepared input changed: "+name)
    return value


def train(seed,variant,target_epochs,device):
    if target_epochs not in [10,30] or (variant in REFERENCES and target_epochs!=10):
        raise ValueError("Invalid training budget")
    ready=check_ready(seed)
    directory=ROOT/f"seed{seed}_{variant}"
    directory.mkdir(parents=True,exist_ok=True)
    previous.freeze(directory/"configuration.json",dict(seed=seed,variant=variant,
        preparation_sha256=common.sha(prepared(seed)/"ready.json"),sources=fingerprint(seed)["sources"],
        max_epochs=30 if variant in PRIMARY else 10,ft_rng=seed+100000,test_evaluated=False))
    model,specs,payload=previous.load_model(seed,device)
    load_initial(model,seed,variant,device)
    masks=branch_masks(seed,variant)
    stats=install(model,specs,masks)
    del payload
    (train_loader,val_loader,_unused_test),_,metadata=core.build_cifar10(PROJECT/"data",128,32,None,None,None,2,True,0,seed,5000)
    if metadata["split_id"]!="42fbb7440cc27c71":
        raise ValueError("Wrong data split")
    diagnostic=directory/"initial_validation.json"
    if not diagnostic.exists() and variant in REUSED:
        shutil.copy2(OLD/f"seed{seed}_{REUSED[variant]}/initial_validation.json",diagnostic)
    if diagnostic.exists():
        initial=json.loads(diagnostic.read_text())
    else:
        loss,accuracy=core.evaluate(model,val_loader,device,use_amp="float16")
        clone=copy.deepcopy(model)
        images=torch.load(old_cache(seed)/"images.pt",map_location="cpu",weights_only=False)["fit"].to(device)
        recalibrate_bn(clone,(images,None))
        bn_loss,bn_accuracy=core.evaluate(clone,val_loader,device,use_amp="float16")
        del clone,images
        if not math.isfinite(loss) or not math.isfinite(bn_loss):
            raise FloatingPointError("Non-finite FT0")
        initial=dict(raw_loss=loss,raw_accuracy=accuracy,bn_clone_loss=bn_loss,bn_clone_accuracy=bn_accuracy)
        common.write_json(diagnostic,initial)
        event(directory,"initial_validation",**initial)
    optimizer=torch.optim.SGD(model.parameters(),lr=.001,momentum=.9,weight_decay=.0005)
    scaler=core.make_scaler(device,"float16")
    core.reset_training_rng(seed+100000,train_loader)
    epoch,history,elapsed=0,[],0.
    resume=directory/"training_state.pt"
    resume_source=resume if resume.exists() else OLD/f"seed{seed}_{REUSED[variant]}/training_state.pt" if variant in REUSED else None
    if resume_source is not None:
        saved=torch.load(resume_source,map_location=device,weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        restore_rng(saved["rng"],train_loader)
        epoch,history=saved["epoch"],saved["history"]
        elapsed=saved.get("elapsed",json.loads((OLD/f"seed{seed}_{REUSED[variant]}/result.json").read_text())["training_seconds"] if variant in REUSED else 0.)
        event(directory,"resume_training",epoch=epoch,source=str(resume_source.relative_to(PROJECT)),optimizer_rng_preserved=True)
        if variant in REUSED and epoch==10:
            old_result=json.loads((OLD/f"seed{seed}_{REUSED[variant]}/result.json").read_text())
            if history[-1]["validation_accuracy"]!=old_result["ft10_accuracy"]:
                raise ValueError("Original epoch10 history/result mismatch")
            before_probe=rng_state(train_loader)
            probe_loss,probe_accuracy=core.evaluate(model,val_loader,device,use_amp="float16")
            restore_rng(before_probe,train_loader)
            if abs(probe_accuracy-old_result["ft10_accuracy"])>1e-8:
                raise ValueError("Historical epoch10 validation cannot be reproduced")
            event(directory,"historical_epoch10_reproduced",accuracy=probe_accuracy,loss=probe_loss,rng_restored=True)
    started=time.monotonic()
    for epoch in range(epoch+1,target_epochs+1):
        train_loss,train_acc=core.train_one_epoch(model,train_loader,optimizer,device,str(epoch),prune_stats=stats,tm=8,scaler=scaler,use_amp="float16")
        val_loss,val_acc=core.evaluate(model,val_loader,device,use_amp="float16")
        if not math.isfinite(val_loss):
            raise FloatingPointError("Non-finite validation")
        row=dict(epoch=epoch,train_loss=train_loss,train_accuracy=train_acc,validation_loss=val_loss,validation_accuracy=val_acc)
        history.append(row)
        event(directory,"epoch_complete",**row)
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),rng=rng_state(train_loader),
            epoch=epoch,history=history,elapsed=elapsed+time.monotonic()-started),directory/"training_state.tmp")
        (directory/"training_state.tmp").replace(resume)
        if epoch==10 and variant in PRIMARY:
            shutil.copy2(resume,directory/"epoch10_state.pt")
    if stats:
        for spec,stat in zip(specs,stats):
            if torch.count_nonzero(spec.module.weight.detach()[stat.full_mask==0]):
                raise ValueError("Forbidden connection regrew")
    common.write_csv(directory/"training_history.csv",history)
    mask_name="global" if variant=="global_unstructured" else "quota" if variant=="quota_unstructured" else "shared"
    kept=ready["total_weights"] if variant=="dense_extra" else ready["totals"][mask_name]["kept"]
    macs=ready["dense_macs"] if variant=="dense_extra" else ready["totals"][mask_name]["macs"]
    result=dict(seed=seed,variant=variant,epochs=target_epochs,ft0_accuracy=initial["raw_accuracy"],ft0_bn_clone_accuracy=initial["bn_clone_accuracy"],
        history=history,kept_weights=kept,total_conv_weights=ready["total_weights"],kept_macs=macs,sparsity=1-kept/ready["total_weights"],
        training_seconds=elapsed+time.monotonic()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        split_id=metadata["split_id"],test_evaluated=False)
    for budget in [1,3,10,30]:
        result[f"ft{budget}_accuracy"]=next((r["validation_accuracy"] for r in history if r["epoch"]==budget),None)
    common.write_json(directory/f"ft{target_epochs}_result.json",result)
    common.write_json(directory/"result.json",result)
    torch.save(dict(model_state_dict={n:v.detach().cpu() for n,v in model.state_dict().items()},seed=seed,variant=variant),directory/"final_model.pt")
    event(directory,"worker_complete",epochs=target_epochs,accuracy=history[-1]["validation_accuracy"],training_seconds=result["training_seconds"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["prepare","train"])
    parser.add_argument("--seed",type=int,choices=range(3),required=True)
    parser.add_argument("--variant",choices=VARIANTS,default="A")
    parser.add_argument("--epochs",type=int,default=30)
    args=parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    core.reset_training_rng(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    device=torch.device("cuda")
    if args.mode=="prepare":
        prepare(args.seed,device)
    else:
        train(args.seed,args.variant,args.epochs,device)


if __name__=="__main__":
    main()
