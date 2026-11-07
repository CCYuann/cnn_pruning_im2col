#!/usr/bin/env python3
"""Finish a bounded self-adjustment and confirm promising candidates on three seeds."""

import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import run_shared_mask_study as study
import run_fair_criterion_r0p9 as common


ROOT=study.ROOT
original_command=study.command


def command(job,mode):
    argv=original_command(job,mode)
    if job["version"]=="v3_mean_safe":
        argv[2]="scripts/run_mean_safe_mask_repair.py"
    return argv


def reuse_patches(job):
    source=ROOT/"v1"/(job["model"]+"_seed"+str(job["seed"]))
    target=study.output(job)
    target.mkdir(parents=True,exist_ok=True)
    for name in ["patches.pt","patch_manifest.json"]:
        if (source/name).exists() and not (target/name).exists():
            os.link(source/name,target/name)


def result(job,variant):
    return json.loads((study.output(job)/"pilots"/(variant+".json")).read_text())


def useful(job):
    values={v:result(job,v)["final_validation_acc"] for v in ["l1","mean_repair","worst_repair"]}
    return values["worst_repair"]-values["l1"]>=.001-1e-12 and values["worst_repair"]>=values["mean_repair"]


def reuse_identical_controls(job):
    import torch
    source=ROOT/"v1"/(job["model"]+"_seed"+str(job["seed"]))
    if not (source/"masks.pt").exists():
        return
    old=torch.load(source/"masks.pt",map_location="cpu",weights_only=False)
    new=torch.load(study.output(job)/"masks.pt",map_location="cpu",weights_only=False)
    output=study.output(job)/"pilots"
    output.mkdir(exist_ok=True)
    for variant in ["l1","normalized"]:
        if not (source/"pilots"/(variant+".json")).exists():
            continue
        if not all(torch.equal(old[variant][name],mask) for name,mask in new[variant].items()):
            raise ValueError("Cannot reuse a control with changed masks")
        for suffix in [".pt",".json"]:
            target=output/(variant+suffix)
            if not target.exists():
                os.link(source/"pilots"/(variant+suffix),target)
        study.event("reused_identical_control",job=job,variant=variant,source=str(source),mask_equality_verified=True)


def final_evaluate(jobs):
    """Evaluate test only after method/configuration is locked for confirmation."""
    import torch
    import resnet34_pruning_experiments as core
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    rows=[]
    for job in jobs:
        model=core.make_model(job["model"],10,32).cuda()
        (_,_,test_loader),_,_=core.build_cifar10(study.PROJECT/"data",128,32,None,None,None,2,True,0,job["seed"],5000)
        for variant in study.VARIANTS:
            checkpoint=study.output(job)/"pilots"/(variant+".pt")
            payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
            model.load_state_dict(payload["model_state_dict"])
            _,accuracy=core.evaluate(model,test_loader,torch.device("cuda"),use_amp="float16")
            rows.append({"version":job["version"],"model":job["model"],"seed":job["seed"],"variant":variant,
                         "test_accuracy":accuracy,"checkpoint":str(checkpoint),"selection":"locked using validation before test"})
        del model
        torch.cuda.empty_cache()
    common.write_csv(ROOT/"confirmation_test.csv",rows)


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    lock=(ROOT/"adjustment.lock").open("w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    common.write_json(ROOT/"adjustment_status.json",{"state":"waiting_for_v1","pid":os.getpid(),"updated_at":common.stamp()})
    while not (ROOT/"study_result.json").exists():
        status=json.loads((ROOT/"status.json").read_text())
        if status.get("state")=="error":
            raise RuntimeError("Resolve the first-stage failure before the adjustment")
        time.sleep(20)
    time.sleep(3)
    study.command=command
    base_jobs=[study.config("v1",m) for m in ["resnet50","densenet121","resnet34","googlenet","mobilenet_v2"]]
    adjustment=[]
    for model in ["resnet50","densenet121"]:
        base=study.config("v1",model)
        if not useful(base):
            job=study.config("v3_mean_safe",model)
            adjustment.append(job)
            reuse_patches(job)
            study.event("mean_safe_adjustment",model=model,reason="v1 fails predeclared validation gain/control comparison",maximum_total_rounds=16,mean_control_rounds=16)
    study.run_batch(adjustment,"diagnose")
    eligible=[job for job in adjustment if study.passes(job)]
    for job in eligible:
        reuse_identical_controls(job)
    study.run_batch([dict(job,variant=v) for job in eligible for v in study.VARIANTS],"pilot")
    all_jobs=base_jobs+adjustment
    study.render(all_jobs)
    promising=[]
    for model in ["resnet50","densenet121"]:
        options=[job for job in all_jobs if job["model"]==model and (job["version"]=="v1" or job in eligible)]
        candidates=[job for job in options if useful(job)]
        if candidates:
            # Prefer the original predeclared version if it already meets the threshold.
            promising.append(candidates[0])
    study.event("confirmation_decision",selected=promising,rule="validation gain >=0.1pp vs l1 and no loss vs matched mean control; never use test to adjust")
    additional=[]
    for job in promising:
        for seed in [1,2]:
            additional.append(dict(job,seed=seed))
    study.run_batch(additional,"diagnose")
    study.run_batch([dict(job,variant=v) for job in additional for v in study.VARIANTS],"pilot")
    all_jobs.extend(additional)
    study.render(all_jobs)
    if promising:
        # Both controllers have drained their GPU jobs before this single-GPU evaluation.
        os.environ["CUDA_VISIBLE_DEVICES"]="1"
        final_evaluate(promising+additional)
    conclusion={"state":"complete","updated_at":common.stamp(),"v3_models":[j["model"] for j in adjustment],
                "eligible_v3_models":[j["model"] for j in eligible],"confirmed_candidates":promising,
                "test_evaluated":bool(promising),"conclusion":"Inspect held-out diagnostics, validation, and any three-seed confirmation without hiding negative versions"}
    common.write_json(ROOT/"adjustment_result.json",conclusion)
    common.write_json(ROOT/"adjustment_status.json",conclusion)
    common.write_json(ROOT/"status.json",conclusion)
    study.event("bounded_adjustment_complete",**{k:v for k,v in conclusion.items() if k!="state"})
    return 0


if __name__=="__main__":
    try:
        sys.exit(main())
    except Exception as error:
        common.write_json(ROOT/"adjustment_status.json",{"state":"error","updated_at":common.stamp(),"error":repr(error)})
        study.event("adjustment_error",error=repr(error))
        raise
