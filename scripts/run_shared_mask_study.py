#!/usr/bin/env python3
"""Resume diagnostics, a bounded calibration adjustment, and justified validation pilots."""

import csv
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import run_fair_criterion_r0p9 as common


PROJECT=common.PROJECT
ROOT=PROJECT/"runs/shared_mask_repair_study"
VARIANTS=["l1","normalized","mean_repair","worst_repair","random"]
PRIORITY=["resnet50","densenet121","mobilenet_v2","resnet34","googlenet"]


def event(kind,**details):
    entry=dict(time=common.stamp(),kind=kind,**details)
    with (ROOT/"execution.jsonl").open("a",encoding="utf-8") as stream:
        stream.write(json.dumps(entry,ensure_ascii=False)+"\n")
    print(json.dumps(entry,ensure_ascii=False),flush=True)


def config(version,model,**changes):
    return dict(version=version,model=model,seed=0,images=512,patches=1024,candidates=8,iterations=8,calibration_seed=20261101,**changes)


def output(job):
    return ROOT/job["version"]/(job["model"]+"_seed"+str(job["seed"]))


def command(job,mode):
    argv=[sys.executable,"-u","scripts/run_shared_mask_repair.py",mode,"--model",job["model"],"--seed",str(job["seed"]),"--root",str(ROOT/job["version"])]
    for key in ["images","patches","candidates","iterations","calibration_seed"]:
        argv.extend(["--"+key.replace("_","-"),str(job[key])])
    if mode=="pilot":
        argv.extend(["--variant",job["variant"]])
    return argv


def expected(job,mode):
    return output(job)/("done.json" if mode=="diagnose" else "pilots/"+job["variant"]+".json")


def run_batch(jobs,mode):
    pending=[j for j in jobs if not expected(j,mode).exists()]
    active={}
    failed=[]
    try:
        while pending or active:
            for gpu,(job,process,log) in list(active.items()):
                code=process.poll()
                if code is None:
                    continue
                log.close()
                del active[gpu]
                if code or not expected(job,mode).exists():
                    failed.append(job)
                    event("job_failed",mode=mode,job=job,exit_code=code)
                else:
                    event("job_completed",mode=mode,job=job)
            for gpu in [1,0]:
                if not pending or gpu in active:
                    continue
                job=pending.pop(0)
                label=f"{job['version']}_{job['model']}_seed{job['seed']}_{mode}_{job.get('variant','all')}"
                log=(ROOT/"logs"/(label+".log")).open("a",buffering=1)
                environment=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="4",MKL_NUM_THREADS="4",OPENBLAS_NUM_THREADS="4")
                process=subprocess.Popen(command(job,mode),cwd=PROJECT,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[gpu]=(job,process,log)
                event("job_started",gpu=gpu,pid=process.pid,mode=mode,job=job)
            common.write_json(ROOT/"status.json",{"updated_at":common.stamp(),"state":"running","stage":mode,"pending":len(pending),
                             "active":[dict(gpu=gpu,pid=p.pid,job=j) for gpu,(j,p,_) in active.items()],"failed":failed,"controller_pid":os.getpid()})
            if pending or active:
                time.sleep(10)
    finally:
        for job,process,log in active.values():
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
                    process.wait()
            log.close()
    if failed:
        raise RuntimeError(f"{len(failed)} {mode} jobs failed; inspect logs and resume after repair")


def worst_row(job):
    return next(r for r in common.read_csv(output(job)/"summary.csv") if r["variant"]=="worst_repair")


def passes(job):
    row=worst_row(job)
    return float(row["check_worst_improvement_pct"])>=5 and float(row["check_mean_increase_pct"])<=1


def render(jobs):
    records=[]
    pilots=[]
    for job in jobs:
        for row in common.read_csv(output(job)/"summary.csv"):
            records.append(dict(version=job["version"],**row))
        for path in sorted((output(job)/"pilots").glob("*.json")):
            pilots.append(dict(version=job["version"],**json.loads(path.read_text())))
    common.write_csv(ROOT/"diagnostic_summary.csv",records)
    if pilots:
        common.write_csv(ROOT/"pilot_validation.csv",pilots)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for ax,model in zip(axes.flat,common.MODELS):
        matching=[r for r in records if r["model"]==model and r["variant"]=="worst_repair"]
        ax.bar(range(len(matching)),[float(r["check_worst_improvement_pct"]) for r in matching],color="#338b74")
        ax.set_xticks(range(len(matching)),[r["version"] for r in matching])
        ax.axhline(5,color="#a54a61",linestyle="--")
        ax.set_title(model)
        ax.set_ylabel("Held-out worst-error improvement (%)")
    axes.flat[-1].axis("off")
    fig.tight_layout()
    fig.savefig(ROOT/"diagnostic_gates.png",dpi=160)
    plt.close(fig)
    lines=["# Shared-Mask Repair Study", "",f"Updated: {common.stamp()}","",
           "Fixed per-cluster Tm8 kept counts, actual ~90% sparsity. Calibration patches are sampled from train; no test accuracy is used for adjustments.","",
           "## Diagnostics", "", "| Version | Model | Variant | Held-out worst error improvement (%) | Held-out mean error increase (%) | Gate |",
           "|---|---|---|---:|---:|---|"]
    for r in records:
        lines.append(f"| {r['version']} | {r['model']} | {r['variant']} | {float(r['check_worst_improvement_pct']):.3f} | {float(r['check_mean_increase_pct']):.3f} | {r['passes_diagnostic_gate']} |")
    lines += ["", "![Diagnostic gates](diagnostic_gates.png)","", "## Validation-only fine-tuning", "",
              "These are exploratory validation results, not final test or three-seed claims.", "",
              "| Version | Model | Seed | Variant | Direct validation (%) | Final validation (%) |", "|---|---|---:|---|---:|---:|"]
    for r in pilots:
        lines.append(f"| {r['version']} | {r['model']} | {r['seed']} | {r['variant']} | {r['direct_validation_acc']*100:.3f} | {r['final_validation_acc']*100:.3f} |")
    lines += ["", "## Interpretation", "", "A calibration reconstruction improvement does not guarantee classification accuracy. Compare worst_repair against both l1 and mean_repair; report negative results and runtime.",
              "", "Execution, errors, and recovery decisions: execution.jsonl and EXECUTION_LOG.md. Per-layer diagnostics and actual exchanges are in each model directory."]
    (ROOT/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return pilots


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/"logs").mkdir(exist_ok=True)
    lock=(ROOT/"study.lock").open("w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (ROOT/"controller.pid").write_text(str(os.getpid())+"\n",encoding="ascii")
    snapshot=ROOT/"source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in ["shared_mask_repair.py","run_shared_mask_repair.py","run_shared_mask_study.py"]:
        source=PROJECT/"scripts"/name
        frozen=snapshot/name
        if frozen.exists() and common.sha(source)!=common.sha(frozen):
            event("source_revision",file=name,previous_sha256=common.sha(frozen),new_sha256=common.sha(source))
        shutil.copy2(source,frozen)
    jobs=[config("v1",m) for m in ["resnet50","densenet121","resnet34","googlenet","mobilenet_v2"]]
    run_batch(jobs,"diagnose")
    render(jobs)
    adjustments=[]
    for job in jobs[:2]:
        if passes(job):
            continue
        changed=dict(job,version="v2")
        row=worst_row(job)
        if float(row["check_mean_increase_pct"])>1:
            changed.update(images=1024,patches=2048,calibration_seed=20261102)
            reason="held-out mean regression: increase independent calibration sample size"
        else:
            changed.update(candidates=16,iterations=16)
            reason="insufficient worst-error improvement: enlarge bounded search budget"
        event("predeclared_adjustment",model=job["model"],reason=reason,configuration=changed,
              caution="Calibration check data now participates in development selection; final validation/test must be reported separately")
        adjustments.append(changed)
    run_batch(adjustments,"diagnose")
    jobs.extend(adjustments)
    render(jobs)
    selected=[]
    for model in PRIORITY:
        choices=[j for j in jobs if j["model"]==model and passes(j)]
        if choices:
            selected.append(choices[-1])
        if len(selected)==2:
            break
    event("pilot_selection",selected=selected,rule="first two eligible models in predeclared priority; no test-based selection")
    pilot_jobs=[dict(j,variant=v) for j in selected for v in VARIANTS]
    run_batch(pilot_jobs,"pilot")
    pilots=render(jobs)
    conclusions=[]
    for j in selected:
        rows={r["variant"]:r for r in pilots if r["model"]==j["model"] and r["version"]==j["version"]}
        gain=(rows["worst_repair"]["final_validation_acc"]-rows["l1"]["final_validation_acc"])*100
        vs_mean=(rows["worst_repair"]["final_validation_acc"]-rows["mean_repair"]["final_validation_acc"])*100
        conclusions.append(dict(model=j["model"],version=j["version"],validation_gain_vs_l1_pp=gain,validation_gain_vs_mean_pp=vs_mean,
                                next_step="three-seed confirmation justified" if gain>=.1 and vs_mean>=0 else "do not claim a method advantage"))
    result={"state":"complete","completed_at":common.stamp(),"diagnostic_jobs":len(jobs),"pilot_jobs":len(pilot_jobs),
            "conclusions":conclusions,"test_evaluated":False,"meaning":"Evidence-backed diagnostic/pilot result; no forced positive result"}
    common.write_json(ROOT/"study_result.json",result)
    common.write_json(ROOT/"status.json",result)
    event("study_complete",**{k:v for k,v in result.items() if k!="state"})
    archive=ROOT.with_suffix(".zip")
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for path in ROOT.rglob("*"):
            if path.is_file() and path.suffix in {".json",".jsonl",".md",".csv",".png",".py"}:
                z.write(path,path.relative_to(ROOT.parent))
    return 0


if __name__=="__main__":
    try:
        sys.exit(main())
    except Exception as error:
        ROOT.mkdir(parents=True,exist_ok=True)
        common.write_json(ROOT/"status.json",{"state":"error","time":common.stamp(),"error":repr(error)})
        event("controller_error",error=repr(error))
        raise
