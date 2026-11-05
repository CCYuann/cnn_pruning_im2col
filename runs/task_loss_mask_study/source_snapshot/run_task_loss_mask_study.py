#!/usr/bin/env python3
"""Run and resume a bounded task-loss shared-mask study with frozen decision rules."""

import fcntl
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import run_fair_criterion_r0p9 as common

PROJECT = common.PROJECT
ROOT = PROJECT / "runs/task_loss_mask_study"


def event(kind, **details):
    record = dict(time=common.stamp(), kind=kind, **details)
    with (ROOT / "execution.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)


def directory(job):
    return ROOT / (job["bn"] + "_" + job["policy"]) / f"resnet50_seed{job['seed']}"


def expected(job, mode):
    if mode == "prepare":
        return PROJECT / "runs/shared_mask_repair_study/v1" / f"resnet50_seed{job['seed']}" / "done.json"
    return directory(job) / ("search.json" if mode == "search" else "pilots/loss_gate.json")


def command(job, mode):
    if mode == "prepare":
        return [sys.executable, "-u", "scripts/run_shared_mask_repair.py", "diagnose", "--model", "resnet50",
                "--seed", str(job["seed"])]
    return [sys.executable, "-u", "scripts/task_loss_mask_search.py", mode, "--model", "resnet50", "--seed", str(job["seed"]),
            "--bn", job["bn"], "--policy", job["policy"], "--root", str(ROOT)]


def run_batch(jobs, mode):
    pending = [j for j in jobs if not expected(j, mode).exists()]
    active = {}
    try:
        while pending or active:
            for gpu, (job, process, log) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del active[gpu]
                if code or not expected(job, mode).exists():
                    raise RuntimeError(f"Worker failed: {job}, mode={mode}, exit={code}")
                event("job_complete", job=job, mode=mode, gpu=gpu)
            for gpu in [1, 0]:
                if not pending or gpu in active:
                    continue
                usage = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
                if int(usage.strip()) < 12000:
                    continue
                job = pending.pop(0)
                label = f"{job['bn']}_{job['policy']}_seed{job['seed']}_{mode}"
                log = (ROOT / "logs" / (label + ".log")).open("a", buffering=1)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
                process = subprocess.Popen(command(job, mode), cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[gpu] = (job, process, log)
                event("job_start", job=job, mode=mode, gpu=gpu, pid=process.pid)
            common.write_json(ROOT / "status.json", dict(state="running", stage=mode, updated_at=common.stamp(),
                              pending=len(pending), active=[dict(gpu=g, pid=p.pid, job=j) for g, (j, p, _) in active.items()]))
            if pending or active:
                time.sleep(10)
    finally:
        for _, process, log in active.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            log.close()


def control(seed, variant):
    source = PROJECT / "runs/shared_mask_repair_study/v1" / f"resnet50_seed{seed}" / "pilots" / (variant + ".json")
    return json.loads(source.read_text())


def ensure_controls(jobs):
    # Use the original pilot worker with original masks and its unchanged recipe.
    for seed in sorted({j["seed"] for j in jobs}):
        for variant in ["l1", "worst_repair"]:
            source = PROJECT / "runs/shared_mask_repair_study/v1" / f"resnet50_seed{seed}" / "pilots" / (variant + ".json")
            if not source.exists():
                env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
                with (ROOT / "logs" / f"control_seed{seed}_{variant}.log").open("a") as log:
                    subprocess.run([sys.executable, "-u", "scripts/run_shared_mask_repair.py", "pilot", "--model", "resnet50",
                                    "--seed", str(seed), "--variant", variant], cwd=PROJECT, env=env,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            target = ROOT / "controls" / f"seed{seed}_{variant}.json"
            shutil.copy2(source, target)
            event("control_available", seed=seed, variant=variant, source=str(source), sha256=common.sha(source))


def useful(job):
    row = json.loads(expected(job, "pilot").read_text())
    score = row["final_validation_acc"]
    return score - control(job["seed"], "l1")["final_validation_acc"] >= .001 - 1e-12 and score >= control(job["seed"], "worst_repair")["final_validation_acc"]


def results(jobs):
    records = []
    for seed in sorted({j["seed"] for j in jobs}):
        for variant in ["l1", "worst_repair"]:
            r = control(seed, variant)
            records.append(dict(version="control", policy=variant, seed=seed, accuracy=r["final_validation_acc"],
                                loss=r["validation_loss"], direct_accuracy=r["direct_validation_acc"],
                                sparsity=r["sparsity"], accepted_layers="", audit_ce_ratio=""))
    diagnostics = []
    for job in jobs:
        s = json.loads(expected(job, "search").read_text())
        p = json.loads(expected(job, "pilot").read_text())
        records.append(dict(version=job["bn"], policy=job["policy"], seed=job["seed"], accuracy=p["final_validation_acc"],
                            loss=p["validation_loss"], direct_accuracy=p["direct_validation_acc"], sparsity=p["sparsity"],
                            accepted_layers=s["accepted_layers"], audit_ce_ratio=s["after"]["audit"]["loss"] / s["before"]["audit"]["loss"]))
        for name in ["search", "check", "audit"]:
            diagnostics.append(dict(version=job["bn"], policy=job["policy"], seed=job["seed"], split=name,
                                    before_ce=s["before"][name]["loss"], after_ce=s["after"][name]["loss"],
                                    before_accuracy=s["before"][name]["accuracy"], after_accuracy=s["after"][name]["accuracy"]))
    for row in records:
        row["gain_vs_l1_pp"] = (row["accuracy"] - control(row["seed"], "l1")["final_validation_acc"]) * 100
    common.write_csv(ROOT / "validation_results.csv", records)
    common.write_csv(ROOT / "proxy_diagnostics.csv", diagnostics)
    return records, diagnostics


def verify(jobs):
    import torch
    checks = []
    for job in jobs:
        out = directory(job)
        masks = torch.load(out / "masks.pt", map_location="cpu", weights_only=False)
        indices = json.loads((out / "data_indices.json").read_text())
        search = json.loads((out / "search.json").read_text())
        validation = set(torch.randperm(50000, generator=torch.Generator().manual_seed(0))[:5000].tolist())
        subsets = [set(indices[name]) for name in ["search", "check", "audit"]]
        if any(subsets[a] & subsets[b] for a, b in [(0, 1), (0, 2), (1, 2)]) or any(s & validation for s in subsets):
            raise ValueError("Data isolation failed")
        original = torch.load(PROJECT / "runs/shared_mask_repair_study/v1" / f"resnet50_seed{job['seed']}" / "masks.pt", map_location="cpu", weights_only=False)["l1"]
        for name, mask in masks["loss_gate"].items():
            if not torch.equal(mask.sum(1), original[name].sum(1)) or not torch.equal(masks["l1"][name], original[name]):
                raise ValueError("Mask quota or baseline equality failed")
        for d in search["decisions"]:
            if d["accepted"]:
                if d["after_search_loss"] >= d["before_search_loss"] - 1e-5:
                    raise ValueError("Accepted a search-loss regression")
                if job["policy"] == "dual" and d["after_check_loss"] >= d["before_check_loss"] - 1e-5:
                    raise ValueError("Accepted a check-loss regression")
        checks.append(dict(job=job, quotas=True, original_l1_identical=True, data_isolation=True, acceptance_trace=True,
                           masks_sha256=common.sha(out / "masks.pt")))
    common.write_json(ROOT / "VALIDATION.json", dict(checks=checks, passed=True, test_evaluated=False))


def finish(jobs, selected):
    rows, diagnostics = results(jobs)
    verify(jobs)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels = [f"{r['version']}\n{r['policy']}\ns{r['seed']}" for r in rows]
    axes[0].bar(range(len(rows)), [r["gain_vs_l1_pp"] for r in rows], color=["#77828b" if r["version"] == "control" else "#298577" for r in rows])
    axes[0].set_xticks(range(len(rows)), labels, fontsize=8)
    axes[0].axhline(0, color="#333333", linewidth=.8)
    axes[0].set_ylabel("Validation accuracy gain vs same-seed l1 (pp)")
    audit = [r for r in diagnostics if r["split"] == "audit"]
    axes[1].bar(range(len(audit)), [r["after_ce"] / r["before_ce"] for r in audit], color="#aa6575")
    axes[1].set_xticks(range(len(audit)), [f"{r['version']}\n{r['policy']}\ns{r['seed']}" for r in audit], fontsize=8)
    axes[1].axhline(1, color="#333333", linestyle="--")
    axes[1].set_ylabel("Untouched train audit CE after / before search")
    fig.tight_layout()
    fig.savefig(ROOT / "comparison.png", dpi=170)
    plt.close(fig)
    lines = ["# 全模型分類Loss篩選共享Mask：實驗結果", "", f"整理：{common.stamp()}", "",
             "## 固定條件", "", "ResNet50、CIFAR-10、Tm8連續分組、實際約90%剪枝。每cluster kept_count逐元素匹配l1；只換mask，不重分配剪枝預算。每分支相同Dense checkpoint與10epoch fine-tune，搜尋BN統計不帶入訓練。", "",
             "search 512 / check 512 / audit 1024均來自互斥train；dual使用check做否決，故check不是獨立測試。audit僅搜尋開始及結束量測；validation5000張用於開發決策。沒有使用test。", "",
             "候選來自已固定的mean/worst局部重建mask；逐層在目前完整剪枝網路比較CE，一次選一個候選，保留其他已接受的mask。search_only是消融，主方法固定為dual。", "",
             "## 分類結果", "", "| BN proxy | Policy | Seed | Direct val (%) | 10ep val (%) | vs l1 (pp) | Accepted layers |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['version']} | {r['policy']} | {r['seed']} | {r['direct_accuracy']*100:.3f} | {r['accuracy']*100:.3f} | {r['gain_vs_l1_pp']:+.3f} | {r['accepted_layers']} |")
    lines += ["", "![分類與獨立audit比較](comparison.png)", "", "## 判斷", ""]
    if selected:
        primary = [r for r in rows if r["version"] == selected["bn"] and r["policy"] == "dual"]
        gains = [r["gain_vs_l1_pp"] for r in primary]
        lines += [f"單seed pilot通過預設條件，已鎖定{selected['bn']}/dual並補seeds1/2。三seed validation平均差異{statistics.mean(gains):+.3f}pp，sample std {statistics.stdev(gains):.3f}pp。所有seed都保留；這仍是validation確認，不是獨立test結論，也不代表其他模型有效。"]
    else:
        lines += ["frozen與BN重新校正兩種proxy均未讓dual達到預設的l1提升至少0.1pp、且不低於無條件worst_repair的條件，因此未擴大三seed。保留負結果，不從search_only或多次試驗中改挑贏家。"]
    lines += ["", "分類proxy改善不等於fine-tune後accuracy改善；局部候選庫仍受Dense teacher輸入限制。若結果不佳，可能是候選庫或未訓練模型proxy不適合預測可訓練性，但這些原因尚未被分離證實。", "",
              "## 重現與限制", "", "程式、計畫、資料隔離檢查、mask、逐層journal、所有訓練logs與來源SHA均保留。VALIDATION.json檢查精確配額、原baseline一致、資料隔離及逐層接受條件。checkpoint與CIFAR資料仍在本機，不含於ZIP。", "",
              "主方法及調整均屬待驗證研究，不主張整網路loss搜尋或BN校正本身新穎。不使用更多fine-tune epochs來補救新方法，也不把單seed的小幅差異當成統計顯著。", "",
              "計畫：../../research_notes/2026-11-02_task_loss_mask_plan.md；工作及問題紀錄：execution.jsonl、EXECUTION_LOG.md。"]
    (ROOT / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = dict(state="complete", updated_at=common.stamp(), jobs=jobs, selected_confirmation=selected,
                  test_evaluated=False, meaning="Bounded reproducible result, not a forced positive outcome")
    common.write_json(ROOT / "study_result.json", result)
    common.write_json(ROOT / "status.json", result)
    event("study_complete", selected=selected, new_pilots=len(jobs))
    files = [p for p in ROOT.rglob("*") if p.is_file() and p.name not in {"study.lock", "MANIFEST.json"} and (p.suffix != ".pt" or p.name == "masks.pt")]
    common.write_json(ROOT / "MANIFEST.json", {str(p.relative_to(ROOT)): common.sha(p) for p in files})
    archive = ROOT.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files + [ROOT / "MANIFEST.json"]:
            z.write(p, p.relative_to(ROOT.parent))
    archive.with_suffix(".zip.sha256").write_text(common.sha(archive) + "  " + archive.name + "\n", encoding="ascii")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    for sub in ["logs", "controls", "source_snapshot"]:
        (ROOT / sub).mkdir(exist_ok=True)
    lock = (ROOT / "study.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sources = [Path(__file__), PROJECT / "scripts/task_loss_mask_search.py", PROJECT / "scripts/run_shared_mask_repair.py",
               common.CORE, PROJECT / "scripts/run_fair_criterion_r0p9.py", PROJECT / "scripts/shared_mask_repair.py",
               PROJECT / "tests/test_task_loss_mask_search.py", PROJECT / "research_notes/2026-11-02_task_loss_mask_plan.md"]
    for source in sources:
        frozen = ROOT / "source_snapshot" / source.name
        if frozen.exists() and common.sha(source) != common.sha(frozen):
            raise ValueError(f"Source changed after experiment freeze: {source}")
        shutil.copy2(source, frozen)
    old_snapshot = PROJECT / "runs/shared_mask_repair_study/source_snapshot/run_shared_mask_repair.py"
    if common.sha(old_snapshot) != common.sha(PROJECT / "scripts/run_shared_mask_repair.py"):
        raise ValueError("Cannot reuse controls after pilot worker changes")
    jobs = [dict(seed=0, bn="frozen", policy=p) for p in ["dual", "search_only"]]
    ensure_controls(jobs)
    event("initial_plan", models=["resnet50"], jobs=jobs, trigger="dual gain >=0.1pp vs l1 and no loss vs unconditional worst", maximum_versions=2)
    run_batch(jobs, "search")
    run_batch(jobs, "pilot")
    selected = jobs[0] if useful(jobs[0]) else None
    if not selected:
        adjustment = [dict(seed=0, bn="recalibrate", policy=p) for p in ["dual", "search_only"]]
        event("bounded_adjustment", reason="Frozen BN dual fails validation promotion; evaluate per-candidate BN recalibration as a selection proxy only", jobs=adjustment)
        run_batch(adjustment, "search")
        run_batch(adjustment, "pilot")
        jobs.extend(adjustment)
        if useful(adjustment[0]):
            selected = adjustment[0]
    if selected:
        extra = [dict(selected, seed=s) for s in [1, 2]]
        event("confirmation_locked", selected=selected, seeds=[1, 2])
        run_batch(extra, "prepare")
        ensure_controls(extra)
        run_batch(extra, "search")
        run_batch(extra, "pilot")
        jobs.extend(extra)
    finish(jobs, selected)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if ROOT.exists():
            common.write_json(ROOT / "status.json", dict(state="error", time=common.stamp(), error=repr(error)))
            event("study_error", error=repr(error))
        raise
