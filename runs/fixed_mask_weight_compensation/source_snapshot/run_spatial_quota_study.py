#!/usr/bin/env python3
"""Resume interpretation controls, conditionally confirm seeds, and report honestly."""

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
ROOT = PROJECT / "runs/spatial_quota_baseline_study"
CRITERIA = ["l1_sum", "top2_mean_abs"]


def event(kind, **details):
    value = dict(time=common.stamp(), kind=kind, **details)
    with (ROOT / "execution.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    print(json.dumps(value, ensure_ascii=False), flush=True)


def output(job):
    return ROOT / f"{job['model']}_seed{job['seed']}_{job['criterion']}_{job['granularity']}"


def expected(job):
    return output(job) / "pilots" / (job["criterion"] + ".json")


def command(job):
    args = [sys.executable, "-u", "scripts/spatial_quota_pruning.py", "--root", str(ROOT)]
    for key in ["model", "seed", "criterion", "granularity"]:
        args.extend(["--" + key, str(job[key])])
    return args


def run_batch(jobs):
    pending = [j for j in jobs if not expected(j).exists()]
    active = {}
    try:
        while pending or active:
            for gpu, (job, process, log) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del active[gpu]
                if code or not expected(job).exists():
                    raise RuntimeError(f"Worker failed: {job}, exit={code}")
                event("job_complete", job=job, gpu=gpu)
            for gpu in [1, 0]:
                if gpu in active or not pending:
                    continue
                free = int(subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
                if free < 12000:
                    continue
                job = pending.pop(0)
                label = output(job).name
                log = (ROOT / "logs" / (label + ".log")).open("a", buffering=1)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
                process = subprocess.Popen(command(job), cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[gpu] = (job, process, log)
                event("job_start", job=job, gpu=gpu, pid=process.pid)
            common.write_json(ROOT / "status.json", dict(state="running", updated_at=common.stamp(), pending=len(pending),
                              active=[dict(gpu=g, pid=p.pid, job=j) for g, (j, p, _) in active.items()]))
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


def score(model, seed, criterion, granularity):
    path = expected(dict(model=model, seed=seed, criterion=criterion, granularity=granularity))
    return json.loads(path.read_text())["final_validation_acc"]


def gap(model, seed, criterion):
    return (score(model, seed, criterion, "spatial") - score(model, seed, criterion, "flat")) * 100


def verify(jobs):
    import torch
    checks = []
    for job in jobs:
        out = output(job)
        audit = torch.load(out / "mask_audit.pt", map_location="cpu", weights_only=False)
        clusters = common.read_csv(out / "clusters.csv")
        for name, flat in audit["flat"].items():
            spatial = audit["spatial"][name]
            if not torch.equal(flat.sum(1), spatial.sum(1)):
                raise ValueError("Per-cluster quota changed")
        for row in clusters:
            if int(row["spatial_positions"]) == 1:
                name, i = row["layer"], int(row["cluster"])
                if not torch.equal(audit["flat"][name][i], audit["spatial"][name][i]):
                    raise ValueError("1x1 masks changed")
        config = json.loads((out / "configuration.json").read_text())
        for key, path in [("core_sha256", common.CORE), ("worker_sha256", PROJECT / "scripts/spatial_quota_pruning.py"),
                          ("checkpoint_sha256", common.checkpoint(job["model"], job["seed"]))]:
            if config[key] != common.sha(path):
                raise ValueError("Frozen source changed")
        checks.append(dict(job=job, quotas=True, pointwise_identical=True, source_hashes=True, test_evaluated=False))
    # Independently generated masks from the two jobs must be identical by interpretation.
    for job in jobs:
        if job["granularity"] != "flat":
            continue
        other = dict(job, granularity="spatial")
        a = torch.load(output(job) / "mask_audit.pt", map_location="cpu", weights_only=False)
        b = torch.load(output(other) / "mask_audit.pt", map_location="cpu", weights_only=False)
        for variant in a:
            if not all(torch.equal(a[variant][name], b[variant][name]) for name in a[variant]):
                raise ValueError("Separate worker mask generation disagrees")
    common.write_json(ROOT / "VALIDATION.json", dict(passed=True, checks=checks))


def report(jobs, expanded):
    rows = []
    for job in jobs:
        out = output(job)
        p = json.loads(expected(job).read_text())
        audit = json.loads((out / "mask_report.json").read_text())
        rows.append(dict(**job, direct_val_pct=p["direct_validation_acc"]*100,
                         final_val_pct=p["final_validation_acc"]*100, val_loss=p["validation_loss"],
                         sparsity_pct=p["sparsity"]*100, kept_weights=audit["kept_weights"],
                         changed_clusters=audit["changed_clusters"], clusters=audit["clusters"]))
    common.write_csv(ROOT / "validation_results.csv", rows)
    summary = []
    for model in ["resnet34", "resnet50"]:
        seeds = [0, 1, 2] if model in expanded else [0]
        for criterion in CRITERIA:
            diffs = [gap(model, seed, criterion) for seed in seeds]
            summary.append(dict(model=model, criterion=criterion, seeds=len(seeds),
                                flat_val_pct=statistics.mean(score(model, s, criterion, "flat")*100 for s in seeds),
                                spatial_val_pct=statistics.mean(score(model, s, criterion, "spatial")*100 for s in seeds),
                                spatial_minus_flat_pp=statistics.mean(diffs),
                                paired_sample_std_pp=statistics.stdev(diffs) if len(diffs)>1 else ""))
    common.write_csv(ROOT / "paired_summary.csv", summary)
    large = [r for r in summary if abs(r["spatial_minus_flat_pp"]) >= .3-1e-12]
    decision = dict(state="baseline_complete", completed_at=common.stamp(), jobs=len(jobs), expanded_models=expanded,
                    gradient_pilot_justified=not large,
                    reason="No >=0.3pp mean baseline gap in this bounded audit" if not large else "Material quota interpretation sensitivity remains; defer new mask method",
                    caution="The threshold controls effort, not statistical significance or equivalence; author implementation is unverified",
                    test_evaluated=False)
    common.write_json(ROOT / "decision.json", decision)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, model in zip(axes, ["resnet34", "resnet50"]):
        values = [r for r in summary if r["model"] == model]
        x = list(range(len(values)))
        ax.bar([v-.18 for v in x], [r["flat_val_pct"] for r in values], .36, label="Flat K", color="#378679")
        ax.bar([v+.18 for v in x], [r["spatial_val_pct"] for r in values], .36, label="Spatial quotas", color="#aa657b")
        ax.set_xticks(x, [r["criterion"] for r in values])
        ax.set_ylabel("Final validation accuracy (%)")
        ax.set_title(model + f" ({values[0]['seeds']} seed(s))")
        ax.set_ylim(min(min(r["flat_val_pct"], r["spatial_val_pct"]) for r in values)-.5, max(max(r["flat_val_pct"], r["spatial_val_pct"]) for r in values)+.5)
        ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "comparison.png", dpi=170)
    plt.close(fig)
    lines = ["# Cross-filter配額粒度基準審核", "", f"完成：{decision['completed_at']}", "",
             "## 比較定義", "", "flat為目前整個Cin×Kh×Kw的共同配額；spatial為每(r,s)以global reference分攤保留數的解讀版。兩者每filter-cluster的總kept_count精確相同；不是拿較低sparsity換accuracy。", "",
             "原文Figure4/5支持共享mask，但配額粒度及整數規則未唯一明確，spatial不得標為作者原始實作。圖示、原文位置與假設見source_snapshot內計畫。", "",
             "## 相同預算結果", "", "| Model | Criterion | Seeds | Flat val (%) | Spatial val (%) | Spatial-flat (pp) | Paired std (pp) |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        std = f"{r['paired_sample_std_pp']:.3f}" if r["paired_sample_std_pp"] != "" else "N/A"
        lines.append(f"| {r['model']} | {r['criterion']} | {r['seeds']} | {r['flat_val_pct']:.3f} | {r['spatial_val_pct']:.3f} | {r['spatial_minus_flat_pp']:+.3f} | {std} |")
    lines += ["", "![比較圖](comparison.png)", "", "## 下一階段決策", "", decision["reason"], "",
              "若進入分類梯度pilot，會另立凍結方法與檔案，不把結果混入本表。若未進入，先處理基準敏感性。0.3pp是投入控制門檻，不是統計顯著或等效檢定。", "",
              "## 可重現性", "", "全部分支固定Dense checkpoint、相同split、Tm8、無保護/cap、10epochs SGD fine-tune及RNG。只評估5000張validation，不使用test。flat masks與現有核心逐元素一致；1×1兩版完全一致。VALIDATION.json另檢查每cluster配額、不同工作重新生成mask一致性及來源SHA。", "",
              "每分支含clusters.csv、spatial_counts.csv、mask_report.json及完整log。獨立round的空間配額誤差也保留。總訓練與問題處理見execution.jsonl及EXECUTION_LOG.md。"]
    (ROOT / "BASELINE_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    common.write_json(ROOT / "status.json", decision)
    event("baseline_complete", decision=decision)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    for sub in ["logs", "source_snapshot"]:
        (ROOT / sub).mkdir(exist_ok=True)
    lock = (ROOT / "study.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sources = [Path(__file__), PROJECT / "scripts/spatial_quota_pruning.py", common.CORE,
               PROJECT / "scripts/run_shared_mask_repair.py", PROJECT / "scripts/run_fair_criterion_r0p9.py",
               PROJECT / "scripts/shared_mask_repair.py", PROJECT / "tests/test_spatial_quota_pruning.py",
               PROJECT / "research_notes/2026-11-02_spatial_quota_audit_plan.md"]
    for source in sources:
        target = ROOT / "source_snapshot" / source.name
        if target.exists() and common.sha(source) != common.sha(target):
            raise ValueError(f"Frozen source changed: {source}")
        shutil.copy2(source, target)
    jobs = [dict(model=m, seed=0, criterion=c, granularity=g) for m in ["resnet34", "resnet50"] for c in CRITERIA for g in ["flat", "spatial"]]
    event("pilot_plan", jobs=jobs, expansion_threshold_pp=.3, test_used=False)
    run_batch(jobs)
    expanded = [m for m in ["resnet34", "resnet50"] if any(abs(gap(m, 0, c)) >= .3-1e-12 for c in CRITERIA)]
    event("seed_confirmation_decision", models=expanded, rule="any absolute gap >=0.3pp; confirm both criteria and both granularities")
    more = [dict(model=m, seed=s, criterion=c, granularity=g) for m in expanded for s in [1, 2] for c in CRITERIA for g in ["flat", "spatial"]]
    run_batch(more)
    jobs.extend(more)
    verify(jobs)
    report(jobs, expanded)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if ROOT.exists():
            common.write_json(ROOT / "status.json", dict(state="error", time=common.stamp(), error=repr(error)))
            event("study_error", error=repr(error))
        raise
