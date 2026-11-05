#!/usr/bin/env python3
"""Bounded adaptive-gradient pilot and symmetric, conditional seed confirmation."""

import fcntl
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import run_fair_criterion_r0p9 as common

PROJECT = common.PROJECT
ROOT = PROJECT / "runs/adapted_gradient_mask_study"
BASE = PROJECT / "runs/spatial_quota_baseline_study"
VARIANTS = ["continuous_control", "gradient_all", "gradient_gate", "matched_random"]


def event(kind, **details):
    value = dict(time=common.stamp(), kind=kind, **details)
    with (ROOT / "execution.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False)+"\n")
    print(json.dumps(value, ensure_ascii=False), flush=True)


def directory(job):
    if job["mode"] == "baseline":
        return BASE / f"resnet50_seed{job['seed']}_{job['criterion']}_{job['granularity']}"
    return ROOT / f"seed{job['seed']}_{job['variant']}"


def expected(job):
    return directory(job) / ("pilots/"+job["criterion"]+".json" if job["mode"] == "baseline" else "result.json")


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
                    raise RuntimeError(f"Worker failed {job}, exit={code}")
                event("job_complete", job=job, gpu=gpu)
            for gpu in [1, 0]:
                if not pending or gpu in active:
                    continue
                free = int(subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
                if free < 12000:
                    continue
                job = pending.pop(0)
                if job["mode"] == "baseline":
                    args = [sys.executable, "-u", "scripts/spatial_quota_pruning.py", "--model", "resnet50", "--seed", str(job["seed"]),
                            "--criterion", job["criterion"], "--granularity", job["granularity"]]
                else:
                    args = [sys.executable, "-u", "scripts/adapted_gradient_mask.py", "--seed", str(job["seed"]), "--variant", job["variant"]]
                log = (ROOT/"logs"/(job["mode"]+"_"+directory(job).name+".log")).open("a", buffering=1)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
                process = subprocess.Popen(args, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[gpu] = (job, process, log)
                event("job_start", gpu=gpu, pid=process.pid, job=job)
            common.write_json(ROOT/"status.json", dict(state="running", updated_at=common.stamp(), pending=len(pending),
                              active=[dict(gpu=g, pid=p.pid, job=j) for g, (j, p, _) in active.items()]))
            if active or pending:
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


def baseline_rows(seed):
    return [dict(seed=seed, criterion=c, granularity=g, **json.loads((BASE/f"resnet50_seed{seed}_{c}_{g}"/"pilots"/(c+".json")).read_text()))
            for c in ["l1_sum", "top2_mean_abs"] for g in ["flat", "spatial"]]


def metric(seed, variant):
    return json.loads((ROOT/f"seed{seed}_{variant}"/"result.json").read_text())


def promising():
    best = max(r["final_validation_acc"] for r in baseline_rows(0))
    gate = metric(0, "gradient_gate")["final_validation_acc"]
    return gate-best >= .001-1e-12 and all(gate >= metric(0, v)["final_validation_acc"] for v in ["gradient_all", "matched_random"])


def verify(seeds):
    import torch
    checks = []
    for seed in seeds:
        control = ROOT/f"seed{seed}_continuous_control"
        actual = torch.load(control/"final_model.pt", map_location="cpu", weights_only=False)["model_state_dict"]
        previous = torch.load(BASE/f"resnet50_seed{seed}_l1_sum_flat/pilots/l1_sum.pt", map_location="cpu", weights_only=False)["model_state_dict"]
        if not all(torch.equal(actual[n], previous[n]) for n in actual):
            raise ValueError("2+8 control does not reproduce original 10-epoch model bit for bit")
        post = torch.load(control/"post_adaptation.pt", map_location="cpu", weights_only=False)
        plans = None
        for variant in VARIANTS:
            out = ROOT/f"seed{seed}_{variant}"
            masks = torch.load(out/"masks.pt", map_location="cpu", weights_only=False)
            if any(not torch.equal(masks["initial"][n].sum(1), m.sum(1)) for n, m in masks["selected"].items()):
                raise ValueError("Changed per-cluster quota")
            adapted = torch.load(out/"post_adaptation.pt", map_location="cpu", weights_only=False)
            if any(not torch.equal(post["model"][n], adapted["model"][n]) for n in post["model"]):
                raise ValueError("Adaptation states differ across branches")
            if variant != "continuous_control":
                candidate = torch.load(out/"mask_plans.pt", map_location="cpu", weights_only=False)
                if plans is not None and any(not torch.equal(plans[v][n], candidate[v][n]) for v in plans for n in plans[v]):
                    raise ValueError("Candidate generation differs across branches")
                plans = candidate
                indices = json.loads((out/"data_indices.json").read_text())
                subsets = [set(indices[n]) for n in ["search", "check", "audit"]]
                val = set(torch.randperm(50000, generator=torch.Generator().manual_seed(0))[:5000].tolist())
                if len(set.union(*subsets)) != 2048 or any(s & val for s in subsets):
                    raise ValueError("Calibration overlaps")
                if any(not torch.equal((masks["initial"][n]&~candidate["gradient_gate"][n]).sum(1),
                                       (masks["initial"][n]&~candidate["matched_random"][n]).sum(1)) for n in masks["initial"]):
                    raise ValueError("Random control exchange budget differs")
            checks.append(dict(seed=seed, variant=variant, quota=True, identical_adaptation=True,
                               continuous_control_bit_exact=True, masks_sha256=common.sha(out/"masks.pt")))
    common.write_json(ROOT/"VALIDATION.json", dict(passed=True, checks=checks, test_evaluated=False))


def report(seeds):
    rows, controls = [], []
    for seed in seeds:
        baseline = baseline_rows(seed)
        controls.extend(baseline)
        best = max(r["final_validation_acc"] for r in baseline)
        for variant in VARIANTS:
            row = metric(seed, variant)
            row["gain_vs_best_baseline_pp"] = (row["final_validation_acc"]-best)*100
            if variant != "continuous_control":
                search = json.loads((ROOT/f"seed{seed}_{variant}"/"search.json").read_text())
                row.update(proposed_swaps=search["proposed_swaps"], gated_swaps=search["gated_swaps"], accepted_layers=search["accepted_layers"])
            else:
                row.update(proposed_swaps=0, gated_swaps=0, accepted_layers=0)
            rows.append(row)
    common.write_csv(ROOT/"validation_results.csv", rows)
    common.write_csv(ROOT/"baseline_controls.csv", controls)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4.5))
    mean = [statistics.mean(r["final_validation_acc"]*100 for r in rows if r["variant"]==v) for v in VARIANTS]
    ax.bar(range(4), mean, color=["#7d878f", "#b1657b", "#398c7b", "#aaa04c"])
    ax.axhline(statistics.mean(max(r["final_validation_acc"] for r in baseline_rows(s))*100 for s in seeds), color="#303030", linestyle="--", label="Best of four baseline controls per seed")
    ax.set_xticks(range(4), VARIANTS)
    ax.set_ylabel("Final validation accuracy (%)")
    ax.set_ylim(min(mean)-.5, max(mean)+.5)
    ax.set_title(f"ResNet50, 2+8 epochs, {len(seeds)} seed(s)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT/"comparison.png", dpi=170)
    plt.close(fig)
    lines = ["# 適應訓練後的共享Gate梯度：結果", "", f"完成：{common.stamp()}", "",
             "固定Dense checkpoint、Tm8、flat共享K、實際約90%權重剪枝，總訓練2+8epochs。保留SGD momentum、AMP scaler及訓練RNG。梯度只用train內新search資料，gate另用check否決；audit不參與決策，test沒有評估。", "",
             "## 分類結果", "", "| Seed | Method | Validation (%) | vs best of four baseline controls (pp) | Gate exchanges |", "|---:|---|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['seed']} | {r['variant']} | {r['final_validation_acc']*100:.3f} | {r['gain_vs_best_baseline_pp']:+.3f} | {r['gated_swaps']} |")
    lines += ["", "![分類比較](comparison.png)", "", "## 對照與可信度", "",
              "continuous_control的完整最終state_dict與原10epoch flat l1逐bit相同；各分支2epoch適應後state_dict亦完全一致。三個搜尋分支的候選／gate／隨機plans逐元素相同；只有最後套用的mask不同。VALIDATION.json檢查配額與資料隔離。", "",
              "比較控制包含本輪flat/spatial的l1/top2，不只與較弱l1比較；baseline_controls.csv列出全部數字。", "", "## 判斷", ""]
    if len(seeds) == 1:
        lines += ["主方法gradient_gate未同時達成：比四個baseline最高者高0.1pp、且不低於gradient_all與matched_random。因此未擴三seed；沒有改挑消融當成功方法，也沒有試更多交換數或epoch位置。"]
    else:
        gains = [r["gain_vs_best_baseline_pp"] for r in rows if r["variant"] == "gradient_gate"]
        lines += [f"seed0通過預設投入門檻後，對稱補齊三seed的四個baseline與四個分支。主方法相對各seed最佳baseline的平均差{statistics.mean(gains):+.3f}pp，sample std {statistics.stdev(gains):.3f}pp。這仍是validation研究，不當作獨立test或跨模型結論。"]
    lines += ["", "共享gate梯度可驗證為正確導數，但單次一階近似、stale-gradient與Dense dormant-weight恢復值都可能限制實際分類效果。這些是限制／假說，不是已證實失敗原因。Taylor與梯度再生已有文獻，未宣稱概念原創。", "",
              "完整方法凍結於source_snapshot計畫。逐層預測與真實CE、交換位置、訓練歷程與問題處理均保存。"]
    (ROOT/"FINAL_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    value = dict(state="complete", completed_at=common.stamp(), seeds=seeds, jobs=4*len(seeds), test_evaluated=False)
    common.write_json(ROOT/"study_result.json", value)
    common.write_json(ROOT/"status.json", value)
    event("study_complete", seeds=seeds)


def main():
    if not json.loads((BASE/"decision.json").read_text())["gradient_pilot_justified"]:
        raise RuntimeError("Baseline audit does not justify this stage")
    ROOT.mkdir(parents=True, exist_ok=True)
    for sub in ["logs", "source_snapshot"]:
        (ROOT/sub).mkdir(exist_ok=True)
    lock = (ROOT/"study.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    sources = [Path(__file__), PROJECT/"scripts/adapted_gradient_mask.py", PROJECT/"scripts/shared_gate_gradient.py",
               PROJECT/"scripts/task_loss_mask_search.py", PROJECT/"scripts/run_shared_mask_repair.py",
               PROJECT/"scripts/run_fair_criterion_r0p9.py", PROJECT/"scripts/shared_mask_repair.py", common.CORE,
               PROJECT/"research_notes/2026-11-02_adapted_gradient_plan.md"]
    for source in sources:
        frozen = ROOT/"source_snapshot"/source.name
        if frozen.exists() and common.sha(source) != common.sha(frozen):
            raise ValueError(f"Frozen source changed {source}")
        shutil.copy2(source, frozen)
    jobs = [dict(mode="gradient", seed=0, variant=v) for v in VARIANTS]
    event("pilot_plan", jobs=jobs, total_training_epochs=10, promotion="gate >= best four baseline controls +0.1pp and >= all/random")
    run_batch(jobs)
    seeds = [0]
    promote = promising()
    event("confirmation_decision", promote=promote)
    if promote:
        seeds.extend([1, 2])
        baseline = [dict(mode="baseline", seed=s, criterion=c, granularity=g) for s in [1, 2]
                    for c in ["l1_sum", "top2_mean_abs"] for g in ["flat", "spatial"]]
        run_batch(baseline)
        run_batch([dict(mode="gradient", seed=s, variant=v) for s in [1, 2] for v in VARIANTS])
    verify(seeds)
    report(seeds)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if ROOT.exists():
            common.write_json(ROOT/"status.json", dict(state="error", time=common.stamp(), error=repr(error)))
            event("study_error", error=repr(error))
        raise
