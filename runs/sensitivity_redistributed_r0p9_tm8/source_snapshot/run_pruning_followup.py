#!/usr/bin/env python3
"""Wait for the criterion sweep, then run the predeclared matched-budget follow-up."""

import argparse
import csv
import fcntl
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import run_fair_criterion_r0p9 as fair


PROJECT = fair.PROJECT
ROOT = PROJECT / "runs/sensitivity_redistributed_r0p9_tm8"
CRITERIA = ["l1_sum", "top2_mean_abs"]


def ranking_dir(model, seed):
    if model == "resnet34":
        return PROJECT / f"runs/fixed_checkpoint_resnet34/sensitivity_validation/seed{seed}_tm8_top2"
    return PROJECT / f"runs/fixed_checkpoint_multi_model/sensitivity_validation/{model}_seed{seed}_tm8_top2"


def identifier(job):
    return f"{job[0]}_seed{job[1]}_tm8_r0p90_{job[2]}_redistributed"


def result_path(job):
    return ROOT / "per_run_csv" / (identifier(job) + "_results.csv")


def prepare(base):
    feasible, budgets, provenance = [], {}, []
    for (model, seed), baseline in base.items():
        directory = ranking_dir(model, seed)
        ranking = fair.read_csv(directory / "layer_sensitivity_ranked.csv")
        ranking.sort(key=lambda r: float(r["mean_drop_pp"]), reverse=True)
        caps = {r["layer_name"]: 0 if i == 0 else .2 for i, r in enumerate(ranking[:4])}
        layers = {r["layer_name"]: r for r in fair.read_csv(directory / "layer_sensitivity.csv")}
        total = sum(int(r["weight_count"]) for r in layers.values())
        maximum = 0
        for name, row in layers.items():
            out = int(row["out_channels"])
            k = int(row["weight_count"]) // out
            if out % 8:
                raise ValueError("Follow-up allocation currently requires full Tm8 clusters")
            maximum += out * (math.floor(k * caps[name] + 1e-12) if name in caps else k - 1)
        target = round(float(baseline["pruning_ratio"]) * total)
        if target % 8 or target > maximum:
            raise ValueError(f"Infeasible matched budget for {model}/seed{seed}")
        budgets[(model, seed)] = target
        feasible.append({"model": model, "seed": seed, "total_conv_weights": total,
                         "target_pruned_weights": target, "target_actual_prune_pct": target / total * 100,
                         "max_prune_pct_under_protection": maximum / total * 100, "feasible": True})
        provenance.append({"model": model, "seed": seed, "checkpoint_id": baseline["checkpoint_id"],
                           "ranking_sha256": fair.sha(directory / "layer_sensitivity_ranked.csv"),
                           "layer_metadata_sha256": fair.sha(directory / "layer_sensitivity.csv"),
                           "checkpoint_sha256": fair.sha(fair.checkpoint(model, seed))})
    fair.write_csv(ROOT / "allocation_feasibility.csv", feasible)
    manifest = {"stage": "matched_weight_budget_sensitivity_redistribution", "criteria": CRITERIA,
                "tm": 8, "seeds": [0, 1, 2], "target_ratio": 0.9, "protected_top_k": 4,
                "skip_top_k": 1, "protected_cap": .2, "all_layer_cap": None,
                "non_sensitive_min_kept_k": 1, "allocation_mobility": "1/(1+max(0,mean_drop_pp))",
                "finetune_epochs": 10, "finetune_lr": 0.001, "batch_size": 128,
                "core_sha256": fair.sha(fair.CORE), "worker_sha256": fair.sha(PROJECT / "scripts/sensitivity_budget_pruning.py"),
                "runner_sha256": fair.sha(Path(__file__)), "provenance": provenance}
    path = ROOT / "experiment_manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Follow-up inputs changed; refusing to mix experiments")
    fair.write_json(path, manifest)
    snapshot = ROOT / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for path in [fair.CORE, Path(__file__), PROJECT / "scripts/sensitivity_budget_pruning.py"]:
        (snapshot / path.name).write_bytes(path.read_bytes())
    return budgets


def command(job, target):
    argv = fair.command(job)
    argv[argv.index(str(fair.CORE))] = str(PROJECT / "scripts/sensitivity_budget_pruning.py")
    argv[argv.index("--save-dir") + 1] = str(ROOT / "per_run_csv")
    argv[argv.index("--run-name") + 1] = identifier(job)
    return argv + ["--ranking-csv", str(ranking_dir(*job[:2]) / "layer_sensitivity_ranked.csv"),
                   "--target-pruned-weights", str(target), "--allocation-dir", str(ROOT / "allocations" / identifier(job))]


def validate(job, base):
    rows = [r for r in fair.read_csv(result_path(job)) if r["criterion"] == job[2]]
    if len(rows) != 1:
        raise ValueError(f"Bad result count: {identifier(job)}")
    row = rows[0]
    baseline = base[job[:2]]
    expected = {"model": job[0], "seed": str(job[1]), "tm": "8", "checkpoint_id": baseline["checkpoint_id"],
                "ratio_source": "sensitivity_redistributed_exact_budget"}
    for key, value in expected.items():
        if row[key] != value:
            raise ValueError(f"Wrong {key}: {identifier(job)}")
    if not math.isclose(float(row["pruning_ratio"]), float(baseline["pruning_ratio"]), abs_tol=1e-6, rel_tol=0):
        raise ValueError(f"Unmatched sparsity: {identifier(job)}")
    audit = json.loads((ROOT / "allocations" / identifier(job) / "budget.json").read_text())
    if audit["allocated_pruned_weights"] != audit["target_pruned_weights"]:
        raise ValueError("Invalid integer weight budget")
    allocation = fair.read_csv(ROOT / "allocations" / identifier(job) / "cluster_allocation.csv")
    for cluster in allocation:
        count = int(cluster["allocated_prune_k"])
        if count > int(cluster["max_prune"]) or count < 0:
            raise ValueError("Protection cap violated")
    if sum(int(c["allocated_pruned_weights"]) for c in allocation) != audit["target_pruned_weights"]:
        raise ValueError("Cluster records disagree with weight budget")
    return row


def summarize(base):
    # Stage-one results supply the no-protection arm of a paired 2x2 comparison.
    rows = []
    for baseline in base.values():
        rows.append(dict(baseline, protection="none"))
    for model in fair.MODELS:
        for seed in range(3):
            job = (model, seed, "top2_mean_abs")
            if fair.result_path(job).exists():
                rows.append(dict(fair.validate_result(job, base[job[:2]]), protection="none",
                                 source_file=str(fair.result_path(job).relative_to(PROJECT))))
    completed = 0
    for model in fair.MODELS:
        for seed in range(3):
            for criterion in CRITERIA:
                job = (model, seed, criterion)
                if result_path(job).exists():
                    rows.append(dict(validate(job, base), protection="sensitivity_redistributed",
                                     source_file=str(result_path(job).relative_to(PROJECT))))
                    completed += 1
    groups = defaultdict(list)
    for row in rows:
        row["dense_extra_test_acc"] = base[(row["model"], int(row["seed"]))]["dense_extra_test_acc"]
        row["accuracy_delta_vs_dense_extra"] = float(row["after_extra_or_ft_acc"]) - float(row["dense_extra_test_acc"])
        groups[(row["model"], row["criterion"], row["protection"])].append(row)
    fair.write_csv(ROOT / "all_results_combined.csv", rows, list(dict.fromkeys(k for r in rows for k in r)))
    paired = []
    for row in rows:
        if row["protection"] != "sensitivity_redistributed":
            continue
        control = next(r for r in rows if r["model"] == row["model"] and r["seed"] == row["seed"]
                       and r["criterion"] == row["criterion"] and r["protection"] == "none")
        paired.append({"model": row["model"], "seed": row["seed"], "criterion": row["criterion"],
                       "gain_from_redistribution_pp": (float(row["after_extra_or_ft_acc"]) - float(control["after_extra_or_ft_acc"])) * 100,
                       "unprotected_actual_sparsity": control["pruning_ratio"], "protected_actual_sparsity": row["pruning_ratio"],
                       "unprotected_mac_reduction": control["mac_reduction"], "protected_mac_reduction": row["mac_reduction"]})
    if paired:
        fair.write_csv(ROOT / "paired_protection_effect.csv", paired)
    summary = []
    for (model, criterion, protection), values in groups.items():
        loss = [(float(r["dense_extra_test_acc"]) - float(r["after_extra_or_ft_acc"])) * 100 for r in values]
        summary.append({"model": model, "criterion": criterion, "protection": protection, "seeds": len(values),
                        "complete_3_seeds": len(values) == 3,
                        "ft_accuracy_pct_mean": statistics.mean(float(r["after_extra_or_ft_acc"]) * 100 for r in values),
                        "loss_pp_mean": statistics.mean(loss), "loss_std_pp": statistics.stdev(loss) if len(loss) > 1 else "",
                        "actual_prune_pct_mean": statistics.mean(float(r["pruning_ratio"]) * 100 for r in values),
                        "mac_reduction_pct_mean": statistics.mean(float(r["mac_reduction"]) * 100 for r in values)})
    fair.write_csv(ROOT / "method_summary.csv", summary)
    report = ["# Matched-Budget Results", "", "Incomplete groups are marked by seed count. Loss is relative to the paired Dense extra; speed is not measured.", "",
              "| Model | Criterion | Protection | Seeds | FT accuracy (%) | Loss (pp) | Actual sparsity (%) | MAC reduction (%) |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        report.append(f"| {r['model']} | {r['criterion']} | {r['protection']} | {r['seeds']} | {r['ft_accuracy_pct_mean']:.3f} | {r['loss_pp_mean']:.3f} | {r['actual_prune_pct_mean']:.4f} | {r['mac_reduction_pct_mean']:.3f} |")
    report += ["", "![Matched-budget accuracy](matched_budget_accuracy.png)"]
    (ROOT / "RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    arms = [(c, p) for c in CRITERIA for p in ["none", "sensitivity_redistributed"]]
    for ax, model in zip(axes.flat, fair.MODELS):
        for i, (criterion, protection) in enumerate(arms):
            row = next((r for r in summary if (r["model"], r["criterion"], r["protection"]) == (model, criterion, protection)), None)
            if row:
                ax.bar(i, row["loss_pp_mean"], yerr=row["loss_std_pp"] if row["loss_std_pp"] != "" else None,
                       color=["#666666", "#338c72", "#b97632", "#ad4b76"][i], capsize=3,
                       alpha=1 if row["complete_3_seeds"] else .4)
        ax.set_title(model)
        ax.set_xticks(range(4), ["L1", "L1+sens", "Top2", "Top2+sens"])
        ax.set_ylabel("Accuracy loss vs Dense extra (pp)")
    axes.flat[-1].axis("off")
    fig.suptitle("Matched actual ~90% weight sparsity; MAC budgets may differ")
    fig.tight_layout()
    fig.savefig(ROOT / "matched_budget_accuracy.png", dpi=160)
    plt.close(fig)
    return completed


def final_report():
    first = fair.read_csv(fair.ROOT / "criterion_summary.csv")
    second = fair.read_csv(ROOT / "method_summary.csv")
    if len(first) != 20 or len(second) != 20 or any(int(r["seeds"]) != 3 for r in first + second):
        raise ValueError("Final report requires all three seeds for all experiment arms")
    lines = ["# 實際約90%剪枝：公平比較最終結果", "", f"整理時間：{fair.stamp()}", "",
             "共完成45個importance比較與30個敏感層重新分配任務；另沿用15組l1_sum基準。Tm8，固定Dense checkpoint，fine-tune10 epochs。", "",
             "## Importance-only", "", "| Model | Criterion | FT acc (%) | Loss (pp) | Actual sparsity (%) | MAC reduction (%) |",
             "|---|---|---:|---:|---:|---:|"]
    for r in first:
        lines.append(f"| {r['model']} | {r['criterion']} | {float(r['ft_acc_pct_mean']):.3f} | {float(r['loss_pp_mean']):.3f} | {float(r['actual_prune_pct_mean']):.4f} | {float(r['mac_reduction_pct_mean']):.3f} |")
    lines += ["", "![Importance-only](fair_criterion_r0p9_tm8/accuracy_loss_comparison.png)", "",
              "## Sensitivity + budget redistribution", "",
              "下表匹配weight sparsity，未匹配MAC budget。此規則是預先固定的heuristic，不保證保護版會提高精度。", "",
              "| Model | Criterion | Protection | FT acc (%) | Loss (pp) | Actual sparsity (%) | MAC reduction (%) |",
              "|---|---|---|---:|---:|---:|---:|"]
    for r in second:
        lines.append(f"| {r['model']} | {r['criterion']} | {r['protection']} | {float(r['ft_accuracy_pct_mean']):.3f} | {float(r['loss_pp_mean']):.3f} | {float(r['actual_prune_pct_mean']):.4f} | {float(r['mac_reduction_pct_mean']):.3f} |")
    lines += ["", "![Matched-budget](sensitivity_redistributed_r0p9_tm8/matched_budget_accuracy.png)", "",
              "原始數據與sample std見兩個資料夾的CSV；gain正值表示改善，必須同時檢查三seed配對差異。圖表為軟體精度實驗，不能推論實測硬體能耗或延遲。"]
    report = PROJECT / "runs/FAIR_PRUNING_R0P9_FINAL.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    archive = PROJECT / "runs/fair_pruning_r0p9_complete_20261027.zip"
    files = {report}
    for root in [fair.ROOT, ROOT]:
        files.update(p for p in root.rglob("*") if p.is_file() and p.suffix in {".csv", ".md", ".png", ".json", ".py"}
                     and not {"logs", "rejected_csv", "smoke_structural_only"}.intersection(p.relative_to(root).parts))
    for baseline in fair.baselines().values():
        files.add(PROJECT / baseline["source_file"])
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        for path in sorted(files):
            output.write(path, path.relative_to(PROJECT))
    with zipfile.ZipFile(archive) as output:
        if output.testzip() is not None:
            raise ValueError("Final ZIP integrity check failed")
    archive.with_suffix(".zip.sha256").write_text(fair.sha(archive) + "  " + archive.name + "\n", encoding="ascii")
    return str(archive)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--max-jobs", type=int, default=3)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.max_jobs < 1:
        parser.error("--max-jobs must be positive")
    for directory in [ROOT, ROOT / "logs", ROOT / "per_run_csv"]:
        directory.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / "scheduler.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    base = fair.baselines()
    budgets = prepare(base)
    if args.prepare_only:
        print("Validated 15 feasible matched budgets; follow-up contains 30 tasks")
        return 0
    (ROOT / "scheduler.pid").write_text(str(os.getpid()) + "\n", encoding="ascii")
    active, failed = {}, {}
    stopped = False
    stage_one_retried = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while not stopped:
            status_path = fair.ROOT / "status.json"
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            if status.get("state") == "complete" and status.get("completed") == 45:
                subprocess.run([sys.executable, "scripts/run_fair_criterion_r0p9.py", "--summarize"], cwd=PROJECT, check=True)
                break
            if status.get("state") == "failed" and not stage_one_retried:
                stage_one_retried = True
                fair.write_json(ROOT / "status.json", {"state": "recovering_criterion_sweep", "updated_at": fair.stamp()})
                time.sleep(5)
                subprocess.run([sys.executable, "scripts/run_fair_criterion_r0p9.py", "--run", "--gpu", str(args.gpu), "--max-jobs", "1"],
                               cwd=PROJECT, check=True)
                continue
            if status.get("state") in {"failed", "error", "stopped"}:
                raise RuntimeError("Criterion sweep needs recovery before starting the follow-up")
            if status.get("state") == "running":
                try:
                    os.kill(int(status["scheduler_pid"]), 0)
                except ProcessLookupError:
                    latest = json.loads(status_path.read_text())
                    if latest.get("state") == "complete" and latest.get("completed") == 45:
                        continue
                    raise
            fair.write_json(ROOT / "status.json", {"state": "waiting_for_criterion_sweep", "updated_at": fair.stamp(),
                            "scheduler_pid": os.getpid(), "stage_one_completed": status.get("completed", 0), "total": 30})
            time.sleep(30)
        pending = []
        for model in fair.MODELS:
            for seed in range(3):
                for criterion in CRITERIA:
                    job = (model, seed, criterion)
                    if result_path(job).exists():
                        validate(job, base)
                    else:
                        pending.append(job)
        completed = summarize(base)
        last_sample = 0
        while (pending or active) and not stopped:
            for job, (process, log) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del active[job]
                try:
                    if code:
                        raise RuntimeError(f"exit={code}")
                    validate(job, base)
                except (OSError, ValueError, RuntimeError) as error:
                    failed[identifier(job)] = str(error)
                    if result_path(job).exists():
                        rejected = ROOT / "rejected_csv"
                        rejected.mkdir(exist_ok=True)
                        result_path(job).rename(rejected / (identifier(job) + f"_{time.time_ns()}.csv"))
                else:
                    completed = summarize(base)
                print(f"{fair.stamp()} FINISH {identifier(job)} exit={code} complete={completed}/30", flush=True)
            while pending and len(active) < args.max_jobs and not stopped:
                manifest = json.loads((ROOT / "experiment_manifest.json").read_text())
                if fair.sha(fair.CORE) != manifest["core_sha256"] or fair.sha(PROJECT / "scripts/sensitivity_budget_pruning.py") != manifest["worker_sha256"]:
                    raise RuntimeError("Frozen experiment source changed")
                job = pending.pop(0)
                log = (ROOT / "logs" / (identifier(job) + ".log")).open("a", buffering=1)
                argv = command(job, budgets[job[:2]])
                log.write(json.dumps({"start": fair.stamp(), "gpu": args.gpu, "argv": argv}) + "\n")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
                process = subprocess.Popen(argv, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[job] = (process, log)
                print(f"{fair.stamp()} START {identifier(job)} pid={process.pid}", flush=True)
            if time.monotonic() - last_sample > 300:
                sample = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
                with (ROOT / "logs/resources.log").open("a") as resource:
                    resource.write(fair.stamp() + "\n" + sample.stdout)
                last_sample = time.monotonic()
            fair.write_json(ROOT / "status.json", {"state": "running" if pending or active else ("failed" if failed else "complete"),
                            "updated_at": fair.stamp(), "scheduler_pid": os.getpid(), "completed": completed, "total": 30,
                            "active": [{"name": identifier(j), "pid": p.pid} for j, (p, _) in active.items()],
                            "pending": len(pending), "failed": failed})
            if pending or active:
                time.sleep(10)
        if not failed and not stopped and completed == 30:
            archive = final_report()
            fair.write_json(ROOT / "status.json", {"state": "complete", "updated_at": fair.stamp(),
                            "completed": 30, "total": 30, "archive": archive, "failed": {}})
        return int(bool(failed) or stopped)
    except Exception as error:
        fair.write_json(ROOT / "status.json", {"state": "error", "updated_at": fair.stamp(), "error": str(error), "failed": failed})
        raise
    finally:
        for process, log in active.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            log.close()
        if stopped:
            fair.write_json(ROOT / "status.json", {"state": "stopped", "updated_at": fair.stamp()})


if __name__ == "__main__":
    sys.exit(main())
