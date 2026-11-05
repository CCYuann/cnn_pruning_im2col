#!/usr/bin/env python3
"""Resume a fixed-checkpoint, uncapped Tm8 criterion sweep on one GPU."""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "runs/fair_criterion_r0p9_tm8"
MODELS = ["resnet34", "resnet50", "googlenet", "densenet121", "mobilenet_v2"]
CRITERIA = ["max_abs", "top2_mean_abs", "top4_mean_abs"]
BASELINE = PROJECT / "runs/fixed_checkpoint_all_ratio_summary/all_results_combined.csv"
CORE = PROJECT / "resnet34_pruning_experiments.py"


def stamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint(model, seed):
    directory = "fixed_checkpoint_resnet34" if model == "resnet34" else "fixed_checkpoint_multi_model"
    return PROJECT / "runs" / directory / "checkpoints" / f"{model}_seed{seed}_dense.pt"


def jobs():
    return [(model, seed, criterion) for model in MODELS for seed in range(3) for criterion in CRITERIA]


def name(job):
    model, seed, criterion = job
    return f"{model}_seed{seed}_tm8_r0p90_{criterion}"


def result_path(job):
    return ROOT / "per_run_csv" / (name(job) + "_results.csv")


def command(job):
    model, seed, criterion = job
    return [
        sys.executable, "-u", str(CORE), "--dataset", "cifar10", "--models", model,
        "--device", "cuda", "--amp", "--amp-dtype", "float16", "--epochs", "100",
        "--finetune-epochs", "10", "--finetune-lr", "0.001", "--batch-size", "128",
        "--num-workers", "2", "--pin-memory", "--prune-ratio", "0.9", "--tm", "8",
        "--cross-criteria", criterion, "--cross-ratio-source", "global_cluster",
        "--filter-grouping", "contiguous", "--protect-early-layers", "0",
        "--sensitivity-protect-top-k", "0", "--cross-only",
        "--dense-checkpoint-in", str(checkpoint(model, seed)), "--seed", str(seed),
        "--val-size", "5000", "--split-seed", "0", "--deterministic",
        "--no-cudnn-benchmark", "--save-dir", str(ROOT / "per_run_csv"), "--run-name", name(job),
    ]


def baselines():
    result = {}
    for row in read_csv(BASELINE):
        if int(row["tm"]) == 8 and float(row["target_prune_ratio"]) == 0.9 and row["criterion"] == "l1_sum":
            key = (row["model"], int(row["seed"]))
            if key in result or row["ratio_source"] != "global_cluster":
                raise ValueError(f"Unexpected baseline: {key}")
            result[key] = row
    if set(result) != {(m, s) for m in MODELS for s in range(3)}:
        raise ValueError("Expected all 15 uncapped l1_sum baselines")
    for row in result.values():
        source = PROJECT / row["source_file"]
        matches = [r for r in read_csv(source) if r["criterion"] == "l1_sum"]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous baseline: {source}")
        for key in ["checkpoint_id", "ratio_source", "after_extra_or_ft_acc", "pruning_ratio"]:
            if matches[0][key] != row[key]:
                raise ValueError(f"Baseline aggregate mismatch: {source} / {key}")
        if abs(float(row["pruning_ratio"]) - 0.9) > 0.0005:
            raise ValueError(f"Baseline is not approximately 90%: {source}")
    return result


def validate_result(job, baseline):
    rows = read_csv(result_path(job))
    matches = [r for r in rows if r["criterion"] == job[2]]
    if len(matches) != 1:
        raise ValueError(f"Missing or duplicate criterion row: {name(job)}")
    row = matches[0]
    expected = {"model": job[0], "seed": str(job[1]), "tm": "8", "dataset": "cifar10",
                "ratio_source": "global_cluster", "checkpoint_id": baseline["checkpoint_id"]}
    for key, value in expected.items():
        if row[key] != value:
            raise ValueError(f"Incorrect {key} for {name(job)}: {row[key]}")
    for key, value in [("target_prune_ratio", 0.9), ("dense_test_acc", float(baseline["dense_test_acc"])),
                       ("pruning_ratio", float(baseline["pruning_ratio"])), ("mac_reduction", float(baseline["mac_reduction"]))]:
        if not math.isclose(float(row[key]), value, rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"Unfair {key} for {name(job)}: {row[key]} vs {value}")
    if not 0 <= float(row["after_extra_or_ft_acc"]) <= 1:
        raise ValueError(f"Invalid accuracy for {name(job)}")
    return row


def prepare(base):
    import torch

    audits = []
    for (model, seed), row in base.items():
        path = checkpoint(model, seed)
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if payload["checkpoint_id"] != row["checkpoint_id"] or payload["seed"] != seed or payload["model"] != model:
            raise ValueError(f"Wrong checkpoint: {path}")
        if payload["split_metadata"]["split_id"] != "42fbb7440cc27c71" or payload["epochs"] != 100:
            raise ValueError(f"Unexpected training metadata: {path}")
        audits.append({"model": model, "seed": seed, "checkpoint_id": row["checkpoint_id"],
                       "path": str(path.relative_to(PROJECT)), "sha256": sha(path),
                       "l1_source_file": row["source_file"]})
    manifest = {"created_at": stamp(), "torch": torch.__version__, "cuda": torch.version.cuda,
                "core_sha256": sha(CORE), "runner_sha256": sha(Path(__file__)),
                "baseline_aggregate_sha256": sha(BASELINE),
                "configuration": {"tm": 8, "target_ratio": 0.9, "filter_grouping": "contiguous",
                                  "sensitivity": False, "caps": False, "finetune_epochs": 10,
                                  "finetune_lr": 0.001, "batch_size": 128, "amp_dtype": "float16"},
                "checkpoints": audits, "jobs": [{"name": name(j), "argv": command(j)} for j in jobs()]}
    path = ROOT / "experiment_manifest.json"
    if path.exists():
        previous = json.loads(path.read_text())
        for key in ["core_sha256", "baseline_aggregate_sha256", "configuration", "checkpoints", "jobs"]:
            if previous[key] != manifest[key]:
                raise ValueError(f"Cannot resume changed experiment inputs: {key}")
        if previous.get("runner_sha256", manifest["runner_sha256"]) != manifest["runner_sha256"]:
            raise ValueError("Cannot resume with a changed runner; inspect its source snapshot")
    if not path.exists() or "runner_sha256" not in json.loads(path.read_text()):
        write_json(path, manifest)
        (ROOT / "source_snapshot").mkdir(exist_ok=True)
        for source in [CORE, Path(__file__).resolve()]:
            (ROOT / "source_snapshot" / source.name).write_bytes(source.read_bytes())
    write_csv(ROOT / "reused_l1_baselines.csv", list(base.values()))


def summarize(base):
    rows, paired, groups = [], [], defaultdict(list)
    for key, baseline in base.items():
        rows.append(dict(baseline, reused_baseline="1"))
    for job in jobs():
        if not result_path(job).exists():
            continue
        baseline = base[job[:2]]
        row = validate_result(job, baseline)
        row = dict(row, source_file=str(result_path(job).relative_to(PROJECT)), reused_baseline="0")
        row["dense_extra_test_acc"] = baseline["dense_extra_test_acc"]
        row["accuracy_delta_vs_dense_extra"] = float(row["after_extra_or_ft_acc"]) - float(row["dense_extra_test_acc"])
        rows.append(row)
        paired.append({"model": job[0], "seed": job[1], "criterion": job[2],
                       "checkpoint_id": row["checkpoint_id"],
                       "l1_ft_acc": float(baseline["after_extra_or_ft_acc"]),
                       "variant_ft_acc": float(row["after_extra_or_ft_acc"]),
                       "gain_vs_l1_pp": (float(row["after_extra_or_ft_acc"]) - float(baseline["after_extra_or_ft_acc"])) * 100,
                       "actual_pruning_ratio": float(row["pruning_ratio"]),
                       "l1_actual_pruning_ratio": float(baseline["pruning_ratio"])})
    for row in rows:
        groups[(row["model"], row["criterion"])].append(row)
    columns = list(dict.fromkeys(k for r in rows for k in r))
    write_csv(ROOT / "all_results_combined.csv", rows, columns)
    if paired:
        write_csv(ROOT / "paired_vs_l1.csv", paired)
    summary = []
    for (model, criterion), samples in groups.items():
        acc = [float(r["after_extra_or_ft_acc"]) * 100 for r in samples]
        loss = [float(r["dense_extra_test_acc"]) * 100 - a for r, a in zip(samples, acc)]
        summary.append({"model": model, "criterion": criterion, "seeds": len(samples),
                        "complete_3_seeds": len(samples) == 3,
                        "ft_acc_pct_mean": statistics.mean(acc),
                        "ft_acc_std_pp": statistics.stdev(acc) if len(acc) > 1 else "",
                        "loss_pp_mean": statistics.mean(loss),
                        "loss_std_pp": statistics.stdev(loss) if len(loss) > 1 else "",
                        "actual_prune_pct_mean": statistics.mean(float(r["pruning_ratio"]) * 100 for r in samples),
                        "mac_reduction_pct_mean": statistics.mean(float(r["mac_reduction"]) * 100 for r in samples)})
    write_csv(ROOT / "criterion_summary.csv", summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, model in zip(axes.flat, MODELS):
        for offset, criterion in enumerate(["l1_sum"] + CRITERIA):
            match = next((r for r in summary if r["model"] == model and r["criterion"] == criterion), None)
            if match:
                ax.bar(offset, match["loss_pp_mean"], yerr=None if match["loss_std_pp"] == "" else match["loss_std_pp"],
                       capsize=3, color=["#666666", "#258e72", "#bb7339", "#a34e78"][offset],
                       alpha=1 if match["complete_3_seeds"] else 0.4)
                ax.text(offset, match["loss_pp_mean"] + 0.12, f"n={match['seeds']}", ha="center", fontsize=8)
        ax.set_title(model)
        ax.set_xticks(range(4), ["L1", "Max", "Top2", "Top4"])
        ax.set_ylabel("Accuracy loss vs Dense extra (pp)")
        ax.axhline(0, color="black", linewidth=0.6)
    axes.flat[-1].axis("off")
    fig.suptitle("Tm8, actual ~90% pruning, no protection/caps; incomplete groups faded")
    fig.tight_layout()
    fig.savefig(ROOT / "accuracy_loss_comparison.png", dpi=160)
    plt.close(fig)
    return len(rows) - 15


def run(args, base):
    lock = (ROOT / "scheduler.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prepare(base)
    pending, active, failed = [], {}, {}
    for job in jobs():
        if result_path(job).exists():
            validate_result(job, base[job[:2]])
        else:
            pending.append(job)
    completed = summarize(base)
    (ROOT / "scheduler.pid").write_text(str(os.getpid()) + "\n", encoding="ascii")
    last_sample = 0
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while pending or active:
            if stopped:
                break
            for job, (process, log) in list(active.items()):
                status = process.poll()
                if status is None:
                    continue
                log.close()
                del active[job]
                try:
                    if status != 0:
                        raise RuntimeError(f"process exit={status}")
                    validate_result(job, base[job[:2]])
                except (OSError, ValueError, RuntimeError) as error:
                    failed[name(job)] = str(error)
                    if result_path(job).exists():
                        rejected = ROOT / "rejected_csv"
                        rejected.mkdir(exist_ok=True)
                        result_path(job).rename(rejected / (name(job) + f"_{time.time_ns()}.csv"))
                else:
                    completed = summarize(base)
                print(f"{stamp()} FINISH {name(job)} exit={status} completed={completed}/45", flush=True)
            while pending and len(active) < args.max_jobs:
                if sha(CORE) != json.loads((ROOT / "experiment_manifest.json").read_text())["core_sha256"]:
                    raise RuntimeError("Core code changed while experiments are running")
                job = pending.pop(0)
                log = (ROOT / "logs" / (name(job) + ".log")).open("a", buffering=1)
                log.write(f"\nSTART {stamp()} GPU={args.gpu}\n")
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="4",
                                   MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", PYTHONUNBUFFERED="1")
                process = subprocess.Popen(command(job), cwd=PROJECT, env=environment,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[job] = (process, log)
                print(f"{stamp()} START {name(job)} pid={process.pid} GPU={args.gpu}", flush=True)
            if time.monotonic() - last_sample > 300:
                sample = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                                        capture_output=True, text=True, timeout=15)
                with (ROOT / "logs/resources.log").open("a") as resource:
                    resource.write(stamp() + "\n" + sample.stdout)
                last_sample = time.monotonic()
            write_json(ROOT / "status.json", {"updated_at": stamp(), "scheduler_pid": os.getpid(),
                       "gpu": args.gpu, "max_jobs": args.max_jobs, "completed": completed, "total": 45,
                       "pending": len(pending), "active": [{"name": name(j), "pid": p.pid} for j, (p, _) in active.items()],
                       "failed": failed, "state": "running" if pending or active else ("failed" if failed else "complete")})
            if pending or active:
                time.sleep(10)
    except Exception as error:
        write_json(ROOT / "status.json", {"updated_at": stamp(), "state": "error", "completed": completed,
                   "failed": failed, "error": str(error)})
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
            write_json(ROOT / "status.json", {"updated_at": stamp(), "state": "stopped", "completed": completed, "failed": failed})
    return int(bool(failed) or stopped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--summarize", action="store_true")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--max-jobs", type=int, default=3)
    args = parser.parse_args()
    if args.max_jobs < 1:
        parser.error("--max-jobs must be >= 1")
    for directory in [ROOT, ROOT / "logs", ROOT / "per_run_csv"]:
        directory.mkdir(parents=True, exist_ok=True)
    base = baselines()
    if args.prepare:
        prepare(base)
        print(f"Validated {len(base)} baselines/checkpoints; prepared {len(jobs())} jobs")
        summarize(base)
        return 0
    if args.summarize:
        print(f"New completed results: {summarize(base)}/45")
        return 0
    return run(args, base)


if __name__ == "__main__":
    sys.exit(main())
