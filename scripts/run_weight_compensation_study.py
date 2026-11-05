#!/usr/bin/env python3
"""Run the fixed-mask compensation pilot and only predeclared seed confirmation."""

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
ROOT = PROJECT/"runs/fixed_mask_weight_compensation"
VARIANTS = ["original", "dense_input", "sequential"]


def event(kind, **details):
    value = dict(time=common.stamp(), kind=kind, **details)
    with (ROOT/"execution.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False)+"\n")
    print(json.dumps(value, ensure_ascii=False), flush=True)


def output(job):
    return ROOT/"calibration"/f"seed{job['seed']}" if job["mode"]=="prepare" else ROOT/f"seed{job['seed']}_{job['variant']}"


def expected(job):
    return output(job)/("ready.json" if job["mode"]=="prepare" else "result.json")


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
                args = [sys.executable, "-u", "scripts/weight_compensation_worker.py", job["mode"], "--seed", str(job["seed"]), "--root", str(ROOT)]
                if job["mode"]=="train":
                    args.extend(["--variant", job["variant"]])
                label = job["mode"]+"_"+output(job).name
                log = (ROOT/"logs"/(label+".log")).open("a", buffering=1)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
                process = subprocess.Popen(args, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[gpu] = (job, process, log)
                event("job_start", job=job, gpu=gpu, pid=process.pid)
            common.write_json(ROOT/"status.json", dict(state="running", updated_at=common.stamp(), pending=len(pending),
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


def result(seed, variant):
    return json.loads((ROOT/f"seed{seed}_{variant}/result.json").read_text())


def strong_control():
    rows = []
    base = PROJECT/"runs/spatial_quota_baseline_study"
    for criterion in ["l1_sum", "top2_mean_abs"]:
        for granularity in ["flat", "spatial"]:
            source = base/f"resnet50_seed0_{criterion}_{granularity}/pilots"/(criterion+".json")
            value = json.loads(source.read_text())
            rows.append(dict(criterion=criterion, granularity=granularity, accuracy=value["final_validation_acc"],
                             source=str(source.relative_to(PROJECT)), sha256=common.sha(source)))
    return rows


def promising():
    a, c = result(0, "original"), result(0, "sequential")
    return c["ft10_accuracy"]-a["ft10_accuracy"] >= .003-1e-12 and c["ft10_accuracy"] >= max(r["accuracy"] for r in strong_control())


def verify(seeds):
    import torch
    checks = []
    for seed in seeds:
        cache = ROOT/"calibration"/f"seed{seed}"
        masks = torch.load(cache/"masks.pt", map_location="cpu", weights_only=False)
        dense = torch.load(common.checkpoint("resnet50", seed), map_location="cpu", weights_only=False)["model_state_dict"]
        indices = json.loads((cache/"data_indices.json").read_text())
        validation = set(torch.randperm(50000, generator=torch.Generator().manual_seed(0))[:5000].tolist())
        if len(set(indices["fit"]+indices["check"])) != 1024 or set(indices["fit"]+indices["check"])&validation:
            raise ValueError("Calibration data isolation failed")
        conv_keys = {n+".weight" for n in masks}
        for variant in VARIANTS:
            directory = ROOT/f"seed{seed}_{variant}"
            other_masks = torch.load(directory/"masks.pt", map_location="cpu", weights_only=False)
            if masks.keys()!=other_masks.keys() or any(not torch.equal(masks[n], other_masks[n]) for n in masks):
                raise ValueError("Masks changed across variants")
            initial = torch.load(directory/"initialization.pt", map_location="cpu", weights_only=False)["model"]
            final = torch.load(directory/"final_model.pt", map_location="cpu", weights_only=False)["model_state_dict"]
            for name in dense:
                if name not in conv_keys and not torch.equal(initial[name], dense[name]):
                    raise ValueError("Offline compensation changed non-convolution parameters or BN")
            for name, mask in masks.items():
                for state in [initial, final]:
                    weight = state[name+".weight"].flatten(1)
                    full = mask.repeat_interleave(8, 0)[:len(weight)]
                    if torch.count_nonzero(weight[~full]):
                        raise ValueError("Forbidden weights changed")
            if variant!="original":
                for row in common.read_csv(directory/"cluster_reconstruction.csv"):
                    if float(row["check_mse_after"])>float(row["check_mse_before"])+1e-12:
                        raise ValueError("Check regression was not reverted")
                if variant=="dense_input" and any(float(r["input_fit_delta_rms"]) != 0 for r in common.read_csv(directory/"cluster_reconstruction.csv")):
                    raise ValueError("Dense input control is not using Dense patches")
            config = json.loads((directory/"configuration.json").read_text())
            for key, path in [("worker_sha256", PROJECT/"scripts/weight_compensation_worker.py"),
                              ("helper_sha256", PROJECT/"scripts/fixed_support_ridge.py"), ("core_sha256", common.CORE),
                              ("checkpoint_sha256", common.checkpoint("resnet50", seed))]:
                if config[key]!=common.sha(path):
                    raise ValueError("Frozen source changed")
            checks.append(dict(seed=seed, variant=variant, masks_identical=True, forbidden_positions_zero=True,
                               nonconv_initial_state_identical=True, data_isolation=True, source_hashes=True))
        old = PROJECT/"runs/spatial_quota_baseline_study"/f"resnet50_seed{seed}_l1_sum_flat/pilots/l1_sum.pt"
        if old.exists():
            previous = torch.load(old, map_location="cpu", weights_only=False)["model_state_dict"]
            current = torch.load(ROOT/f"seed{seed}_original/final_model.pt", map_location="cpu", weights_only=False)["model_state_dict"]
            if any(not torch.equal(previous[n], current[n]) for n in current):
                raise ValueError("Original control does not reproduce prior 10-epoch model bit for bit")
    common.write_json(ROOT/"VALIDATION.json", dict(passed=True, checks=checks, test_evaluated=False))


def render(seeds):
    rows = []
    histories = []
    for seed in seeds:
        for variant in VARIANTS:
            row = result(seed, variant)
            row["gain_ft10_vs_original_pp"] = (row["ft10_accuracy"]-result(seed, "original")["ft10_accuracy"])*100
            rows.append(row)
            for record in common.read_csv(ROOT/f"seed{seed}_{variant}/training_history.csv"):
                histories.append(dict(seed=seed, variant=variant, **record))
    common.write_csv(ROOT/"validation_results.csv", rows)
    common.write_csv(ROOT/"all_training_history.csv", histories)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = dict(original="#74818b", dense_input="#ad667d", sequential="#2c8a78")
    for variant in VARIANTS:
        means = [statistics.mean(float(r["validation_accuracy"])*100 for r in histories if r["variant"]==variant and int(r["epoch"])==epoch) for epoch in range(1, 11)]
        axes[0].plot(range(1, 11), means, marker="o", label=variant, color=colors[variant])
    axes[0].set_xlabel("Fine-tuning epoch")
    axes[0].set_ylabel("Validation accuracy (%)")
    axes[0].legend()
    for offset, key, label in [(-.18, "ft0_accuracy", "Raw Dense BN"), (.18, "ft0_bn_clone_accuracy", "BN-recalibrated clone")]:
        axes[1].bar([i+offset for i in range(3)], [statistics.mean(r[key]*100 for r in rows if r["variant"]==v) for v in VARIANTS], .36, label=label)
    axes[1].set_xticks(range(3), VARIANTS)
    axes[1].set_ylabel("FT0 validation accuracy (%)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(ROOT/"comparison.png", dpi=170)
    plt.close(fig)
    lines = ["# 固定共享Mask權重補償：結果", "", f"更新：{common.stamp()}", "",
             "ResNet50/CIFAR-10、Tm8、flat l1固定共享mask、實際約90%conv-weight sparsity。original不補償，dense_input使用Dense patches重建，sequential使用實際student prefix逐層重建。三分支mask逐bit相同，離線不改BN/bias/分類器。", "",
             "## Validation結果", "", "| Seed | Variant | FT0 (%) | FT0 BN clone (%) | FT2 (%) | FT10 (%) | FT10 gain vs A (pp) |", "|---:|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['seed']} | {r['variant']} | {r['ft0_accuracy']*100:.3f} | {r['ft0_bn_clone_accuracy']*100:.3f} | {r['ft2_accuracy']*100:.3f} | {r['ft10_accuracy']*100:.3f} | {r['gain_ft10_vs_original_pp']:+.3f} |")
    lines += ["", "![訓練曲線與BN診斷](comparison.png)", "", "## 決策與限制", ""]
    if len(seeds)==1:
        lines += ["主方法C未達預設投入門檻：FT10比A高0.3pp，且不低於既有較強同seed控制。因此未擴seeds1/2，也未把B的可能優勢改稱C成功。0/2epochs結果只作低FT預算線索。"]
    else:
        gains = [r["gain_ft10_vs_original_pp"] for r in rows if r["variant"]=="sequential"]
        lines += [f"seed0通過投入門檻，已對稱補A/B/C三seeds。C相對A的平均差{statistics.mean(gains):+.3f}pp，sample std {statistics.stdev(gains):.3f}pp。這仍是validation確認，不是獨立test或跨模型結論。"]
    lines += ["", "check資料參與回退，不能視為獨立泛化證據。Dense與FT使用train，所有分類開發僅看validation；本輪未評估test。BN診斷只在clone執行，沒有改變主FT起點。補償的時間與記憶體是額外離線成本，不能以同epochs宣稱總成本完全相同。", "",
              "每層fit/check MSE、權重範圍、input偏移與回退數在各分支cluster_reconstruction.csv。q=0保持零，不新增連接。FP64 ridge解先cast回FP32才檢查；未把重建或ridge本身稱為新穎。", "",
              "## 量化狀態", "", "本報告只有浮點/AMP；oneDNN不視為硬體Q15 bit-exact。若浮點主方法成立，需先核對硬體參考模型覆蓋與相同INT8格式，才進量化，不能把硬體相容性當成已完成。", "",
              "## 可重現性", "", "完整mask、資料indices、來源hash、校正benchmark、全部epoch、checkpoint及問題紀錄均保留。VALIDATION.json檢查mask/零位/BN/資料隔離；A若有舊控制則要求最終state_dict逐bit相同。"]
    (ROOT/"FINAL_REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    value = dict(state="complete", completed_at=common.stamp(), seeds=seeds, new_ft_branches=len(rows),
                 promoted_seed0=len(seeds)>1, test_evaluated=False, int8_evaluated=False)
    common.write_json(ROOT/"study_result.json", value)
    common.write_json(ROOT/"status.json", value)
    event("study_complete", **{k:v for k,v in value.items() if k!="state"})


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    for name in ["logs", "source_snapshot"]:
        (ROOT/name).mkdir(exist_ok=True)
    lock = (ROOT/"study.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    sources = [Path(__file__), PROJECT/"scripts/weight_compensation_worker.py", PROJECT/"scripts/fixed_support_ridge.py",
               PROJECT/"scripts/shared_mask_repair.py", PROJECT/"scripts/run_shared_mask_repair.py",
               PROJECT/"scripts/task_loss_mask_search.py", PROJECT/"scripts/adapted_gradient_mask.py",
               PROJECT/"scripts/shared_gate_gradient.py", PROJECT/"scripts/run_fair_criterion_r0p9.py", common.CORE,
               PROJECT/"research_notes/2026-11-02_fixed_mask_weight_compensation_plan.md"]
    for source in sources:
        target = ROOT/"source_snapshot"/source.name
        if target.exists() and common.sha(source)!=common.sha(target):
            raise ValueError(f"Frozen source changed: {source}")
        shutil.copy2(source, target)
    references = strong_control()
    path = ROOT/"strong_controls.json"
    if path.exists() and json.loads(path.read_text())!=references:
        raise ValueError("Strong control sources changed")
    common.write_json(path, references)
    event("study_plan", primary="sequential", variants=VARIANTS, calibration_seed=20261107,
          promotion="C-A >=0.3pp and C >= best existing seed0 control; no test")
    run_batch([dict(mode="prepare", seed=0)])
    run_batch([dict(mode="train", seed=0, variant=v) for v in VARIANTS])
    seeds = [0]
    promote = promising()
    event("confirmation_decision", promote=promote)
    if promote:
        seeds += [1, 2]
        run_batch([dict(mode="prepare", seed=s) for s in [1, 2]])
        run_batch([dict(mode="train", seed=s, variant=v) for s in [1, 2] for v in VARIANTS])
    verify(seeds)
    render(seeds)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if ROOT.exists():
            common.write_json(ROOT/"status.json", dict(state="error", time=common.stamp(), error=repr(error)))
            event("study_error", error=repr(error))
        raise
