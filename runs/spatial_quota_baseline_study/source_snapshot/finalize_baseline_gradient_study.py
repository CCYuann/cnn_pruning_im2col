#!/usr/bin/env python3
"""Audit both completed stages and package evidence without large model checkpoints."""

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

import run_adapted_gradient_study as gradient
import run_fair_criterion_r0p9 as common
import run_spatial_quota_study as baseline


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def main():
    torch.set_num_threads(4)
    base, grad = baseline.ROOT, gradient.ROOT
    decision = json.loads((base/"decision.json").read_text())
    completed = json.loads((grad/"study_result.json").read_text())
    if decision["state"] != "baseline_complete" or completed["state"] != "complete":
        raise RuntimeError("Finish both stages before packaging")
    seeds = completed["seeds"]
    gradient.verify(seeds)
    jobs = [dict(model=m, seed=s, criterion=c, granularity=g) for m in ["resnet34", "resnet50"]
            for s in ([0, 1, 2] if m in decision["expanded_models"] else [0])
            for c in baseline.CRITERIA for g in ["flat", "spatial"]]
    baseline.verify(jobs)
    audit = []
    for seed in seeds:
        first = torch.load(grad/f"seed{seed}_continuous_control/post_adaptation.pt", map_location="cpu", weights_only=False)
        for variant in gradient.VARIANTS:
            out = grad/f"seed{seed}_{variant}"
            second = torch.load(out/"post_adaptation.pt", map_location="cpu", weights_only=False)
            if not equal(first, second):
                raise ValueError("Post-adaptation model/optimizer/scaler/RNG differs")
            config = json.loads((out/"configuration.json").read_text())
            for key, path in [("worker_sha256", common.PROJECT/"scripts/adapted_gradient_mask.py"),
                              ("helper_sha256", common.PROJECT/"scripts/shared_gate_gradient.py"),
                              ("evaluator_sha256", common.PROJECT/"scripts/task_loss_mask_search.py"),
                              ("core_sha256", common.CORE), ("checkpoint_sha256", common.checkpoint("resnet50", seed)),
                              ("initial_masks_sha256", base/f"resnet50_seed{seed}_l1_sum_flat/masks.pt")]:
                if config[key] != common.sha(path):
                    raise ValueError(f"Source hash changed: {path}")
            masks = torch.load(out/"masks.pt", map_location="cpu", weights_only=False)
            swaps = sum(int((masks["initial"][n] & ~m).sum()) for n, m in masks["selected"].items())
            audit.append(dict(seed=seed, variant=variant, identical_full_adaptation_state=True,
                              source_hashes=True, actual_applied_swaps=swaps))
    common.write_json(grad/"ADDITIONAL_AUDIT.json", dict(passed=True, checks=audit))
    paper = common.PROJECT.parent/"Cross-Filter_Structured_Pruning_for_Efficient_Sparse_CNN_Acceleration.pdf"
    common.write_json(base/"PAPER_SOURCE.json", dict(filename=paper.name, sha256=common.sha(paper),
                      reviewed_pdf_pages=[5, 6], printed_pages=[129465, 129466], doi="10.1109/ACCESS.2025.3587027"))
    test = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=common.PROJECT,
                          env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"), text=True, capture_output=True)
    test.check_returncode()
    sources = [common.PROJECT/"scripts"/name for name in ["spatial_quota_pruning.py", "run_spatial_quota_study.py",
               "adapted_gradient_mask.py", "run_adapted_gradient_study.py", "shared_gate_gradient.py", "task_loss_mask_search.py",
               "run_shared_mask_repair.py", "shared_mask_repair.py", "run_fair_criterion_r0p9.py", "sensitivity_budget_pruning.py"]]
    sources += list((common.PROJECT/"tests").glob("test_*.py")) + [Path(__file__)]
    for root in [base, grad]:
        (root/"UNIT_TESTS.txt").write_text(test.stdout+test.stderr, encoding="utf-8")
        for source in sources:
            target = root/"source_snapshot"/source.name
            if target.exists() and common.sha(source) != common.sha(target):
                raise ValueError(f"Frozen snapshot differs: {target}")
            if not target.exists():
                shutil.copy2(source, target)
    summary = common.PROJECT/"runs/BASELINE_AND_GRADIENT_20261102.md"
    lines = ["# 2026-11-02：配額基準與適應後梯度搜尋總結", "", f"整理：{common.stamp()}", "",
             "本輪完成原文圖示／公式審閱、兩模型8個基準分支，再依預設規則完成ResNet50四個2+8epoch分支。全為seed0 validation，沒有使用test，沒有宣稱統計顯著。", "",
             "## 第一階段：配額粒度", "", "| Model | Criterion | Flat (%) | Spatial (%) | Spatial-flat (pp) |", "|---|---|---:|---:|---:|"]
    for row in common.read_csv(base/"paired_summary.csv"):
        lines.append(f"| {row['model']} | {row['criterion']} | {float(row['flat_val_pct']):.2f} | {float(row['spatial_val_pct']):.2f} | {float(row['spatial_minus_flat_pp']):+.2f} |")
    lines += ["", "![配額比較](spatial_quota_baseline_study/comparison.png)", "",
              "兩版每cluster總kept_count精確相同，1×1 mask完全相同；flat與既有核心逐元素一致。最大差0.26pp，未達預先0.3pp擴大門檻；不代表兩版本等效。spatial是對原文歧義的操作性解讀，不是已確認作者實作。", "",
              "本設定的配額粒度差距，不能直接解釋先前與原論文數個百分點的raw accuracy差；模型入口與完整訓練設定等仍未對齊。", "",
              "## 第二階段：適應後共享Gate梯度", "", "| Method | Validation (%) | 實際交換對數 |", "|---|---:|---:|"]
    for row in common.read_csv(grad/"validation_results.csv"):
        swaps = next(r["actual_applied_swaps"] for r in audit if r["seed"] == int(row["seed"]) and r["variant"] == row["variant"])
        lines.append(f"| {row['variant']} (seed{row['seed']}) | {float(row['final_validation_acc'])*100:.2f} | {swaps} |")
    lines += ["", "![梯度比較](adapted_gradient_mask_study/comparison.png)", "",
              "主方法gradient_gate未超過本輪最佳基準spatial top2的93.68%，也未超過匹配交換數的隨機控制。因此未擴三seed，不從消融挑贏家，不再改交換數或適應epoch。", "",
              "31項unit tests通過。continuous_control最終state_dict與原10epoch l1逐bit相同；四分支適應2epochs後model、optimizer、AMP scaler及完整RNG相同。三個搜尋分支產生的候選／gate／random plans逐元素相同。配額與資料隔離亦通過。", "",
              "報表曾有重複seed keyword TypeError，worker完成不受影響；已修正並測試，原controller保留於revisions，SHA與恢復事件記錄完整。沒有模型重跑、OOM或kill。", "",
              "## 研究判斷", "", "本輪完成了基準審核及一個依結果啟動的新候選機制，但未取得可靠的分類優勢。不能宣稱shared-mask方法無效；也不能把數學正確導數、單seed小提升或較低search CE當作有效論文貢獻。", "",
              "目前不建議再擴大這個單次梯度交換版本。若下一輪仍追精度，應另立有明確比較預算的權重補償或訓練期mask更新假說，並先比對既有文獻；本輪未啟動未凍結的新實驗。", "",
              "## 檔案與重現", "", "[原文審核](spatial_quota_baseline_study/PAPER_AUDIT.md)、[基準報告](spatial_quota_baseline_study/BASELINE_REPORT.md)、[梯度報告](adapted_gradient_mask_study/FINAL_REPORT.md)。兩資料夾均有CSV、mask、log、問題紀錄、測試輸出及程式快照。", "",
              "ZIP包含本總結及兩批證據，不包含CIFAR、Dense checkpoints或大型訓練權重。重跑訓練仍需原repository、資料與checkpoints；本機大型檔案未刪除。"]
    summary.write_text("\n".join(lines)+"\n", encoding="utf-8")
    keep_models = {"masks.pt", "mask_audit.pt", "mask_plans.pt"}
    files = [summary]
    for root in [base, grad]:
        files.extend(p for p in root.rglob("*") if p.is_file() and p.name not in {"study.lock", "MANIFEST.json"}
                     and (p.suffix != ".pt" or p.name in keep_models))
    manifest = common.PROJECT/"runs/baseline_and_gradient_20261102_manifest.json"
    common.write_json(manifest, {str(p.relative_to(common.PROJECT/"runs")): common.sha(p) for p in files})
    archive = common.PROJECT/"runs/baseline_and_gradient_20261102.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in files+[manifest]:
            z.write(path, path.relative_to(common.PROJECT/"runs"))
    archive.with_suffix(".zip.sha256").write_text(common.sha(archive)+"  "+archive.name+"\n", encoding="ascii")
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("Archive integrity error")
    for name, digest in json.loads(manifest.read_text()).items():
        if common.sha(common.PROJECT/"runs"/name) != digest:
            raise ValueError(f"Manifest mismatch {name}")
    print(json.dumps(dict(passed=True, files=len(files), archive_mb=archive.stat().st_size/1048576, test_evaluated=False)))


if __name__ == "__main__":
    main()
