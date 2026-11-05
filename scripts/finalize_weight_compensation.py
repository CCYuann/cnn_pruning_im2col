#!/usr/bin/env python3
"""Audit fixed-mask compensation and package reproducible evidence, not raw caches."""

import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import zipfile
from pathlib import Path

import torch

import run_fair_criterion_r0p9 as common
import run_weight_compensation_study as study


def main():
    torch.set_num_threads(4)
    root = study.ROOT
    completed = json.loads((root/"study_result.json").read_text())
    if completed["state"] != "complete":
        raise RuntimeError("Finish all workers first")
    seeds = completed["seeds"]
    study.verify(seeds)
    common.write_json(root/"ENVIRONMENT.json", dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                      cudnn=torch.backends.cudnn.version(), solve_dtype="float64", stored_weight_dtype="float32",
                      fine_tune_amp="float16", threads=4,
                      gpus=subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).splitlines()))
    for record in json.loads((root/"strong_controls.json").read_text()):
        if common.sha(common.PROJECT/record["source"]) != record["sha256"]:
            raise ValueError("Promotion control source changed")
    cache_audit, coefficients = [], []
    for seed in seeds:
        cache = root/"calibration"/f"seed{seed}"
        ready = json.loads((cache/"ready.json").read_text())
        masks = torch.load(cache/"masks.pt", map_location="cpu", weights_only=False)
        for record, digest in zip(ready["layers"], ready["cache_files"]):
            path = cache/record["file"]
            if common.sha(path) != digest["sha256"]:
                raise ValueError("Calibration cache checksum mismatch")
            data = torch.load(path, map_location="cpu", weights_only=False)
            for split in ["fit", "check"]:
                part = data[split]
                pos = part["positions"]
                if len(pos) != 512 or part["x"].shape[1] != pos.numel() or part["y"].shape[1] != pos.numel():
                    raise ValueError("Teacher coordinate/sample alignment changed")
                if not all(len(row.unique()) == len(row) for row in pos):
                    raise ValueError("Repeated sampled coordinates within an image")
                if not torch.isfinite(part["x"]).all() or not torch.isfinite(part["y"]).all():
                    raise ValueError("Non-finite teacher data")
            cache_audit.append(dict(seed=seed, layer=record["name"], checksum=True, aligned_samples=True,
                                    unique_positions=True, fit_patches=data["fit"]["x"].shape[1]))
        dense = torch.load(common.checkpoint("resnet50", seed), map_location="cpu", weights_only=False)["model_state_dict"]
        for variant in study.VARIANTS:
            out = root/f"seed{seed}_{variant}"
            init = torch.load(out/"initialization.pt", map_location="cpu", weights_only=False)["model"]
            for name, mask in masks.items():
                weight = init[name+".weight"].flatten(1)
                original = dense[name+".weight"].flatten(1)
                full = mask.repeat_interleave(8, 0)[:len(weight)]
                if variant=="original" and not torch.equal(weight, original*full):
                    raise ValueError("Original initialization differs from Dense retained values")
                selected = weight[full]
                delta = selected-original[full]
                coefficients.append(dict(seed=seed, variant=variant, layer=name,
                                         planned_kept_weights=int(full.sum()), actual_nonzero_weights=int(torch.count_nonzero(weight)),
                                         kept_abs_max=float(selected.abs().max()) if selected.numel() else 0.,
                                         kept_l2=float(selected.norm()), delta_l2=float(delta.norm())))
            if variant != "original":
                records = common.read_csv(out/"cluster_reconstruction.csv")
                if any(not math.isfinite(float(r[k])) for r in records for k in ["fit_mse_before", "fit_mse_after", "check_mse_before", "check_mse_after"]):
                    raise ValueError("Non-finite reconstruction report")
                if variant=="sequential" and not any(float(r["input_fit_delta_rms"])>0 for r in records if r["layer"]!="conv1"):
                    raise ValueError("Sequential branch did not use changed student-prefix inputs")
    common.write_json(root/"CACHE_AUDIT.json", dict(passed=True, checks=cache_audit))
    common.write_csv(root/"initial_weight_ranges.csv", coefficients)
    paired = []
    for seed in seeds:
        a, b, c = [study.result(seed, v) for v in study.VARIANTS]
        paired.append(dict(seed=seed, original_ft10_pct=a["ft10_accuracy"]*100, dense_input_ft10_pct=b["ft10_accuracy"]*100,
                           sequential_ft10_pct=c["ft10_accuracy"]*100,
                           dense_input_minus_original_pp=(b["ft10_accuracy"]-a["ft10_accuracy"])*100,
                           sequential_minus_original_pp=(c["ft10_accuracy"]-a["ft10_accuracy"])*100,
                           sequential_minus_dense_input_pp=(c["ft10_accuracy"]-b["ft10_accuracy"])*100))
    common.write_csv(root/"paired_results.csv", paired)
    means = []
    for variant in study.VARIANTS:
        rows = [study.result(s, variant) for s in seeds]
        record = dict(variant=variant, seeds=len(seeds))
        for key in ["ft0_accuracy", "ft0_bn_clone_accuracy", "ft2_accuracy", "ft10_accuracy"]:
            values = [r[key]*100 for r in rows]
            record[key+"_pct_mean"] = statistics.mean(values)
            record[key+"_std_pp"] = statistics.stdev(values) if len(values)>1 else ""
        means.append(record)
    common.write_csv(root/"mean_results.csv", means)
    references = common.PROJECT.parent/"mask_aware_conv_accelerator/scripts"
    inspected = [references/n for n in ["quantization.py", "analyze_resnet34_quantization.py", "evaluate_bitexact_resnet34_backbone.py"]]
    common.write_json(root/"INT8_READINESS.json", dict(int8_evaluated=False,
        inspected_references=[dict(path=str(p), sha256=common.sha(p)) for p in inspected if p.exists()],
        inspected_path_supports_resnet50_bottleneck=False,
        reasons=["Inspected Conv-BN mapping only handles conv1/conv2; conv3 is not mapped",
                 "Inspected integer backbone executes two convolutions per block, not ResNet50's three",
                 "Scalar Q15 primitives exist, but do not establish full-model ResNet50 bit-exact coverage"],
        action="Do not substitute oneDNN for hardware Q15; synchronize or extend and verify a ResNet50 reference in a separately scoped step"))
    lines = ["", "## 三seed與權重補償機制分析", "",
             "| Variant | Seeds | FT0 mean (%) | BN clone FT0 mean (%) | FT2 mean (%) | FT10 mean (%) | FT10 sample std (pp) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in means:
        std = f"{r['ft10_accuracy_std_pp']:.3f}" if r["ft10_accuracy_std_pp"]!="" else "N/A"
        lines.append(f"| {r['variant']} | {r['seeds']} | {r['ft0_accuracy_pct_mean']:.3f} | {r['ft0_bn_clone_accuracy_pct_mean']:.3f} | {r['ft2_accuracy_pct_mean']:.3f} | {r['ft10_accuracy_pct_mean']:.3f} | {std} |")
    for key in ["dense_input_minus_original_pp", "sequential_minus_original_pp", "sequential_minus_dense_input_pp"]:
        values = [r[key] for r in paired]
        std = f"{statistics.stdev(values):.3f}" if len(values)>1 else "N/A"
        lines += ["", f"{key}：平均{statistics.mean(values):+.3f}pp，配對sample std {std}pp；逐seed {', '.join(f'{v:+.3f}' for v in values)}pp。"]
    lines += ["", "C相對A的改善與C是否優於B是兩個問題。即使權重補償整體有效，若C-B不一致或差距很小，也不能把student-prefix逐層版本的額外複雜度當作已成立貢獻。三seed validation仍不能直接當作獨立test、新穎性或跨模型保證。", "",
              "## INT8相容性待辦", "",
              "已找到另一個本機專案的Q15 scalar primitives與ResNet34 integer backbone，但已檢查路徑僅兩卷積block，未覆蓋ResNet50 Bottleneck conv3。未修改該專案或RTL，也未用oneDNN替代bit-exact結論；詳INT8_READINESS.json。須先取得或建立經驗證的ResNet50參考，才可做同規約INT8比較。", "",
              "## 檢查與封裝", "", "UNIT_TESTS.txt保存全測試輸出；VALIDATION.json及CACHE_AUDIT.json檢查相同mask、禁止位置、BN/bias/classifier不變、校正資料隔離與teacher樣本對齊。initial_weight_ranges.csv區分mask計畫保留數與實際數值非零數。", "",
              "ZIP保留mask、indices、每層/每epoch數據、圖表、全部log與程式快照；不含每seed約2.42GiB的teacher cache、校正images或大型模型權重。這些仍保留本機；重跑需要原repository、CIFAR與Dense checkpoints。"]
    path = root/"FINAL_REPORT.md"
    original = path.read_text()
    marker = "\n## 三seed與權重補償機制分析"
    path.write_text(original.split(marker)[0]+"\n".join(lines)+"\n", encoding="utf-8")
    test = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], cwd=common.PROJECT,
                          env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4"), text=True, capture_output=True)
    (root/"UNIT_TESTS.txt").write_text(test.stdout+test.stderr, encoding="utf-8")
    test.check_returncode()
    supplemental = [common.PROJECT/"scripts"/n for n in ["run_adapted_gradient_study.py", "spatial_quota_pruning.py",
                    "run_spatial_quota_study.py", "sensitivity_budget_pruning.py"]]
    supplemental += list((common.PROJECT/"tests").glob("test_*.py")) + [Path(__file__)]
    for source in supplemental:
        target = root/"source_snapshot"/source.name
        if target.exists() and common.sha(target)!=common.sha(source):
            raise ValueError("Frozen snapshot differs from current supplemental source")
        if not target.exists():
            shutil.copy2(source, target)
    files = [p for p in root.rglob("*") if p.is_file() and p.name not in {"study.lock", "MANIFEST.json"}
             and (p.suffix!=".pt" or p.name=="masks.pt")]
    common.write_json(root/"MANIFEST.json", {str(p.relative_to(root)):common.sha(p) for p in files})
    archive = root.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files+[root/"MANIFEST.json"]:
            z.write(p, p.relative_to(root.parent))
    archive.with_suffix(".zip.sha256").write_text(common.sha(archive)+"  "+archive.name+"\n", encoding="ascii")
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("Archive integrity error")
    for name, digest in json.loads((root/"MANIFEST.json").read_text()).items():
        if common.sha(root/name)!=digest:
            raise ValueError("Archive manifest changed")
    print(json.dumps(dict(passed=True, seeds=seeds, files=len(files), archive_mb=archive.stat().st_size/1048576, test_evaluated=False)))


if __name__ == "__main__":
    main()
