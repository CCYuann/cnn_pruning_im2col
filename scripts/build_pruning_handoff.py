#!/usr/bin/env python3
"""Package pruning evidence, regenerate comparison tables, and verify its provenance."""

import csv
import hashlib
import json
import math
import shutil
import statistics
import subprocess
import zipfile
from collections import defaultdict
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
BUNDLE = PROJECT / "handoffs/pruning_contributions_20261027"
ALLOWED = {".csv", ".png", ".md", ".json"}
MODELS = ["resnet34", "resnet50", "googlenet", "densenet121", "mobilenet_v2"]
SUMMARY = "runs/fixed_checkpoint_all_ratio_summary/cross_filter_summary.csv"
RAW = "runs/fixed_checkpoint_all_ratio_summary/all_results_combined.csv"
ROOTS = [
    "runs/fixed_checkpoint_all_ratio_summary",
    "runs/fixed_checkpoint_resnet34",
    "runs/fixed_checkpoint_multi_model/pruning_r0p7",
    "runs/fixed_checkpoint_multi_model/sensitivity_validation",
    "runs/fixed_checkpoint_r0p8",
    "runs/fixed_checkpoint_resnet34_r0p85_r0p90",
    "runs/fixed_checkpoint_high_ratio_sweep",
    "runs/improvement_topk_protect_cifar10_split",
    "runs/sensitivity_aware_topk_cifar10_split",
    "runs/sensitivity_aware_l1_sum_cifar10_split",
    "runs/layer_sensitivity_summary",
    "runs/filter_similarity_grouping",
]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual, expected, label):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"Inconsistent {label}: {actual} != {expected}")


def copy_evidence():
    files = set()
    for relative in ROOTS:
        root = PROJECT / relative
        if not root.is_dir():
            raise FileNotFoundError(root)
        files.update(
            path for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in ALLOWED
            and not {"logs", "locks", "checkpoints"}.intersection(path.relative_to(root).parts)
        )
    files.update(PROJECT.glob("*.py"))
    files.update(PROJECT.glob("run_gpu_cifar10.*"))
    files.update((PROJECT / "scripts").glob("*.py"))
    files.update((PROJECT / "scripts").glob("*.sh"))
    files.update(PROJECT / name for name in [
        "requirements.txt", "research_improvement_ideas.md", "runs/README.md",
        "runs/fixed_checkpoint_multi_model/pruning_r0p7/README.md",
        "runs/filter_similarity_grouping_invalid_mask_remap/README.md",
    ])
    for path in sorted(files):
        destination = BUNDLE / path.relative_to(PROJECT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    shutil.copy2(PROJECT / "README.md", BUNDLE / "SOURCE_PROJECT_README.md")
    return len(files) + 1


def validate_and_summarize():
    raw, summary = read_csv(BUNDLE / RAW), read_csv(BUNDLE / SUMMARY)
    groups = defaultdict(list)
    all_groups = defaultdict(list)
    for row in raw:
        key = (row["model"], float(row["target_prune_ratio"]), int(row["tm"]))
        all_groups[key + (int(row["seed"]),)].append(row)
        if row["criterion"] in {"l1_sum", "top2_mean_abs"}:
            groups[key + (row["criterion"],)].append(row)
    for key, rows in all_groups.items():
        if len(rows) != 5 or len({r["checkpoint_id"] for r in rows}) != 1:
            raise ValueError(f"Invalid checkpoint pairing: {key}")
        baseline = next(r for r in rows if r["method"] == "Dense extra training")
        for row in rows:
            close(row["dense_extra_test_acc"], baseline["after_extra_or_ft_acc"], str(key))
    checked = 0
    for row in summary:
        criterion = "l1_sum" if row["variant"] == "Cross l1_sum" else "top2_mean_abs"
        key = (row["model"], float(row["target_prune_ratio"]), int(row["tm"]), criterion)
        samples = groups.pop(key)
        if len(samples) != 3 or {int(r["seed"]) for r in samples} != {0, 1, 2}:
            raise ValueError(f"Invalid seed coverage: {key}")
        accs = [float(r["after_extra_or_ft_acc"]) for r in samples]
        losses = [(float(r["dense_extra_test_acc"]) - a) * 100 for r, a in zip(samples, accs)]
        for column, values in [
            ("final_accuracy", accs), ("accuracy_loss_pp", losses),
        ]:
            close(row[column + "_mean"], statistics.mean(values), str(key))
            close(row[column + "_std"], statistics.stdev(values), str(key))
        for column, source in [
            ("actual_pruning_ratio_mean", "pruning_ratio"),
            ("mac_reduction_mean", "mac_reduction"),
            ("estimated_speedup_mean", "estimated_speedup"),
        ]:
            close(row[column], statistics.mean(float(r[source]) for r in samples), str(key))
        checked += 1
    if groups:
        raise ValueError("Unaccounted raw cross-filter groups")
    source_files = {r["source_file"] for r in raw}
    for source in source_files:
        if not (BUNDLE / source).is_file():
            raise FileNotFoundError(source)

    selected = []
    ranking_paths = sorted((BUNDLE / "runs/fixed_checkpoint_resnet34/sensitivity_validation").glob("*/layer_sensitivity_ranked.csv"))
    ranking_paths += sorted((BUNDLE / "runs/fixed_checkpoint_multi_model/sensitivity_validation").glob("*/layer_sensitivity_ranked.csv"))
    for path in ranking_paths:
        rows = [r for r in read_csv(path) if float(r["mean_drop_pp"]) >= 0]
        rows.sort(key=lambda r: float(r["mean_drop_pp"]), reverse=True)
        for rank, row in enumerate(rows[:4], start=1):
            if row["evaluation_split"] != "validation":
                raise ValueError(f"Non-validation sensitivity ranking: {path}")
            selected.append({
                "model": row["model"], "seed": row["seed"], "tm": row["tm"],
                "rank": rank, "layer_name": row["layer_name"],
                "kernel": row["kernel"], "is_depthwise": row["is_depthwise"],
                "mean_drop_pp": row["mean_drop_pp"], "prune_ratio_cap": 0 if rank == 1 else 0.2,
                "source_file": str(path.relative_to(BUNDLE)),
            })
        for row in read_csv(path.with_name("layer_sensitivity.csv")):
            if row["evaluation_split"] != "validation" or not row["checkpoint"].endswith("-e100"):
                raise ValueError(f"Unexpected sensitivity provenance: {path}")
    if len(ranking_paths) != 30 or len(selected) != 120:
        raise ValueError(f"Unexpected formal sensitivity coverage: {len(ranking_paths)}")
    write_csv(BUNDLE / "tables/selected_sensitive_layers.csv", selected)

    pairs = read_csv(BUNDLE / "runs/filter_similarity_grouping/paired_results.csv")
    if len(pairs) != 45:
        raise ValueError("Expected 45 valid similarity pairs")
    for row in pairs:
        close(row["ft_gain_pp"], (float(row["after_extra_or_ft_acc_similarity"]) - float(row["after_extra_or_ft_acc_contiguous"])) * 100, "similarity gain")
    return raw, summary, {
        "status": "passed", "raw_rows": len(raw), "raw_source_files": len(source_files),
        "formal_cross_filter_summary_groups_recomputed": checked,
        "required_seeds_per_group": [0, 1, 2], "sample_std_ddof": 1,
        "within_run_checkpoint_and_dense_extra_pairing": "passed",
        "sensitivity_rankings": len(ranking_paths), "protected_layer_entries": len(selected),
        "formal_sensitivity_validation_split_and_e100_checkpoint_ids": "passed",
        "valid_similarity_pairs": len(pairs),
        "notes": ["Checks verify recorded data consistency; no GPU inference or training was rerun.",
                  "Checkpoint IDs were checked; model checkpoint binaries are not included.",
                  "Early exploratory results were copied for provenance, not certified as formal controls."],
    }


def generate_tables(summary):
    rows = []
    for model in MODELS:
        selected = [r for r in summary if r["model"] == model]
        selected.sort(key=lambda r: (float(r["target_prune_ratio"]), int(r["tm"]), r["variant"]))
        for r in selected:
            rows.append({
                "model": model, "target_prune_pct": float(r["target_prune_ratio"]) * 100,
                "tm": int(r["tm"]), "variant": r["variant"], "seeds": int(r["seeds"]),
                "dense_extra_acc_pct": float(r["dense_extra_accuracy_mean"]) * 100,
                "ft_acc_pct": float(r["final_accuracy_mean"]) * 100,
                "ft_acc_std_pp": float(r["final_accuracy_std"]) * 100,
                "loss_pp": float(r["accuracy_loss_pp_mean"]),
                "loss_std_pp": float(r["accuracy_loss_pp_std"]),
                "actual_prune_pct": float(r["actual_pruning_ratio_mean"]) * 100,
                "mac_reduction_pct": float(r["mac_reduction_mean"]) * 100,
                "estimated_speedup": float(r["estimated_speedup_mean"]),
            })
    write_csv(BUNDLE / "tables/formal_comparison_percent_units.csv", rows)
    lines = ["# 正式實驗數據表", "", "來源：`runs/fixed_checkpoint_all_ratio_summary/cross_filter_summary.csv`，數值由打包腳本直接換算。", "",
             "FT accuracy 與 actual sparsity 為百分比；± 為三 seed sample std，單位 pp。Loss 相對同 seed Dense extra。Speedup 是軟體估算。", "",
             "## Target=70%", "",
             "| Model | Tm | Method | FT accuracy (%) ± std | Loss (pp) | Actual sparsity (%) | MAC reduction (%) | Estimated speedup |",
             "|---|---:|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["target_prune_pct"] == 70:
            lines.append(f"| {r['model']} | {r['tm']} | {r['variant']} | {r['ft_acc_pct']:.3f} ± {r['ft_acc_std_pp']:.3f} | {r['loss_pp']:.3f} | {r['actual_prune_pct']:.3f} | {r['mac_reduction_pct']:.3f} | {r['estimated_speedup']:.3f}x |")
    lines += ["", "## 全部正式 Cross-filter 結果", "", "同 target 不代表同 actual sparsity；請同時閱讀各欄。", "",
              "| Model | Target (%) | Tm | Method | FT acc (%) | Loss ± std (pp) | Actual sparsity (%) | MAC reduction (%) |",
              "|---|---:|---:|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['target_prune_pct']:.0f} | {r['tm']} | {r['variant']} | {r['ft_acc_pct']:.3f} | {r['loss_pp']:.3f} ± {r['loss_std_pp']:.3f} | {r['actual_prune_pct']:.3f} | {r['mac_reduction_pct']:.3f} |")
    lines += ["", "## 原始圖表", "",
              "![Actual sparsity / accuracy trade-off](../runs/fixed_checkpoint_all_ratio_summary/accuracy_sparsity_tradeoffs.png)", "",
              "![L1 accuracy loss curves](../runs/fixed_checkpoint_all_ratio_summary/l1_accuracy_loss_curves.png)", "",
              "![L1 accuracy loss heatmap](../runs/fixed_checkpoint_all_ratio_summary/l1_accuracy_loss_heatmap.png)", "",
              "## 相似分組備選結果", "", "以下是修正版結果；目前預定主方法不採用相似分組。FT gain 正值代表相似分組較好。", "",
              "| Model | Target (%) | Seeds | Contiguous loss (pp) | Similarity loss (pp) | FT gain ± std (pp) |", "|---|---:|---:|---:|---:|---:|"]
    for r in read_csv(BUNDLE / "runs/filter_similarity_grouping/summary.csv"):
        lines.append(f"| {r['model']} | {float(r['target_prune_ratio'])*100:.0f} | {r['seeds']} | {float(r['contiguous_loss_pp_mean']):.3f} | {float(r['similarity_loss_pp_mean']):.3f} | {float(r['ft_gain_pp_mean']):+.3f} ± {float(r['ft_gain_pp_std']):.3f} |")
    (BUNDLE / "tables/RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    if not (BUNDLE / "README.md").is_file():
        raise FileNotFoundError("The authored handoff README must exist before packaging")
    (BUNDLE / "tables").mkdir(exist_ok=True)
    copied = copy_evidence()
    _, summary, checks = validate_and_summarize()
    generate_tables(summary)
    write_json(BUNDLE / "VALIDATION.json", checks)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT, text=True).strip()
    files = []
    for path in sorted(BUNDLE.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            relative = path.relative_to(BUNDLE)
            original = PROJECT / relative
            same_source = original.is_file() and original.resolve() != path.resolve() and digest(original) == digest(path)
            files.append({"path": str(relative), "bytes": path.stat().st_size, "sha256": digest(path),
                          "source": str(relative) if same_source else ("README.md" if path.name == "SOURCE_PROJECT_README.md" else "handoff_authored_or_generated")})
    write_json(BUNDLE / "MANIFEST.json", {
        "date": "2026-10-27", "repository": "https://github.com/CCYuann/cnn_pruning_im2col",
        "base_commit": commit, "snapshot_includes_uncommitted_files": bool(dirty),
        "copied_evidence_files": copied, "file_count_excluding_manifest": len(files), "files": files,
    })
    archive = BUNDLE.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        for path in sorted(BUNDLE.rglob("*")):
            if path.is_file():
                output.write(path, path.relative_to(BUNDLE.parent))
    with zipfile.ZipFile(archive) as output:
        if output.testzip() is not None:
            raise ValueError("ZIP integrity check failed")
    checksum = digest(archive)
    archive.with_suffix(".zip.sha256").write_text(f"{checksum}  {archive.name}\n", encoding="ascii")
    print(json.dumps({"bundle": str(BUNDLE), "archive": str(archive), "archive_mb": round(archive.stat().st_size / 1048576, 2),
                      "packaged_files": len(files) + 1, "sha256": checksum, "validation": checks}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
