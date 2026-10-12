#!/usr/bin/env python3
"""Summarize the unified fixed-checkpoint target-0.8 experiments."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("runs/fixed_checkpoint_r0p8")
SUMMARY = ROOT / "summary"
METHOD_LABELS = {
    "Global unstructured": "Global unstructured",
    "Cross-filter l1_sum (global_cluster)": "Cross l1_sum",
    "Cross-filter top2_mean_abs (global_cluster+cap0.75+sens4@0.2+skip1)": (
        "Cross top2 + sensitivity"
    ),
}
MODEL_ORDER = ["resnet34", "resnet50", "googlenet", "densenet121", "mobilenet_v2"]
MODEL_LABELS = {
    "resnet34": "ResNet34",
    "resnet50": "ResNet50",
    "googlenet": "GoogLeNet",
    "densenet121": "DenseNet121",
    "mobilenet_v2": "MobileNetV2",
}


def load_results():
    frames = []
    for path in sorted(ROOT.glob("*/*_results.csv")):
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        raise SystemExit(f"No per-run result CSV files found under {ROOT}")
    return pd.concat(frames, ignore_index=True)


def summarize(data):
    pruned = data[data["method"].isin(METHOD_LABELS)].copy()
    pruned["variant"] = pruned["method"].map(METHOD_LABELS)
    pruned["accuracy_loss_pp"] = -100.0 * pruned["accuracy_delta_vs_dense_extra"]
    result = (
        pruned.groupby(["model", "variant", "tm"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            direct_accuracy_mean=("direct_acc", "mean"),
            direct_accuracy_std=("direct_acc", "std"),
            final_accuracy_mean=("after_extra_or_ft_acc", "mean"),
            final_accuracy_std=("after_extra_or_ft_acc", "std"),
            accuracy_loss_pp_mean=("accuracy_loss_pp", "mean"),
            accuracy_loss_pp_std=("accuracy_loss_pp", "std"),
            actual_pruning_ratio_mean=("pruning_ratio", "mean"),
            mac_reduction_mean=("mac_reduction", "mean"),
            cycle_reduction_mean=("cycle_reduction", "mean"),
            estimated_speedup_mean=("estimated_speedup", "mean"),
        )
        .sort_values(["model", "variant", "tm"])
    )
    result["under_1pp_mean"] = result["accuracy_loss_pp_mean"] < 1.0
    return result


def dense_summary(data):
    dense = data[["model", "seed", "dense_test_acc", "dense_extra_test_acc"]].drop_duplicates()
    return dense.groupby("model", as_index=False).agg(
        seeds=("seed", "nunique"),
        dense_accuracy_mean=("dense_test_acc", "mean"),
        dense_accuracy_std=("dense_test_acc", "std"),
        dense_extra_accuracy_mean=("dense_extra_test_acc", "mean"),
        dense_extra_accuracy_std=("dense_extra_test_acc", "std"),
    )


def accuracy_plot(summary):
    variants = ["Cross l1_sum", "Cross top2 + sensitivity"]
    colors = {"Cross l1_sum": "#3973ac", "Cross top2 + sensitivity": "#d97932"}
    x = np.arange(len(MODEL_ORDER))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, tm in zip(axes, [8, 16]):
        part = summary[summary["tm"] == tm].set_index(["model", "variant"])
        for index, variant in enumerate(variants):
            means = [part.loc[(model, variant), "accuracy_loss_pp_mean"] for model in MODEL_ORDER]
            stds = [part.loc[(model, variant), "accuracy_loss_pp_std"] for model in MODEL_ORDER]
            ax.bar(
                x + (index - 0.5) * width,
                means,
                width,
                yerr=stds,
                capsize=3,
                color=colors[variant],
                edgecolor="#333333",
                linewidth=0.5,
                label=variant,
            )
        ax.axhline(1.0, color="#b3261e", linestyle="--", linewidth=1.4, label="1 pp limit")
        ax.axhline(0.0, color="#555555", linewidth=0.7)
        ax.set_title(f"Tm = {tm}")
        ax.set_xticks(x, [MODEL_LABELS[m] for m in MODEL_ORDER], rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Accuracy loss vs Dense extra (percentage points)")
    handles, labels = axes[1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Fixed-checkpoint CIFAR-10 pruning at target ratio 0.8", y=1.02)
    fig.tight_layout()
    fig.savefig(SUMMARY / "accuracy_loss_by_model_tm.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def tradeoff_plot(summary):
    variants = ["Cross l1_sum", "Cross top2 + sensitivity"]
    colors = {"Cross l1_sum": "#3973ac", "Cross top2 + sensitivity": "#d97932"}
    markers = {8: "o", 16: "s"}
    fig, ax = plt.subplots(figsize=(9, 6))
    for variant in variants:
        for tm in [8, 16]:
            part = summary[(summary["variant"] == variant) & (summary["tm"] == tm)]
            ax.scatter(
                100.0 * part["actual_pruning_ratio_mean"],
                part["accuracy_loss_pp_mean"],
                s=75,
                color=colors[variant],
                marker=markers[tm],
                edgecolor="#222222",
                linewidth=0.6,
                label=f"{variant}, Tm={tm}",
            )
            for row in part.itertuples():
                ax.annotate(
                    MODEL_LABELS[row.model],
                    (100.0 * row.actual_pruning_ratio_mean, row.accuracy_loss_pp_mean),
                    xytext=(5, 4),
                    textcoords="offset points",
                    fontsize=8,
                )
    ax.axhline(1.0, color="#b3261e", linestyle="--", linewidth=1.4, label="1 pp limit")
    ax.set_xlabel("Actual weight sparsity (%)")
    ax.set_ylabel("Accuracy loss vs Dense extra (percentage points)")
    ax.set_title("Target 0.8 accuracy-sparsity trade-off")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(SUMMARY / "accuracy_sparsity_tradeoff.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    SUMMARY.mkdir(parents=True, exist_ok=True)
    data = load_results()
    summary = summarize(data)
    dense = dense_summary(data)
    cross = summary[summary["variant"].str.startswith("Cross")].copy()
    data.to_csv(ROOT / "all_results_combined.csv", index=False)
    summary.to_csv(SUMMARY / "all_methods_r0p8_summary.csv", index=False)
    cross.to_csv(SUMMARY / "cross_filter_under_1pp_check.csv", index=False)
    dense.to_csv(SUMMARY / "dense_summary.csv", index=False)
    accuracy_plot(summary)
    tradeoff_plot(summary)
    print(cross[["model", "variant", "tm", "accuracy_loss_pp_mean", "actual_pruning_ratio_mean", "under_1pp_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
