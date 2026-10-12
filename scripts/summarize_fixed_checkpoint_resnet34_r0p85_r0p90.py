#!/usr/bin/env python3
"""Summarize fixed-checkpoint ResNet34 experiments at target ratios 0.85 and 0.90."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("runs/fixed_checkpoint_resnet34_r0p85_r0p90")
SUMMARY = ROOT / "summary"
METHOD_LABELS = {
    "Global unstructured": "Global unstructured",
    "Cross-filter l1_sum (global_cluster)": "Cross l1_sum",
    "Cross-filter top2_mean_abs (global_cluster+cap0.75+sens4@0.2+skip1)": (
        "Cross top2 + sensitivity"
    ),
}


def load_results():
    paths = sorted(ROOT.glob("*_results.csv"))
    if len(paths) != 12:
        raise SystemExit(f"Expected 12 result files, found {len(paths)} under {ROOT}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize(data):
    pruned = data[data["method"].isin(METHOD_LABELS)].copy()
    pruned["variant"] = pruned["method"].map(METHOD_LABELS)
    pruned["accuracy_loss_pp"] = -100.0 * pruned["accuracy_delta_vs_dense_extra"]
    result = (
        pruned.groupby(["target_prune_ratio", "variant", "tm"], as_index=False)
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
        .sort_values(["target_prune_ratio", "variant", "tm"])
    )
    result["under_1pp_mean"] = result["accuracy_loss_pp_mean"] < 1.0
    return result


def plot_accuracy_loss(summary):
    variants = ["Cross l1_sum", "Cross top2 + sensitivity"]
    colors = {"Cross l1_sum": "#3973ac", "Cross top2 + sensitivity": "#d97932"}
    markers = {8: "o", 16: "s"}
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for variant in variants:
        for tm in [8, 16]:
            part = summary[(summary["variant"] == variant) & (summary["tm"] == tm)]
            ax.errorbar(
                100.0 * part["target_prune_ratio"],
                part["accuracy_loss_pp_mean"],
                yerr=part["accuracy_loss_pp_std"],
                color=colors[variant],
                marker=markers[tm],
                linewidth=1.8,
                capsize=4,
                label=f"{variant}, Tm={tm}",
            )
    ax.axhline(1.0, color="#b3261e", linestyle="--", linewidth=1.4, label="1 pp limit")
    ax.set_xticks([85, 90])
    ax.set_xlabel("Target pruning ratio (%)")
    ax.set_ylabel("Accuracy loss vs Dense extra (percentage points)")
    ax.set_title("ResNet34 fixed-checkpoint pruning")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(SUMMARY / "accuracy_loss_r0p85_r0p90.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(summary):
    variants = ["Cross l1_sum", "Cross top2 + sensitivity"]
    colors = {"Cross l1_sum": "#3973ac", "Cross top2 + sensitivity": "#d97932"}
    markers = {8: "o", 16: "s"}
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for variant in variants:
        for tm in [8, 16]:
            part = summary[(summary["variant"] == variant) & (summary["tm"] == tm)]
            ax.plot(
                100.0 * part["actual_pruning_ratio_mean"],
                part["accuracy_loss_pp_mean"],
                color=colors[variant],
                marker=markers[tm],
                linewidth=1.8,
                label=f"{variant}, Tm={tm}",
            )
    ax.axhline(1.0, color="#b3261e", linestyle="--", linewidth=1.4, label="1 pp limit")
    ax.set_xlabel("Actual weight sparsity (%)")
    ax.set_ylabel("Accuracy loss vs Dense extra (percentage points)")
    ax.set_title("ResNet34 accuracy-sparsity trade-off")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(SUMMARY / "accuracy_sparsity_tradeoff.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    SUMMARY.mkdir(parents=True, exist_ok=True)
    data = load_results()
    summary = summarize(data)
    cross = summary[summary["variant"].str.startswith("Cross")].copy()
    data.to_csv(ROOT / "all_results_combined.csv", index=False)
    summary.to_csv(SUMMARY / "all_methods_summary.csv", index=False)
    cross.to_csv(SUMMARY / "cross_filter_summary.csv", index=False)
    plot_accuracy_loss(summary)
    plot_tradeoff(summary)
    print(
        cross[
            [
                "target_prune_ratio",
                "variant",
                "tm",
                "accuracy_loss_pp_mean",
                "actual_pruning_ratio_mean",
                "under_1pp_mean",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
