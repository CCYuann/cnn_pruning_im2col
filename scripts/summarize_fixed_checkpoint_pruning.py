#!/usr/bin/env python3
"""Summarize the fixed-checkpoint ResNet34 pruning comparison."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("runs/fixed_checkpoint_resnet34/pruning_comparison")
INPUT = ROOT / "all_results_combined.csv"
CHARTS = ROOT / "charts"


def variant_name(row):
    method = row["method"]
    if method == "Dense baseline":
        return "Dense"
    if method == "Dense extra training":
        return "Dense extra"
    if method == "Global unstructured":
        return "Global"
    if row.get("criterion") == "l1_sum":
        return "Cross l1_sum"
    if row.get("criterion") == "top2_mean_abs":
        return "Cross top2 + sensitivity"
    return method


def prepare():
    df = pd.read_csv(INPUT)
    df["variant"] = df.apply(variant_name, axis=1)
    df["final_acc"] = df["after_extra_or_ft_acc"].fillna(df["direct_acc"])
    return df


def write_summaries(df):
    dense = (
        df[df["variant"].isin(["Dense", "Dense extra"])]
        .drop_duplicates(["variant", "seed"])
        .groupby("variant", as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            accuracy_mean=("final_acc", "mean"),
            accuracy_std=("final_acc", "std"),
        )
    )
    dense.to_csv(ROOT / "dense_summary.csv", index=False)

    pruned = df[df["variant"].isin(["Global", "Cross l1_sum", "Cross top2 + sensitivity"])]
    summary = (
        pruned.groupby(["variant", "tm", "target_prune_ratio"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            direct_acc_mean=("direct_acc", "mean"),
            direct_acc_std=("direct_acc", "std"),
            final_acc_mean=("final_acc", "mean"),
            final_acc_std=("final_acc", "std"),
            delta_vs_dense_extra_mean=("accuracy_delta_vs_dense_extra", "mean"),
            delta_vs_dense_extra_std=("accuracy_delta_vs_dense_extra", "std"),
            actual_pruning_ratio_mean=("pruning_ratio", "mean"),
            mac_reduction_mean=("mac_reduction", "mean"),
            cycle_reduction_mean=("cycle_reduction", "mean"),
            estimated_speedup_mean=("estimated_speedup", "mean"),
        )
        .sort_values(["tm", "target_prune_ratio", "variant"])
    )
    summary.to_csv(ROOT / "method_summary_mean_std.csv", index=False)

    paired = pruned.pivot_table(
        index=["seed", "tm", "target_prune_ratio"],
        columns="variant",
        values=["final_acc", "pruning_ratio", "estimated_speedup"],
        aggfunc="first",
    )
    paired.columns = [f"{metric}_{variant}" for metric, variant in paired.columns]
    paired = paired.reset_index()
    paired["top2_minus_l1_acc"] = (
        paired["final_acc_Cross top2 + sensitivity"] - paired["final_acc_Cross l1_sum"]
    )
    paired.to_csv(ROOT / "paired_top2_vs_l1.csv", index=False)
    return dense, summary


def plot_accuracy(dense, summary):
    styles = {
        "Global": ("#555555", "o"),
        "Cross l1_sum": ("#c84c43", "s"),
        "Cross top2 + sensitivity": ("#168c82", "^"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, constrained_layout=True)
    dense_mean = dense.loc[dense["variant"] == "Dense", "accuracy_mean"].iloc[0] * 100
    dense_extra = dense.loc[dense["variant"] == "Dense extra", "accuracy_mean"].iloc[0] * 100
    for ax, tm in zip(axes, [8, 16]):
        part = summary[summary["tm"] == tm]
        for variant, (color, marker) in styles.items():
            rows = part[part["variant"] == variant].sort_values("target_prune_ratio")
            ax.errorbar(
                rows["target_prune_ratio"] * 100,
                rows["final_acc_mean"] * 100,
                yerr=rows["final_acc_std"] * 100,
                color=color,
                marker=marker,
                linewidth=1.8,
                capsize=3,
                label=variant,
            )
        ax.axhline(dense_mean, color="#2d63ad", linestyle="--", linewidth=1.3, label="Dense")
        ax.axhline(dense_extra, color="#8b63a8", linestyle=":", linewidth=1.5, label="Dense extra")
        ax.set_title(f"Tm = {tm}")
        ax.set_xlabel("Target pruning ratio (%)")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Test accuracy after fine-tuning (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=5, frameon=False)
    fig.savefig(CHARTS / "accuracy_vs_pruning_ratio.png", dpi=180)
    plt.close(fig)


def plot_delta(summary):
    variants = ["Global", "Cross l1_sum", "Cross top2 + sensitivity"]
    colors = ["#555555", "#c84c43", "#168c82"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, constrained_layout=True)
    width = 0.23
    for ax, tm in zip(axes, [8, 16]):
        part = summary[summary["tm"] == tm]
        ratios = sorted(part["target_prune_ratio"].unique())
        x = list(range(len(ratios)))
        for index, (variant, color) in enumerate(zip(variants, colors)):
            rows = part[part["variant"] == variant].set_index("target_prune_ratio").loc[ratios]
            offset = (index - 1) * width
            ax.bar(
                [value + offset for value in x],
                rows["delta_vs_dense_extra_mean"] * 100,
                width,
                yerr=rows["delta_vs_dense_extra_std"] * 100,
                color=color,
                capsize=3,
                label=variant,
            )
        ax.axhline(0, color="#222222", linewidth=0.9)
        ax.set_xticks(x, [f"{ratio:.0%}" for ratio in ratios])
        ax.set_title(f"Tm = {tm}")
        ax.set_xlabel("Target pruning ratio")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Accuracy change vs Dense extra (pp)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False)
    fig.savefig(CHARTS / "accuracy_delta_vs_dense_extra.png", dpi=180)
    plt.close(fig)


def plot_tradeoff(summary):
    colors = {
        "Global": "#555555",
        "Cross l1_sum": "#c84c43",
        "Cross top2 + sensitivity": "#168c82",
    }
    markers = {8: "o", 16: "s"}
    fig, ax = plt.subplots(figsize=(8.6, 5.6), constrained_layout=True)
    for _, row in summary.iterrows():
        ax.scatter(
            row["estimated_speedup_mean"],
            row["final_acc_mean"] * 100,
            color=colors[row["variant"]],
            marker=markers[int(row["tm"])],
            s=70,
        )
        ax.annotate(
            f"{row['target_prune_ratio']:.0%}",
            (row["estimated_speedup_mean"], row["final_acc_mean"] * 100),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    for variant, color in colors.items():
        ax.scatter([], [], color=color, label=variant)
    for tm, marker in markers.items():
        ax.scatter([], [], color="#222222", marker=marker, label=f"Tm={tm}")
    ax.set_xlabel("Estimated speedup")
    ax.set_ylabel("Test accuracy after fine-tuning (%)")
    ax.grid(alpha=0.25)
    ax.legend(ncols=2, frameon=False)
    fig.savefig(CHARTS / "accuracy_speedup_tradeoff.png", dpi=180)
    plt.close(fig)


def main():
    CHARTS.mkdir(parents=True, exist_ok=True)
    df = prepare()
    dense, summary = write_summaries(df)
    plot_accuracy(dense, summary)
    plot_delta(summary)
    plot_tradeoff(summary)
    print(f"Wrote summaries and charts to {ROOT}")


if __name__ == "__main__":
    main()
