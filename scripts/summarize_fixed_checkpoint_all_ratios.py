#!/usr/bin/env python3
"""Combine fixed-checkpoint pruning results across models and target ratios."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT = Path("runs/fixed_checkpoint_all_ratio_summary")
COMMON_RATIOS = [0.70, 0.80, 0.85, 0.90, 0.95]
MODELS = ["resnet34", "resnet50", "googlenet", "densenet121", "mobilenet_v2"]
MODEL_LABELS = {
    "resnet34": "ResNet34",
    "resnet50": "ResNet50",
    "googlenet": "GoogLeNet",
    "densenet121": "DenseNet121",
    "mobilenet_v2": "MobileNetV2",
}
METHOD_LABELS = {
    "Global unstructured": "Global unstructured",
    "Cross-filter l1_sum (global_cluster)": "Cross l1_sum",
    "Cross-filter top2_mean_abs (global_cluster+cap0.75+sens4@0.2+skip1)": (
        "Cross top2 + sensitivity"
    ),
}
COLORS = {
    "resnet34": "#3973ac",
    "resnet50": "#3a8f5b",
    "googlenet": "#d97932",
    "densenet121": "#8b5fbf",
    "mobilenet_v2": "#c44545",
}


def result_paths():
    paths = []
    paths.extend(Path("runs/fixed_checkpoint_resnet34/pruning_comparison").glob("*_results.csv"))
    paths.extend(Path("runs/fixed_checkpoint_multi_model/pruning_r0p7").glob("*/*_results.csv"))
    paths.extend(Path("runs/fixed_checkpoint_r0p8").glob("*/*_results.csv"))
    paths.extend(Path("runs/fixed_checkpoint_resnet34_r0p85_r0p90").glob("*_results.csv"))
    paths.extend(Path("runs/fixed_checkpoint_high_ratio_sweep").glob("*_results.csv"))
    return sorted(set(paths))


def load_results():
    frames = []
    for path in result_paths():
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        raise SystemExit("No fixed-checkpoint result files found")
    data = pd.concat(frames, ignore_index=True)
    key = ["model", "target_prune_ratio", "seed", "tm", "method"]
    duplicates = data.duplicated(key, keep=False)
    if duplicates.any():
        rows = data.loc[duplicates, key + ["source_file"]].sort_values(key)
        raise SystemExit(f"Duplicate result rows found:\n{rows.to_string(index=False)}")
    return data


def validate_common_runs(data):
    files = data[["model", "target_prune_ratio", "seed", "tm", "source_file"]].drop_duplicates()
    existing = {
        (row.model, round(float(row.target_prune_ratio), 2), int(row.seed), int(row.tm))
        for row in files.itertuples()
    }
    expected = {
        (model, ratio, seed, tm)
        for model in MODELS
        for ratio in COMMON_RATIOS
        for seed in [0, 1, 2]
        for tm in [8, 16]
    }
    missing = sorted(expected - existing)
    if missing:
        preview = "\n".join(map(str, missing[:20]))
        raise SystemExit(f"Missing {len(missing)} common-ratio runs:\n{preview}")


def summarize(data):
    pruned = data[data["method"].isin(METHOD_LABELS)].copy()
    pruned["variant"] = pruned["method"].map(METHOD_LABELS)
    pruned["accuracy_loss_pp"] = -100.0 * pruned["accuracy_delta_vs_dense_extra"]
    summary = (
        pruned.groupby(["model", "target_prune_ratio", "variant", "tm"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            dense_extra_accuracy_mean=("dense_extra_test_acc", "mean"),
            direct_accuracy_mean=("direct_acc", "mean"),
            direct_accuracy_std=("direct_acc", "std"),
            final_accuracy_mean=("after_extra_or_ft_acc", "mean"),
            final_accuracy_std=("after_extra_or_ft_acc", "std"),
            accuracy_loss_pp_mean=("accuracy_loss_pp", "mean"),
            accuracy_loss_pp_std=("accuracy_loss_pp", "std"),
            accuracy_loss_pp_max=("accuracy_loss_pp", "max"),
            actual_pruning_ratio_mean=("pruning_ratio", "mean"),
            mac_reduction_mean=("mac_reduction", "mean"),
            cycle_reduction_mean=("cycle_reduction", "mean"),
            estimated_speedup_mean=("estimated_speedup", "mean"),
        )
        .sort_values(["model", "target_prune_ratio", "variant", "tm"])
    )
    summary["under_1pp_mean"] = summary["accuracy_loss_pp_mean"] < 1.0
    summary["under_1pp_all_seeds"] = summary["accuracy_loss_pp_max"] < 1.0
    return summary


def l1_tables(summary):
    l1 = summary[summary["variant"] == "Cross l1_sum"].copy()
    common = l1[l1["target_prune_ratio"].round(2).isin(COMMON_RATIOS)].copy()
    pivot = common.pivot(
        index="model", columns=["target_prune_ratio", "tm"], values="accuracy_loss_pp_mean"
    ).reindex(MODELS)
    pivot.columns = [f"r{ratio:.2f}_tm{tm}" for ratio, tm in pivot.columns]
    pivot.reset_index().to_csv(OUTPUT / "l1_accuracy_loss_pivot.csv", index=False)

    rows = []
    for model in MODELS:
        for tm in [8, 16]:
            part = common[(common["model"] == model) & (common["tm"] == tm)].sort_values(
                "target_prune_ratio"
            )
            passing = part[part["under_1pp_mean"]]
            failing = part[~part["under_1pp_mean"]]
            rows.append(
                {
                    "model": model,
                    "tm": tm,
                    "highest_tested_ratio_under_1pp_mean": (
                        passing["target_prune_ratio"].max() if not passing.empty else np.nan
                    ),
                    "first_tested_ratio_at_or_above_1pp_mean": (
                        failing["target_prune_ratio"].min() if not failing.empty else np.nan
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(OUTPUT / "l1_under_1pp_thresholds.csv", index=False)


def plot_l1_curves(summary):
    l1 = summary[summary["variant"] == "Cross l1_sum"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, tm in zip(axes, [8, 16]):
        for model in MODELS:
            part = l1[(l1["model"] == model) & (l1["tm"] == tm)].sort_values(
                "target_prune_ratio"
            )
            ax.errorbar(
                100.0 * part["target_prune_ratio"],
                part["accuracy_loss_pp_mean"],
                yerr=part["accuracy_loss_pp_std"],
                marker="o",
                linewidth=1.7,
                capsize=3,
                color=COLORS[model],
                label=MODEL_LABELS[model],
            )
        ax.axhline(1.0, color="#b3261e", linestyle="--", linewidth=1.4, label="1 pp limit")
        ax.set_title(f"Tm = {tm}")
        ax.set_xlabel("Target pruning ratio (%)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Accuracy loss vs Dense extra (percentage points)")
    handles, labels = axes[1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=6, frameon=False)
    fig.suptitle("Cross-filter l1_sum accuracy loss", y=1.03)
    fig.tight_layout()
    fig.savefig(OUTPUT / "l1_accuracy_loss_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_l1_heatmap(summary):
    l1 = summary[
        (summary["variant"] == "Cross l1_sum")
        & summary["target_prune_ratio"].round(2).isin(COMMON_RATIOS)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    image = None
    for ax, tm in zip(axes, [8, 16]):
        table = (
            l1[l1["tm"] == tm]
            .pivot(index="model", columns="target_prune_ratio", values="accuracy_loss_pp_mean")
            .reindex(index=MODELS, columns=COMMON_RATIOS)
        )
        values = table.to_numpy()
        image = ax.imshow(values, cmap="RdYlGn_r", vmin=0.0, vmax=max(3.0, np.nanmax(values)))
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values[row, col]
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=9)
        ax.set_xticks(range(len(COMMON_RATIOS)), [f"{100*r:.0f}%" for r in COMMON_RATIOS])
        ax.set_yticks(range(len(MODELS)), [MODEL_LABELS[m] for m in MODELS])
        ax.set_xlabel("Target pruning ratio")
        ax.set_title(f"Tm = {tm}")
    fig.colorbar(image, ax=axes, label="Accuracy loss (pp)", shrink=0.85)
    fig.suptitle("Cross-filter l1_sum loss heatmap")
    fig.savefig(OUTPUT / "l1_accuracy_loss_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoffs(summary):
    common = summary[summary["target_prune_ratio"].round(2).isin(COMMON_RATIOS)]
    variants = ["Cross l1_sum", "Cross top2 + sensitivity"]
    variant_colors = {"Cross l1_sum": "#3973ac", "Cross top2 + sensitivity": "#d97932"}
    markers = {8: "o", 16: "s"}
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), sharex=True)
    for ax, model in zip(axes.flat, MODELS):
        for variant in variants:
            for tm in [8, 16]:
                part = common[
                    (common["model"] == model)
                    & (common["variant"] == variant)
                    & (common["tm"] == tm)
                ].sort_values("actual_pruning_ratio_mean")
                ax.plot(
                    100.0 * part["actual_pruning_ratio_mean"],
                    part["accuracy_loss_pp_mean"],
                    color=variant_colors[variant],
                    marker=markers[tm],
                    linewidth=1.5,
                    label=f"{variant}, Tm={tm}",
                )
        ax.axhline(1.0, color="#b3261e", linestyle="--", linewidth=1.1)
        ax.set_title(MODEL_LABELS[model])
        ax.grid(alpha=0.25)
    axes[1, 2].axis("off")
    for ax in axes[1, :2]:
        ax.set_xlabel("Actual weight sparsity (%)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy loss (pp)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.97, 0.08), fontsize=8)
    fig.suptitle("Accuracy vs actual sparsity")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(OUTPUT / "accuracy_sparsity_tradeoffs.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = load_results()
    validate_common_runs(data)
    summary = summarize(data)
    data.to_csv(OUTPUT / "all_results_combined.csv", index=False)
    summary.to_csv(OUTPUT / "all_methods_summary.csv", index=False)
    summary[summary["variant"].str.startswith("Cross")].to_csv(
        OUTPUT / "cross_filter_summary.csv", index=False
    )
    l1_tables(summary)
    plot_l1_curves(summary)
    plot_l1_heatmap(summary)
    plot_tradeoffs(summary)
    print(
        summary[
            (summary["variant"] == "Cross l1_sum")
            & summary["target_prune_ratio"].round(2).isin(COMMON_RATIOS)
        ][
            [
                "model",
                "target_prune_ratio",
                "tm",
                "accuracy_loss_pp_mean",
                "accuracy_loss_pp_std",
                "under_1pp_mean",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
