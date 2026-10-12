#!/usr/bin/env python3
"""Compare similarity-based filter grouping against contiguous Tm=8 groups."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("runs/filter_similarity_grouping")
BASELINE_CSV = Path("runs/fixed_checkpoint_all_ratio_summary/all_results_combined.csv")
MODELS = ["resnet34", "resnet50", "googlenet", "densenet121", "mobilenet_v2"]
RATIOS = [0.80, 0.85, 0.90]


def load_similarity_results() -> pd.DataFrame:
    frames = []
    for path in sorted(ROOT.glob("*_similarity_jaccard_results.csv")):
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        raise SystemExit(f"No completed similarity results found in {ROOT}")
    data = pd.concat(frames, ignore_index=True)
    return data[
        (data["criterion"] == "l1_sum")
        & data["ratio_source"].str.contains("group_similarity_jaccard", na=False)
    ].copy()


def load_baseline() -> pd.DataFrame:
    data = pd.read_csv(BASELINE_CSV)
    return data[
        (data["method"] == "Cross-filter l1_sum (global_cluster)")
        & (data["tm"] == 8)
        & data["target_prune_ratio"].isin(RATIOS)
    ].copy()


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    similarity = load_similarity_results()
    baseline = load_baseline()
    keys = ["model", "seed", "tm", "target_prune_ratio", "checkpoint_id"]
    columns = keys + [
        "direct_acc",
        "after_extra_or_ft_acc",
        "pruning_ratio",
        "mac_reduction",
        "estimated_speedup",
        "dense_extra_test_acc",
    ]
    paired = baseline[columns].merge(
        similarity[columns[:-1]],
        on=keys,
        how="inner",
        suffixes=("_contiguous", "_similarity"),
        validate="one_to_one",
    )
    paired["direct_gain_pp"] = (
        paired["direct_acc_similarity"] - paired["direct_acc_contiguous"]
    ) * 100
    paired["ft_gain_pp"] = (
        paired["after_extra_or_ft_acc_similarity"]
        - paired["after_extra_or_ft_acc_contiguous"]
    ) * 100
    paired["contiguous_loss_pp"] = (
        paired["dense_extra_test_acc"] - paired["after_extra_or_ft_acc_contiguous"]
    ) * 100
    paired["similarity_loss_pp"] = (
        paired["dense_extra_test_acc"] - paired["after_extra_or_ft_acc_similarity"]
    ) * 100
    paired.to_csv(ROOT / "paired_results.csv", index=False)

    summary = (
        paired.groupby(["model", "target_prune_ratio"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            contiguous_ft_acc_mean=("after_extra_or_ft_acc_contiguous", "mean"),
            similarity_ft_acc_mean=("after_extra_or_ft_acc_similarity", "mean"),
            contiguous_loss_pp_mean=("contiguous_loss_pp", "mean"),
            similarity_loss_pp_mean=("similarity_loss_pp", "mean"),
            direct_gain_pp_mean=("direct_gain_pp", "mean"),
            ft_gain_pp_mean=("ft_gain_pp", "mean"),
            ft_gain_pp_std=("ft_gain_pp", "std"),
            ft_gain_pp_min=("ft_gain_pp", "min"),
            ft_gain_pp_max=("ft_gain_pp", "max"),
            similarity_actual_sparsity=("pruning_ratio_similarity", "mean"),
            similarity_mac_reduction=("mac_reduction_similarity", "mean"),
            similarity_estimated_speedup=("estimated_speedup_similarity", "mean"),
        )
        .sort_values(["model", "target_prune_ratio"])
    )
    summary.to_csv(ROOT / "summary.csv", index=False)

    pivot = summary.pivot(
        index="model", columns="target_prune_ratio", values="ft_gain_pp_mean"
    ).reindex(MODELS)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    image = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), [f"{ratio:.0%}" for ratio in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_xlabel("Target pruning ratio")
    ax.set_title("Fine-tuned accuracy gain: similarity vs contiguous grouping (pp)")
    for row in range(len(pivot.index)):
        for column in range(len(pivot.columns)):
            value = pivot.iloc[row, column]
            ax.text(column, row, f"{value:+.3f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Accuracy gain (percentage points)")
    fig.tight_layout()
    fig.savefig(ROOT / "ft_accuracy_gain_heatmap.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)
    for ax, model in zip(axes.flat, MODELS):
        model_data = summary[summary["model"] == model]
        x = model_data["target_prune_ratio"] * 100
        ax.plot(x, model_data["contiguous_loss_pp_mean"], marker="o", label="Contiguous")
        ax.plot(x, model_data["similarity_loss_pp_mean"], marker="s", label="Similarity")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(model)
        ax.set_ylabel("Accuracy loss (pp)")
        ax.grid(alpha=0.25)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.94, 0.13))
    fig.supxlabel("Target pruning ratio (%)")
    fig.suptitle("Cross-filter l1_sum, Tm=8, fixed checkpoints, 3 seeds")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(ROOT / "accuracy_loss_comparison.png", dpi=180)
    plt.close(fig)

    expected = len(MODELS) * len(RATIOS) * 3
    complete = len(paired) == expected
    lines = [
        "# Filter 相似度分組實驗",
        "",
        "固定 CIFAR-10 Dense checkpoints，`Tm=8`、Cross-filter `l1_sum`、fine-tune 10 epochs。",
        "相似分組使用 global unstructured keep masks 的 K-position Jaccard similarity。",
        "模型 channel 順序沒有改動；filter group 僅決定哪些 filters 共用 cross-filter mask。",
        "",
        f"完成進度：`{len(paired)}/{expected}` 組配對結果（{'完成' if complete else '尚未完成'}）。",
        "",
        "## 平均結果",
        "",
        "| 模型 | Ratio | 連續分組損失 (pp) | 相似分組損失 (pp) | FT 提升 (pp) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(
            f"| {row.model} | {row.target_prune_ratio:.0%} | "
            f"{row.contiguous_loss_pp_mean:.3f} | {row.similarity_loss_pp_mean:.3f} | "
            f"{row.ft_gain_pp_mean:+.3f} |"
        )
    lines.extend(
        [
            "",
            "正值表示相似分組的 fine-tuned accuracy 較高。比較使用相同 model、seed、",
            "target ratio、Tm 與 Dense checkpoint 配對。",
        ]
    )
    (ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"paired {len(paired)}/{expected}; wrote {ROOT / 'summary.csv'}")


if __name__ == "__main__":
    main()
