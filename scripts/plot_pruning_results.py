#!/usr/bin/env python3
"""Plot pruning experiment CSV files into summary charts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_ORDER = [
    "resnet34",
    "resnet50",
    "googlenet",
    "densenet121",
    "mobilenet_v2",
    "efficientnet_v2_m",
]

COLORS = {
    "Dense extra training": "#4C566A",
    "Global unstructured": "#5E81AC",
    "l1_sum": "#D08770",
    "max_abs": "#A3BE8C",
}


def _ordered_models(values: pd.Series) -> list[str]:
    seen = set(values.dropna().astype(str))
    ordered = [m for m in MODEL_ORDER if m in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


def _method_label(row: pd.Series) -> str:
    if str(row["method"]).startswith("Cross-filter"):
        return str(row["criterion"])
    return str(row["method"])


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["plot_method"] = df.apply(_method_label, axis=1)
    df["acc_pct"] = df["after_extra_or_ft_acc"].fillna(df["direct_acc"]) * 100.0
    df["ratio_bucket"] = df["pruning_ratio"].round(1)
    df["ratio_label"] = df["ratio_bucket"].map(lambda x: f"{x:.1f}")
    return df


def plot_ratio_delta(df: pd.DataFrame, out_dir: Path) -> None:
    cross = df[df["method"].astype(str).str.startswith("Cross-filter")].copy()
    if cross.empty or cross["ratio_bucket"].nunique() <= 1:
        return

    pivot = cross.pivot_table(
        index="model",
        columns=["ratio_bucket", "criterion"],
        values="acc_pct",
        aggfunc="mean",
    )
    ratios = sorted(cross["ratio_bucket"].dropna().unique())
    models = _ordered_models(cross["model"])
    delta = pd.DataFrame(index=models, columns=ratios, dtype=float)
    for ratio in ratios:
        if (ratio, "max_abs") in pivot.columns and (ratio, "l1_sum") in pivot.columns:
            delta[ratio] = pivot[(ratio, "max_abs")] - pivot[(ratio, "l1_sum")]
    delta = delta.dropna(how="all")
    if delta.empty:
        return

    fig, ax = plt.subplots(figsize=(8.5, max(3.6, 0.55 * len(delta) + 1.6)))
    values = delta.to_numpy(dtype=float)
    max_abs = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
    im = ax.imshow(values, cmap="RdYlGn", vmin=-max_abs, vmax=max_abs, aspect="auto")
    ax.set_xticks(range(len(delta.columns)), [f"{r:.1f}" for r in delta.columns])
    ax.set_yticks(range(len(delta.index)), delta.index)
    ax.set_xlabel("Pruning ratio")
    ax.set_title("Accuracy delta: max_abs - l1_sum (percentage points)")
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            val = values[y, x]
            if np.isfinite(val):
                ax.text(x, y, f"{val:+.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="pp")
    fig.tight_layout()
    fig.savefig(out_dir / "ratio_delta_heatmap.png", dpi=180)
    plt.close(fig)


def plot_accuracy_vs_ratio(df: pd.DataFrame, out_dir: Path) -> None:
    data = df[
        df["plot_method"].isin(["Global unstructured", "l1_sum", "max_abs"])
        & df["pruning_ratio"].gt(0)
    ].copy()
    if data.empty or data["ratio_bucket"].nunique() <= 1:
        return

    models = _ordered_models(data["model"])
    fig, axes = plt.subplots(
        len(models),
        1,
        figsize=(8.5, max(3.2, 2.05 * len(models))),
        sharex=True,
        constrained_layout=True,
    )
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = data[data["model"].eq(model)]
        for method in ["Global unstructured", "l1_sum", "max_abs"]:
            line = (
                sub[sub["plot_method"].eq(method)]
                .groupby("ratio_bucket", as_index=False)["acc_pct"]
                .mean()
                .sort_values("ratio_bucket")
            )
            if line.empty:
                continue
            ax.plot(
                line["ratio_bucket"],
                line["acc_pct"],
                marker="o",
                linewidth=1.8,
                label=method,
                color=COLORS.get(method),
            )
        ax.set_title(model)
        ax.set_ylabel("FT acc (%)")
        ax.grid(True, linewidth=0.5, alpha=0.35)
    axes[-1].set_xlabel("Pruning ratio")
    axes[0].legend(loc="best", ncol=3)
    fig.savefig(out_dir / "accuracy_vs_ratio.png", dpi=180)
    plt.close(fig)


def plot_speedup_vs_ratio(df: pd.DataFrame, out_dir: Path) -> None:
    data = df[
        df["method"].astype(str).str.startswith("Cross-filter")
        & df["criterion"].eq("max_abs")
        & df["pruning_ratio"].gt(0)
    ].copy()
    if data.empty or data["ratio_bucket"].nunique() <= 1:
        return

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for model in _ordered_models(data["model"]):
        line = (
            data[data["model"].eq(model)]
            .groupby("ratio_bucket", as_index=False)["estimated_speedup"]
            .mean()
            .sort_values("ratio_bucket")
        )
        if not line.empty:
            ax.plot(line["ratio_bucket"], line["estimated_speedup"], marker="o", label=model)
    ax.set_title("Estimated speedup vs pruning ratio (max_abs)")
    ax.set_xlabel("Pruning ratio")
    ax.set_ylabel("Estimated speedup (x)")
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "speedup_vs_ratio_max_abs.png", dpi=180)
    plt.close(fig)


def plot_model_accuracy_bars(df: pd.DataFrame, out_dir: Path) -> None:
    data = df[df["plot_method"].isin(["Dense extra training", "Global unstructured", "l1_sum", "max_abs"])].copy()
    if data.empty:
        return

    if data["ratio_bucket"].nunique() > 1:
        # For sweeps, make the compact bar chart at the middle available ratio.
        ratio = sorted(data.loc[data["pruning_ratio"].gt(0), "ratio_bucket"].dropna().unique())
        if not ratio:
            return
        target_ratio = ratio[len(ratio) // 2]
        data = pd.concat(
            [
                data[data["plot_method"].eq("Dense extra training")],
                data[data["ratio_bucket"].eq(target_ratio)],
            ],
            ignore_index=True,
        )
        suffix = f"_r{target_ratio:.1f}".replace(".", "p")
    else:
        suffix = ""

    models = _ordered_models(data["model"])
    methods = ["Dense extra training", "Global unstructured", "l1_sum", "max_abs"]
    table = data.pivot_table(index="model", columns="plot_method", values="acc_pct", aggfunc="mean").reindex(models)

    x = np.arange(len(models))
    width = 0.18
    fig, ax = plt.subplots(figsize=(max(8.5, 1.25 * len(models) + 3.0), 4.8))
    for idx, method in enumerate(methods):
        if method not in table.columns:
            continue
        ax.bar(x + (idx - 1.5) * width, table[method], width, label=method, color=COLORS.get(method))
    ax.set_title("Fine-tuned accuracy by model")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x, models, rotation=20, ha="right")
    ax.grid(axis="y", linewidth=0.5, alpha=0.35)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / f"model_accuracy_bars{suffix}.png", dpi=180)
    plt.close(fig)


def write_summary(df: pd.DataFrame, out_dir: Path) -> None:
    cross = df[df["method"].astype(str).str.startswith("Cross-filter")].copy()
    lines = ["# 圖表說明", ""]
    if not cross.empty:
        paired = cross.pivot_table(
            index=["model", "ratio_bucket"],
            columns="criterion",
            values="acc_pct",
            aggfunc="mean",
        )
        if {"l1_sum", "max_abs"}.issubset(paired.columns):
            paired["max_abs_minus_l1_sum_pp"] = paired["max_abs"] - paired["l1_sum"]
            avg = paired["max_abs_minus_l1_sum_pp"].mean()
            lines.append(f"- Cross-filter `max_abs - l1_sum` 平均差：`{avg:+.2f}` percentage points。")
            lines.append(f"- 配對數量：`{paired['max_abs_minus_l1_sum_pp'].notna().sum()}` 組 model/ratio。")
    descriptions = {
        "ratio_delta_heatmap.png": "不同 model/ratio 下，`max_abs` 相對 `l1_sum` 的 accuracy 差。",
        "accuracy_vs_ratio.png": "剪枝比例提高時，各方法 fine-tuned accuracy 的變化。",
        "speedup_vs_ratio_max_abs.png": "`max_abs` 的估計加速比趨勢。",
    }
    for filename, description in descriptions.items():
        if (out_dir / filename).exists():
            lines.append(f"- `{filename}`：{description}")
            lines.append(f"  ![]({filename})")
    for png in sorted(out_dir.glob("model_accuracy_bars*.png")):
        lines.append(f"- `{png.name}`：各模型在同一設定下的 accuracy 條狀比較。")
        lines.append(f"  ![]({png.name})")
    lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Input experiment CSV.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for generated charts.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for old_png in args.out_dir.glob("*.png"):
        old_png.unlink()
    df = _prepare(pd.read_csv(args.csv))
    plot_ratio_delta(df, args.out_dir)
    plot_accuracy_vs_ratio(df, args.out_dir)
    plot_speedup_vs_ratio(df, args.out_dir)
    plot_model_accuracy_bars(df, args.out_dir)
    write_summary(df, args.out_dir)


if __name__ == "__main__":
    main()
