#!/usr/bin/env python3
"""Accuracy against V3 predicted latency, CIFAR-100, five models (from CURVE_CIFAR100.csv)."""
import csv, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

R = Path("runs/other_datasets")
ARMS = [("floor", "Gradual + floor (ours)", "#2a78d6", "o"), ("rewind", "Paper mask + LR rewind", "#eb6834", "s"),
        ("fill15", "Gradual + fill15 (ours)", "#1baf7a", "^"), ("gradual_unstructured", "Gradual unstructured", "#eda100", "D")]
NAMES = {"resnet34": "ResNet34", "resnet56": "ResNet-56", "vgg16_bn": "VGG16-BN", "googlenet": "GoogLeNet", "densenet121": "DenseNet121"}
rows = [r for r in csv.DictReader((R/"CURVE_CIFAR100.csv").open()) if r["ms"]]
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e"})
fig, axes = plt.subplots(1, 5, figsize=(17, 3.8))
for ax, (model, name) in zip(axes, NAMES.items()):
    mine = [r for r in rows if r["model"] == model]
    dense = float(mine[0]["dense"]); dense_ms = float(mine[0]["dense_ms"])
    for arm, label, colour, marker in ARMS:
        pts = sorted((float(r["ms"]), float(r["test"]), int(r["ratio"])) for r in mine if r["arm"] == arm and float(r["test"]) > 20)
        if not pts:
            continue
        xs, ys, rs = zip(*pts)
        ax.plot(xs, ys, color=colour, lw=2, marker=marker, ms=6, mec="#fcfcfb", mew=1.2, label=label, zorder=3)
        if arm in ("floor", "gradual_unstructured"):
            for x, y, r in pts:
                ax.annotate(f"{r}", (x, y), textcoords="offset points", xytext=(4, -9 if arm == "floor" else 4), fontsize=7, color="#52514e")
    ax.axhline(dense, color="#52514e", lw=1, ls="--", zorder=1)
    ax.text(0.02, dense, f"Dense {dense:.1f}% @ {dense_ms:.0f} ms", transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=7, color="#52514e")
    ax.set_xscale("log"); ax.set_title(name, fontsize=10, color="#0b0b0b")
    lo, hi = ax.get_xlim()
    ticks = [t for t in (2, 3, 5, 10, 20, 50, 100, 200) if lo <= t <= hi]
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("V3 latency, conv layers (ms, log)")
    ax.grid(True, which="major", color="#e6e5e0", lw=.6); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_ylabel("CIFAR-100 test accuracy (%)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(.5, 1.02))
fig.text(.5, -.04, "Mean of seeds 0/1/2, FP32. Point labels: pruning ratio (%). Latency is the V3 formula prediction at 50 MHz. "
         "Gradual-unstructured latency at 98/99% from the seed-0 rerun; VGG16-BN rewind at 98/99% (1% accuracy) is omitted.", ha="center", fontsize=7.5, color="#52514e")
fig.tight_layout(rect=(0, 0, 1, .93))
out = R/"CURVE_CIFAR100.png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="#fcfcfb")
print(out)
