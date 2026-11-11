#!/usr/bin/env python3
"""Summarise runs/other_datasets: test accuracy, drop against Dense and the unpruned 60-epoch control, cycles."""
import csv, json, subprocess, sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path("runs/other_datasets")
CYCLES = Path(__file__).resolve().parents[2]/"mask_aware_conv_accelerator/scripts/estimate_network_cycles.py"


def load():
    runs = defaultdict(dict)
    for path in ROOT.glob("*/result.json"):
        r = json.loads(path.read_text())
        # This table is the frozen seed 0/1/2 study; plans H, I and J are judged in their own scripts.
        special = r.get("kd") or r.get("allocator", "cluster") != "cluster" or r.get("q_floor_n", 1) > 1 or r.get("free_fill") or r.get("tag") or r.get("mask_from") or r.get("quota_from") or (r.get("ft_lr") or .001) != .001 or r.get("weight_decay", 5e-4) != 5e-4
        if special or r["seed"] > 2 or (r["arm"] != "dense" and r["epochs"] != 60):
            continue
        ratio = round(r["ratio"]*100)
        arm = {"dense": "dense", "rewind": "control" if ratio == 0 else "rewind"}.get(r["arm"], r["arm"])
        arm += "_stem" if r.get("stem_keep") else ""
        runs[r["dataset"], r["model"], arm, ratio][r["seed"]] = (r, path.parent)
    return runs


def stem_colours(directory):
    mask = torch.load(directory/"masks.pt", weights_only=False)["conv1"]
    view = mask.reshape(mask.shape[0], 3, -1)
    return "".join("RGB"[c] if view[:, c].any() else "-" for c in range(3))


def cycles(dataset, model, label, seeds):
    geometry = ROOT/f"geometry_{dataset}_{model}.json"
    out = ROOT/"cycles_tmp.csv"
    masks = [f"{label}_{s}={d}/masks.pt" for s, (_, d) in seeds.items()]
    subprocess.run([sys.executable, str(CYCLES), "--geometry", str(geometry), "--masks", *masks, "--out", str(out)], check=True, stdout=subprocess.DEVNULL)
    rows = [float(r["ms_50mhz"]) for r in csv.DictReader(out.open()) if r["label"] != "dense"]
    dense = next(float(r["ms_50mhz"]) for r in csv.DictReader(out.open()) if r["label"] == "dense")
    out.unlink()
    return sum(rows)/len(rows), dense


def mean(values):
    values = list(values)
    return sum(values)/len(values)


def main():
    runs = load()
    table = []
    for dataset, model in sorted({k[:2] for k in runs}):
        get = lambda arm, ratio: runs.get((dataset, model, arm, ratio), {})
        test = lambda seeds: mean(r["final_test"]*100 for r, _ in seeds.values())
        dense, control = get("dense", 0), get("control", 0)
        print(f"\n{dataset} {model}: Dense {test(dense):.2f}" if dense else f"\n{dataset} {model}", (f"control(+60ep) {test(control):.2f}" if control else "control pending"))
        for ratio in (90, 95, 98, 99):
            base = get("rewind", ratio)
            for arm in ("rewind", "gradual", "gradual_stem", "unstructured", "random"):
                seeds = get(arm, ratio)
                if not seeds:
                    continue
                accuracy, row = test(seeds), dict(dataset=dataset, model=model, ratio=ratio, arm=arm, n=len(seeds))
                row["test"] = round(accuracy, 3)
                row["seeds"] = "/".join(f"{seeds[s][0]['final_test']*100:.2f}" for s in sorted(seeds))
                row["drop_dense"] = round(test(dense)-accuracy, 3) if dense else ""
                row["drop_control"] = round(test(control)-accuracy, 3) if control else ""
                if arm != "unstructured":
                    ms, dense_ms = cycles(dataset, model, f"{arm}_{ratio}", seeds)
                    row.update(ms=round(ms, 2), dense_ms=round(dense_ms, 2), colours=",".join(stem_colours(d) for _, d in seeds.values()),
                               q_zero=round(mean(r["q_zero"] for r, _ in seeds.values()), 1), sparsity=round(mean(r["sparsity"] for r, _ in seeds.values()), 4))
                if arm != "rewind" and base:
                    common = sorted(set(seeds) & set(base))
                    gains = [(seeds[s][0]["final_test"]-base[s][0]["final_test"])*100 for s in common]
                    row["gain_vs_rewind"] = "/".join(f"{g:+.2f}" for g in gains)
                    row["gain_mean"] = round(mean(gains), 3)
                table.append(row)
                print(f" {ratio}% {arm:13s} n={len(seeds)} test {accuracy:6.2f} [{row['seeds']}] drop(D) {row['drop_dense']} drop(C) {row['drop_control']} ms {row.get('ms', '-')} {row.get('colours', '')} gain {row.get('gain_vs_rewind', '')}")
    fields = list(dict.fromkeys(k for r in table for k in r))
    with (ROOT/"SUMMARY.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)


if __name__ == "__main__":
    main()
