#!/usr/bin/env python3
"""Plan H verdicts from validation accuracy and V3 cycles. Never reads final_test."""
import csv, json, subprocess, sys
from pathlib import Path

import numpy as np

ROOT = Path("runs/other_datasets")
CYCLES = Path(__file__).resolve().parents[2]/"mask_aware_conv_accelerator/scripts/estimate_network_cycles.py"
SEEDS = (0, 1, 2)
ARMS = {"floor": "floor", "fs": "floor_stem", "b15": "floor_stem_n15", "mac.25": "floor_stem_mac0.25", "mac.5": "floor_stem_mac0.5"}


def directory(arm, ratio, seed):
    return ROOT/f"cifar100_resnet34_seed{seed}_gradual_r{ratio}_{ARMS[arm]}"


def main():
    done = [(a, r, s) for a in ARMS for r in (95, 98, 99) for s in SEEDS if (directory(a, r, s)/"result.json").exists()]
    labels = [f"{a}_{r}_{s}={directory(a, r, s).resolve()}/masks.pt" for a, r, s in done]
    out = ROOT/"PLAN_H_CYCLES.csv"
    subprocess.run([sys.executable, str(CYCLES), "--geometry", str(ROOT/"geometry_cifar100_resnet34.json"), "--masks", *labels, "--out", str(out)], check=True, stdout=subprocess.DEVNULL)
    ms = {r["label"]: float(r["ms_50mhz"]) for r in csv.DictReader(out.open())}
    val = {(a, r, s): json.loads((directory(a, r, s)/"result.json").read_text())["final_validation"]*100 for a, r, s in done}
    get = lambda a, r: np.array([val[a, r, s] for s in SEEDS])
    time = lambda a, r: np.array([ms[f"{a}_{r}_{s}"] for s in SEEDS])
    for a in ARMS:
        for r in (95, 98, 99):
            if all((a, r, s) in val for s in SEEDS):
                print(f"{a:7s} {r}% val {np.round(get(a, r), 2)} mean {get(a, r).mean():.3f} ms {time(a, r).mean():.2f}")
    print("\nH1 fs - floor")
    for r in (98, 99):
        d = get("fs", r)-get("floor", r)
        print(f" {r}%", np.round(d, 2), f"mean {d.mean():+.3f}", "PASS" if d.mean() >= .3 and (d > 0).all() else "fail")
    print("H2 b15 - fs")
    for r in (98, 99):
        d, c = get("b15", r)-get("fs", r), time("b15", r).sum()/time("fs", r).sum()-1
        print(f" {r}%", np.round(d, 2), f"mean {d.mean():+.3f} cycles {c*100:+.2f}%", "PASS" if r == 99 and c <= -.05 and d.mean() >= -.1 else "-")
    print("H3 equal-cycle gain against the fs curve")
    for a in ("mac.25", "mac.5"):
        for r in (98, 99):
            gains = []
            for i, s in enumerate(SEEDS):
                x, y = np.log([ms[f"fs_{q}_{s}"] for q in (99, 98, 95)]), [val["fs", q, s] for q in (99, 98, 95)]
                here = np.log(ms[f"{a}_{r}_{s}"])
                gains.append(val[a, r, s]-np.interp(here, x, y) if x[0] <= here <= x[-1] else float("nan"))
            gains = np.array(gains)
            verdict = "outside fs range" if np.isnan(gains).any() else ("PASS" if gains.mean() >= .3 and (gains > 0).all() else "fail")
            print(f" {a} {r}% ms {time(a, r).mean():.2f}", np.round(gains, 2), f"mean {np.nanmean(gains):+.3f}", verdict)


if __name__ == "__main__":
    main()
