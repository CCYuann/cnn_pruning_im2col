#!/usr/bin/env python3
"""Plan U: decompose gradual+floor minus rewind into quota allocation, position choice and the gradual process.

A0 rewind, A1 rewind with gradual's per-cluster q (Dense-L1 positions), A2 rewind with gradual's final mask, A3 gradual+floor.
Paired test diffs per step; +effect if mean >= +0.3pp and 3/3 > 0, -effect if mean <= -0.3pp and 3/3 < 0, else unclear.
Writes runs/other_datasets/PLAN_U_RESULTS.json.
"""
import json
from pathlib import Path

R = Path("runs/other_datasets")
SETS = [("resnet56", 95), ("vgg16_bn", 95), ("googlenet", 95), ("densenet121", 95), ("resnet110", 95), ("resnet34", 98), ("densenet121", 90)]
ARMS = {"A0": "rewind_r{r}", "A1": "rewind_r{r}_qfrom", "A2": "rewind_r{r}_mfrom", "A3": "gradual_r{r}_floor"}
STEPS = (("allocation", "A1", "A0"), ("positions", "A2", "A1"), ("process", "A3", "A2"))


def get(m, r, s, arm):
    p = R/f"cifar100_{m}_seed{s}_{ARMS[arm].format(r=r)}"/"result.json"
    return json.loads(p.read_text()) if p.exists() else None


def call(d):
    mean = sum(d)/3
    if mean >= .3 and all(x > 0 for x in d):
        return "+effect"
    if mean <= -.3 and all(x < 0 for x in d):
        return "-effect"
    return "unclear"


out = {}
for m, r in SETS:
    res = {a: [get(m, r, s, a) for s in (0, 1, 2)] for a in ARMS}
    if any(x is None for v in res.values() for x in v):
        print(f"{m:11s} {r}%: pending ({sum(x is None for v in res.values() for x in v)} missing)")
        continue
    acc = {a: [x["final_test"]*100 for x in v] for a, v in res.items()}
    tr = {a: sum(x["final_train"] for x in v)/3*100 for a, v in res.items()}
    total = sum(a-b for a, b in zip(acc["A3"], acc["A0"]))/3
    rec = dict(test={a: sum(v)/3 for a, v in acc.items()}, train=tr, total=total, steps={})
    line = f"{m:11s} {r}%: " + " ".join(f"{a} {sum(v)/3:.2f}(tr {tr[a]:.1f})" for a, v in acc.items()) + f" | total {total:+.2f}"
    for name, hi, lo in STEPS:
        d = [a-b for a, b in zip(acc[hi], acc[lo])]
        mean = sum(d)/3
        rec["steps"][name] = dict(diffs=d, mean=mean, call=call(d), share=mean/total if abs(total) > 1e-9 else None)
        line += f" | {name} {'/'.join(f'{x:+.2f}' for x in d)} ({mean:+.2f}) {call(d)}"
    print(line)
    out[f"{m}_r{r}"] = rec
(R/"PLAN_U_RESULTS.json").write_text(json.dumps(out, indent=1))
