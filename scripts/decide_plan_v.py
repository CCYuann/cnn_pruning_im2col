#!/usr/bin/env python3
"""Plan V: replicate plan U's decomposition on seeds 3/4/5 (ResNet-56, VGG16-BN, 95%). Writes PLAN_V_RESULTS.json."""
import json
from pathlib import Path

R = Path("runs/other_datasets")
ARMS = {"A0": "rewind_r95", "A1": "rewind_r95_qfrom", "A2": "rewind_r95_mfrom", "A3": "gradual_r95_floor"}
STEPS = (("allocation", "A1", "A0"), ("positions", "A2", "A1"), ("process", "A3", "A2"))
SEEDS = (3, 4, 5)


def call(d):
    mean = sum(d)/len(d)
    if mean >= .3 and all(x > 0 for x in d):
        return "+effect"
    if mean <= -.3 and all(x < 0 for x in d):
        return "-effect"
    return "unclear"


out, calls = {}, {}
for m in ("resnet56", "vgg16_bn"):
    paths = {a: [R/f"cifar100_{m}_seed{s}_{v}"/"result.json" for s in SEEDS] for a, v in ARMS.items()}
    missing = sum(not p.exists() for v in paths.values() for p in v)
    if missing:
        print(f"{m:9s}: pending ({missing} missing)")
        continue
    acc = {a: [json.loads(p.read_text())["final_test"]*100 for p in v] for a, v in paths.items()}
    tr = {a: sum(json.loads(p.read_text())["final_train"] for p in v)/3*100 for a, v in paths.items()}
    rec = dict(test={a: sum(v)/3 for a, v in acc.items()}, train=tr, steps={})
    line = f"{m:9s} seeds 3/4/5: " + " ".join(f"{a} {sum(v)/3:.2f}(tr {tr[a]:.1f})" for a, v in acc.items())
    for name, hi, lo in STEPS:
        d = [a-b for a, b in zip(acc[hi], acc[lo])]
        rec["steps"][name] = dict(diffs=d, mean=sum(d)/3, call=call(d))
        calls[(m, name)] = call(d)
        line += f" | {name} {'/'.join(f'{x:+.2f}' for x in d)} ({sum(d)/3:+.2f}) {call(d)}"
    print(line)
    out[m] = rec
if len(out) == 2:
    claims = {"V1 process positive (both models)": all(calls[(m, "process")] == "+effect" for m in out),
              "V2 positions positive (both models)": all(calls[(m, "positions")] == "+effect" for m in out),
              "V3 allocation alone negative (ResNet-56)": calls[("resnet56", "allocation")] == "-effect"}
    for k, v in claims.items():
        print(f"{k}: {'REPLICATED' if v else 'not replicated'}")
    out["claims"] = claims
(R/"PLAN_V_RESULTS.json").write_text(json.dumps(out, indent=1))
