#!/usr/bin/env python3
"""Plan X: LR grid {0.0003, 0.001, 0.003, 0.01, 0.03} for the paper flow (ft_const) at 90%.

--select : validation only; writes PLAN_X_LR_SELECTION.json (refuses to overwrite an existing selection).
default  : reads the saved selection, then test; X1 gradual+floor vs chosen ft_const (paired, >= +0.3pp and 3/3).
Writes runs/other_datasets/PLAN_X_RESULTS.json.
"""
import json, sys
from pathlib import Path

R = Path("runs/other_datasets")
MODELS = ("resnet18", "resnet34", "resnet50", "resnet56", "resnet110", "vgg16_bn", "googlenet", "densenet121")
LRS = (.0003, .001, .003, .01, .03)
SEEDS = (0, 1, 2)
SEL = R/"PLAN_X_LR_SELECTION.json"
W = json.loads((R/"PLAN_W_RESULTS.json").read_text())


def run(m, s, suffix):
    p = R/f"cifar100_{m}_seed{s}_{suffix}"/"result.json"
    return json.loads(p.read_text()) if p.exists() else None


def ft(m, s, lr):
    return run(m, s, "ft_const_r90" if lr == .001 else f"ft_const_r90_lr{lr:g}")


grid = {m: {lr: [ft(m, s, lr) for s in SEEDS] for lr in LRS} for m in MODELS}
missing = {m: sum(x is None for v in g.values() for x in v) for m, g in grid.items()}
if any(missing.values()):
    print("pending:", {m: n for m, n in missing.items() if n})
    sys.exit(0)

if "--select" in sys.argv:
    if SEL.exists():
        sys.exit(f"{SEL} already exists; the selection is frozen")
    sel = {}
    for m in MODELS:
        val = {lr: sum(r["final_validation"] for r in v)/3*100 for lr, v in grid[m].items()}
        best = max(LRS, key=lambda lr: (round(val[lr], 6), -lr))
        sel[m] = dict(validation={f"{lr:g}": v for lr, v in val.items()}, chosen_lr=best, at_lower_bound=best == min(LRS),
                      at_upper_bound=best == max(LRS), same_as_plan_w=best == W[m]["chosen_lr"])
        print(f"{m:11s} val " + " ".join(f"{lr:g}:{val[lr]:.2f}" for lr in LRS) + f" -> {best:g}"
              + (" (LOWER BOUND)" if best == min(LRS) else "") + ("" if best == W[m]["chosen_lr"] else f" (plan W chose {W[m]['chosen_lr']:g})"))
    SEL.write_text(json.dumps(sel, indent=1))
    sys.exit(0)

if not SEL.exists():
    sys.exit("Run with --select first")
sel = json.loads(SEL.read_text())
out = {}
for m in MODELS:
    best = sel[m]["chosen_lr"]
    test = {lr: [r["final_test"]*100 for r in v] for lr, v in grid[m].items()}
    f = [run(m, s, "gradual_r90_floor")["final_test"]*100 for s in SEEDS]
    w = [run(m, s, "rewind_r90")["final_test"]*100 for s in SEEDS]
    d = [a-b for a, b in zip(f, test[best])]
    mean = sum(d)/3
    verdict = "PASS" if mean >= .3 and all(x > 0 for x in d) else "fail"
    d2 = [a-b for a, b in zip(w, test[best])]
    out[m] = dict(chosen_lr=best, reused_from_plan_w=sel[m]["same_as_plan_w"], at_lower_bound=sel[m]["at_lower_bound"],
                  ft_test={f"{lr:g}": sum(v)/3 for lr, v in test.items()}, floor_test=sum(f)/3, rewind_test=sum(w)/3,
                  X1=dict(diffs=d, mean=mean, verdict=verdict), X2_rewind_minus_ft=dict(diffs=d2, mean=sum(d2)/3))
    print(f"{m:11s} chosen {best:g}{' [same as W, reused]' if sel[m]['same_as_plan_w'] else ' [NEW]'}{' (LOWER BOUND)' if sel[m]['at_lower_bound'] else ''}"
          f" | ft {sum(test[best])/3:.2f} | floor {sum(f)/3:.2f} | X1 {'/'.join(f'{x:+.2f}' for x in d)} ({mean:+.2f}) {verdict}"
          f" | X2 rewind-ft {sum(d2)/3:+.2f} | all-LR ft test " + " ".join(f"{lr:g}:{sum(v)/3:.2f}" for lr, v in test.items()))
(R/"PLAN_X_RESULTS.json").write_text(json.dumps(out, indent=1))
