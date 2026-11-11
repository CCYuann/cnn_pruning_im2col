#!/usr/bin/env python3
"""Plan W: pick each model's constant fine-tuning LR for the paper flow (ft_const) on validation, then judge
gradual+floor vs that ft_const on test (paired, >= +0.3pp and 3/3 positive). 90%, CIFAR-100, seeds 0/1/2.
Writes runs/other_datasets/PLAN_W_RESULTS.json.
"""
import json
from pathlib import Path

R = Path("runs/other_datasets")
MODELS = ("resnet18", "resnet34", "resnet50", "resnet56", "resnet110", "vgg16_bn", "googlenet", "densenet121")
LRS = (.001, .003, .01, .03)
SEEDS = (0, 1, 2)


def run(m, s, suffix):
    p = R/f"cifar100_{m}_seed{s}_{suffix}"/"result.json"
    return json.loads(p.read_text()) if p.exists() else None


def ft(m, s, lr):
    return run(m, s, "ft_const_r90" if lr == .001 else f"ft_const_r90_lr{lr:g}")


out = {}
for m in MODELS:
    grid = {lr: [ft(m, s, lr) for s in SEEDS] for lr in LRS}
    floor = [run(m, s, "gradual_r90_floor") for s in SEEDS]
    rewind = [run(m, s, "rewind_r90") for s in SEEDS]
    missing = sum(x is None for v in grid.values() for x in v)+sum(x is None for x in floor+rewind)
    if missing:
        print(f"{m:11s}: pending ({missing} missing)")
        continue
    # Selection uses validation only; ties go to the smaller LR.
    val = {lr: sum(r["final_validation"] for r in v)/3*100 for lr, v in grid.items()}
    best = max(LRS, key=lambda lr: (round(val[lr], 6), -lr))
    test = {lr: [r["final_test"]*100 for r in v] for lr, v in grid.items()}
    f = [r["final_test"]*100 for r in floor]
    w = [r["final_test"]*100 for r in rewind]
    d = [a-b for a, b in zip(f, test[best])]
    mean = sum(d)/3
    verdict = "PASS" if mean >= .3 and all(x > 0 for x in d) else "fail"
    d2 = [a-b for a, b in zip(w, test[best])]
    rec = dict(validation=val, chosen_lr=best, chosen_at_upper_bound=best == max(LRS),
               ft_test={lr: sum(v)/3 for lr, v in test.items()}, floor_test=sum(f)/3, rewind_test=sum(w)/3,
               W1=dict(diffs=d, mean=mean, verdict=verdict), W2_rewind_minus_ft=dict(diffs=d2, mean=sum(d2)/3))
    out[m] = rec
    print(f"{m:11s} val " + " ".join(f"{lr:g}:{val[lr]:.2f}" for lr in LRS) + f" -> chosen {best:g}{' (UPPER BOUND)' if best == max(LRS) else ''}"
          f" | ft test {sum(test[best])/3:.2f} | floor {sum(f)/3:.2f} | W1 {'/'.join(f'{x:+.2f}' for x in d)} ({mean:+.2f}) {verdict}"
          f" | W2 rewind-ft {sum(d2)/3:+.2f} | all-LR ft test " + " ".join(f"{lr:g}:{sum(v)/3:.2f}" for lr, v in test.items()))
(R/"PLAN_W_RESULTS.json").write_text(json.dumps(out, indent=1))
