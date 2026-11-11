#!/usr/bin/env python3
"""Plan M confidence intervals: KD 98% pooled over seeds 3-8, and the new models on CIFAR-100. FP32 test."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_bootstrap import C100, compare, loader


def main():
    test, targets = loader(True)
    path = lambda m, s, a: C100/f"cifar100_{m}_seed{s}_{a}/final_model.pt"
    rows = [compare("CIFAR-100 ResNet34 kd - gradual+floor 98%, seeds 3-8", [path("resnet34", s, "gradual_r98_floor_kd") for s in range(3, 9)],
                    [path("resnet34", s, "gradual_r98_floor") for s in range(3, 9)], 100, test, targets),
            compare("CIFAR-100 ResNet34 kd - gradual+floor 98%, seeds 6-8", [path("resnet34", s, "gradual_r98_floor_kd") for s in (6, 7, 8)],
                    [path("resnet34", s, "gradual_r98_floor") for s in (6, 7, 8)], 100, test, targets)]
    for m in ("googlenet", "densenet121"):
        for r in (98, 99):
            rows.append(compare(f"CIFAR-100 {m} gradual+floor - rewind {r}%", [path(m, s, f"gradual_r{r}_floor") for s in (0, 1, 2)],
                                [path(m, s, f"rewind_r{r}") for s in (0, 1, 2)], 100, test, targets, m))
    (C100/"BOOTSTRAP_PLAN_M.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
