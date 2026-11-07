#!/usr/bin/env python3
"""Paired image-level bootstrap and per-seed McNemar on frozen test predictions; never used for selection."""
import json, sys
from math import comb
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
import resnet34_pruning_experiments_path  # noqa: F401
import resnet34_pruning_experiments as core
from int8_eval_pruned import MEAN, STD

ROOT = Path("runs/confirmation_r0p95")
SEEDS = (3, 4, 5)


def correct(arm, seed, test, device):
    model = core.make_model("resnet50", 10, 32).to(device).eval()
    model.load_state_dict(torch.load(ROOT/f"seed{seed}_{arm}/final_model.pt", map_location=device, weights_only=False))
    with torch.inference_mode():
        return np.concatenate([(model(x.to(device)).argmax(1).cpu() == y).numpy() for x, y in test])


def mcnemar(a, b):
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n, k = only_a+only_b, min(only_a, only_b)
    return only_a, only_b, min(1., 2*sum(comb(n, i) for i in range(k+1))/2**n)


def main():
    device = torch.device("cuda")
    data = datasets.CIFAR10("data", train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)]))
    test = DataLoader(data, batch_size=512, shuffle=False, num_workers=2)
    out = {}
    for pair in sys.argv[1:]:
        arm, base = pair.split(":")
        a = np.stack([correct(arm, s, test, device) for s in SEEDS])
        b = np.stack([correct(base, s, test, device) for s in SEEDS])
        rng = np.random.default_rng(0)
        idx = rng.integers(0, 10000, (20000, 10000))
        diff = (a.astype(np.int8)-b.astype(np.int8)).mean(0)
        boot = diff[idx].mean(1)*100
        out[pair] = dict(mean_pp=float(diff.mean()*100), ci95_pp=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                         mcnemar={str(s): dict(zip(("arm_only", "base_only", "p"), mcnemar(a[i], b[i]))) for i, s in enumerate(SEEDS)})
        print(pair, json.dumps(out[pair]), flush=True)
    (ROOT/"PAIRED_BOOTSTRAP.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
