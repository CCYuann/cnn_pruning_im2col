#!/usr/bin/env python3
"""Paired test-set confidence intervals for the main claims, FP32, from finished models. Reads no new training.

Image bootstrap resamples test images jointly across seeds, so it covers test-set sampling only.
The seed interval uses the t distribution over paired seed differences, so it covers seed variation only.
"""
import json, sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
import resnet34_pruning_experiments_path  # noqa: F401
import resnet34_pruning_experiments as core
from dataset_pruning_worker import DATA
from int8_eval_pruned import MEAN, STD

C10, C100 = Path("runs/confirmation_r0p95"), Path("runs/other_datasets")
DRAWS = 10000
device = torch.device("cuda")


def loader(cifar100):
    if cifar100:
        data = datasets.CIFAR100("data", train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(*DATA["cifar100"]["norm"])]))
    else:
        data = datasets.CIFAR10("data", train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)]))
    return DataLoader(data, batch_size=512, shuffle=False, num_workers=2), np.array(data.targets)


@torch.inference_mode()
def correct(path, classes, test, targets):
    model = core.make_model("resnet34", classes, 32).to(device).eval()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    return np.concatenate([model(x.to(device)).argmax(1).cpu().numpy() for x, _ in test]) == targets


def compare(label, a, b, classes, test, targets):
    pairs = [correct(p, classes, test, targets).astype(float)-correct(q, classes, test, targets).astype(float) for p, q in zip(a, b)]
    pairs = np.stack(pairs)
    rng = np.random.default_rng(0)
    draws = [pairs[:, rng.integers(0, pairs.shape[1], pairs.shape[1])].mean()*100 for _ in range(DRAWS)]
    seeds = pairs.mean(1)*100
    half = stats.t.ppf(.975, len(seeds)-1)*seeds.std(ddof=1)/len(seeds)**.5
    row = dict(label=label, n_seeds=len(seeds), diff=seeds.mean(), seed_diffs=seeds.round(2).tolist(), image_ci95=np.percentile(draws, [2.5, 97.5]).round(3).tolist(), seed_ci95=[seeds.mean()-half, seeds.mean()+half])
    print(json.dumps(row), flush=True)
    return row


def main():
    rows = []
    test, targets = loader(False)
    seeds = (6, 7, 8, 9, 10, 11)
    for r in (98, 99):
        rows.append(compare(f"CIFAR-10 fs - rewind {r}%", [C10/f"seed{s}_gradual_r34r{r}_fs/final_model.pt" for s in seeds], [C10/f"seed{s}_rewind_r34r{r}/final_model.pt" for s in seeds], 10, test, targets))
    test, targets = loader(True)
    path = lambda s, a: C100/f"cifar100_resnet34_seed{s}_{a}/final_model.pt"
    for r in (98, 99):
        rows.append(compare(f"CIFAR-100 gradual+floor - rewind {r}%", [path(s, f"gradual_r{r}_floor") for s in (0, 1, 2)], [path(s, f"rewind_r{r}") for s in (0, 1, 2)], 100, test, targets))
    rows.append(compare("CIFAR-100 kd - gradual+floor 98%", [path(s, "gradual_r98_floor_kd") for s in (3, 4, 5)], [path(s, "gradual_r98_floor") for s in (3, 4, 5)], 100, test, targets))
    (C100/"BOOTSTRAP_FINAL.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
