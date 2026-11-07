#!/usr/bin/env python3
"""CIFAR-100 test accuracy under the frozen Q15 flow. Same recipe as final_q15_eval.py, nothing is tuned here.

Usage: final_q15_cifar100.py <model> <run suffix,...> <seeds>, for example resnet34 rewind_r98,gradual_r98_floor 0,1,2
Calibration uses 2048 training images outside validation. Finished rows are never recomputed.
"""
import csv, sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
import resnet34_pruning_experiments_path  # noqa: F401
import resnet34_pruning_experiments as core
from dataset_pruning_worker import DATA
from int8_hw_spec_eval import IntegerResNet, calibrate, equalize

ROOT = Path("runs/other_datasets")
OUT = ROOT/"TEST_Q15_CIFAR100.csv"


def main():
    device = torch.device("cuda")
    name, arms, seeds = sys.argv[1], sys.argv[2].split(","), sys.argv[3].split(",")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*DATA["cifar100"]["norm"])])
    train = datasets.CIFAR100("data", train=True, transform=transform)
    test = DataLoader(datasets.CIFAR100("data", train=False, transform=transform), batch_size=256, shuffle=False, num_workers=2)
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(0)).tolist()
    calibration = DataLoader(Subset(train, order[5000:7048]), batch_size=256, shuffle=False, num_workers=2)
    done = list(csv.DictReader(OUT.open())) if OUT.exists() else []
    finished = {(r["model"], r["arm"], r["seed"]) for r in done}
    for seed in seeds:
        for arm in arms:
            if (name, arm, seed) in finished:
                continue
            model = core.make_model(name, 100, 32).to(device).eval()
            model.load_state_dict(torch.load(ROOT/f"cifar100_{name}_seed{seed}_{arm}/final_model.pt", map_location=device, weights_only=False))
            with torch.inference_mode():
                fp32 = sum(int((model(x.to(device)).argmax(1).cpu() == y).sum()) for x, y in test)/10000
                equalize(model, calibration, device, 8.)
                scale, logit = calibrate(model, calibration, device, .5)
                integer = IntegerResNet(model, scale, logit, device)
                integer.correct_bias(model, next(iter(calibration))[0][:256])
                q15 = sum(int((integer(x).argmax(1).cpu() == y).sum()) for x, y in test)/10000
            done.append(dict(model=name, arm=arm, seed=seed, test_fp32=fp32, test_q15=q15, peak_acc=integer.stats["peak"]))
            with OUT.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(done[0]))
                writer.writeheader()
                writer.writerows(done)
            print(done[-1], flush=True)


if __name__ == "__main__":
    main()
