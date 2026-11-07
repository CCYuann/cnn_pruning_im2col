#!/usr/bin/env python3
"""Test accuracy of frozen seed3/4/5 models under the final Q15 flow: MSE scale, CLE k<=8, bias correction.

Usage: final_q15_eval.py <model> arm1,arm2,... [seeds, default 3,4,5]  Finished (model, arm, seed) rows are never recomputed.
The flow was chosen on validation; ResNet50 95-99% rows are a second look at test, because the first
MSE-only numbers in TEST_RESULTS.csv had already been read when the failure was diagnosed.
"""
import csv, sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
import resnet34_pruning_experiments_path  # noqa: F401
import resnet34_pruning_experiments as core
from int8_eval_pruned import MEAN, STD, loaders
from int8_hw_spec_eval import IntegerResNet, calibrate, equalize

ROOT = Path("runs/confirmation_r0p95")
OUT = ROOT/"TEST_Q15_FINAL.csv"


def main():
    device = torch.device("cuda")
    name, arms = sys.argv[1], sys.argv[2].split(",")
    data = datasets.CIFAR10("data", train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)]))
    test = DataLoader(data, batch_size=256, shuffle=False, num_workers=2)
    calibration, _ = loaders("data")
    done = list(csv.DictReader(OUT.open())) if OUT.exists() else []
    finished = {(r["model"], r["arm"], r["seed"]) for r in done}
    for seed in (int(s) for s in (sys.argv[3].split(",") if len(sys.argv) > 3 else "345")):
        for arm in arms:
            if (name, arm, str(seed)) in finished:
                continue
            model = core.make_model(name, 10, 32).to(device).eval()
            model.load_state_dict(torch.load(ROOT/f"seed{seed}_{arm}/final_model.pt", map_location=device, weights_only=False))
            with torch.inference_mode():
                fp32 = sum(int((model(x.to(device)).argmax(1).cpu() == y).sum()) for x, y in test)/10000
                q15, peak = "", ""
                if not arm.startswith("unstructured"):
                    equalize(model, calibration, device, 8.)
                    scale, logit = calibrate(model, calibration, device, .5)
                    integer = IntegerResNet(model, scale, logit, device)
                    integer.correct_bias(model, next(iter(calibration))[0][:256])
                    q15 = sum(int((integer(x).argmax(1).cpu() == y).sum()) for x, y in test)/10000
                    peak = integer.stats["peak"]
            done.append(dict(model=name, arm=arm, seed=seed, test_fp32=fp32, test_q15=q15, peak_acc=peak))
            with OUT.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(done[0]))
                writer.writeheader()
                writer.writerows(done)
            print(done[-1], flush=True)


if __name__ == "__main__":
    main()
