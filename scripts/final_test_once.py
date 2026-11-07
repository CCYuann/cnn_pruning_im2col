#!/usr/bin/env python3
"""One-shot CIFAR-10 test evaluation of frozen confirmation models; finished labels are never recomputed."""
import csv, sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
import resnet34_pruning_experiments_path  # noqa: F401
import resnet34_pruning_experiments as core
from int8_eval_pruned import MEAN, STD, loaders
from int8_hw_spec_eval import IntegerResNet, calibrate

OUT = Path("runs/confirmation_r0p95/TEST_RESULTS.csv")
ROOT = Path("runs/confirmation_r0p95")


def main():
    device = torch.device("cuda")
    arms = sys.argv[1].split(",")
    data = datasets.CIFAR10("data", train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)]))
    test = DataLoader(data, batch_size=256, shuffle=False, num_workers=2)
    calibration, _ = loaders("data")
    done = list(csv.DictReader(OUT.open())) if OUT.exists() else []
    finished = {(r["arm"], r["seed"]) for r in done}
    for seed in (3, 4, 5):
        for arm in arms:
            if (arm, str(seed)) in finished:
                continue
            model = core.make_model("resnet50", 10, 32).to(device).eval()
            if arm == "dense":
                state = torch.load(ROOT/f"checkpoints/resnet50_seed{seed}_dense.pt", map_location=device, weights_only=False)["model_state_dict"]
            else:
                state = torch.load(ROOT/f"seed{seed}_{arm}/final_model.pt", map_location=device, weights_only=False)
            model.load_state_dict(state)
            with torch.inference_mode():
                fp32 = sum(int((model(x.to(device)).argmax(1).cpu() == y).sum()) for x, y in test)/10000
                q15 = ""
                if arm != "dense" and not arm.startswith("unstructured"):
                    scale, logit = calibrate(model, calibration, device, .5)
                    integer = IntegerResNet(model, scale, logit, device)
                    q15 = sum(int((integer(x).argmax(1).cpu() == y).sum()) for x, y in test)/10000
            done.append(dict(arm=arm, seed=seed, test_fp32=fp32, test_q15=q15))
            with OUT.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(done[0]))
                writer.writeheader()
                writer.writerows(done)
            print(done[-1], flush=True)


if __name__ == "__main__":
    main()
