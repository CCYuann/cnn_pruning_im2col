#!/usr/bin/env python3
"""INT8 post-training quantization of pruned models on the 5000-image validation split.

oneDNN affine-activation FX PTQ (not the accelerator's numeric spec; see int8_hw_spec_eval.py).
Calibration uses 2048 training images outside validation.
"""
import argparse, csv, json, sys
from pathlib import Path

import torch
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
import resnet34_pruning_experiments_path  # noqa: F401
import resnet34_pruning_experiments as core

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


def loaders(model_data, batch=256):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    data = datasets.CIFAR10(model_data, train=True, transform=transform)
    order = torch.randperm(50000, generator=torch.Generator().manual_seed(0)).tolist()
    kwargs = dict(batch_size=batch, shuffle=False, num_workers=2)
    return DataLoader(Subset(data, order[5000:7048]), **kwargs), DataLoader(Subset(data, order[:5000]), **kwargs)


@torch.inference_mode()
def accuracy(model, loader):
    good = sum(int((model(x).argmax(1) == y).sum()) for x, y in loader)
    return good/len(loader.dataset)


def quantize(model, calibration, mode):
    example = (torch.randn(1, 3, 32, 32),)
    torch.backends.quantized.engine = "onednn"
    mapping = get_default_qconfig_mapping("onednn")
    prepared = prepare_fx(model.eval(), mapping, example)
    with torch.inference_mode():
        for x, _ in calibration:
            prepared(x)
    return convert_fx(prepared)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", help="label=model_name=path=expected_validation")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--modes", default="onednn")
    args = parser.parse_args()
    torch.set_num_threads(6)
    calibration, validation = loaders("data")
    done = list(csv.DictReader(args.out.open())) if args.out.exists() else []
    finished = {(r["label"], r["mode"]) for r in done}
    for item in args.models:
        label, name, path, expected = item.split("=")
        for mode in args.modes.split(","):
            if (label, mode) in finished:
                continue
            model = core.make_model(name, 10, 32)
            state = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(state.get("model_state_dict", state) if isinstance(state, dict) else state)
            model.eval()
            fp32 = accuracy(model, validation)
            if abs(fp32-float(expected)) > 3e-3:
                raise ValueError(f"{label}: FP32 {fp32} disagrees with recorded {expected}")
            int8 = accuracy(quantize(model, calibration, mode), validation)
            done.append(dict(label=label, mode=mode, fp32=fp32, int8=int8, loss_pp=(fp32-int8)*100))
            with args.out.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(done[0]))
                writer.writeheader()
                writer.writerows(done)
            print(done[-1], flush=True)


if __name__ == "__main__":
    main()
