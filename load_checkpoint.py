#!/usr/bin/env python3
"""Load a CIFAR-100 ResNet34 pruned checkpoint and check it. Needs only torch and torchvision.

    python3 load_checkpoint.py <run directory> [--data <dir holding cifar-100-python>] [--q-json <q_per_cluster json>]

The run directory holds final_model.pt or final_model.pt.gz (a plain state_dict, masks already multiplied
into the weights) and masks.pt ({conv name: bool tensor of shape (clusters, K)}, K = Cin*kh*kw in
(Cin, kh, kw) order; cluster c is filters 8c..8c+7, which share one mask).
CIFAR-100 is downloaded into --data if it is not there.
"""
import argparse, gzip, io, json
from pathlib import Path

import torch
from torch import nn
from torchvision import datasets, transforms
from torchvision.models import resnet34

MEAN, STD = (.5071, .4867, .4408), (.2675, .2565, .2761)


def make_model():
    """torchvision ResNet34 with the CIFAR stem: 3x3 stride-1 conv1, no maxpool, 100 classes."""
    model = resnet34(weights=None, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[2]/"data")
    parser.add_argument("--q-json", type=Path, default=None)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = make_model()
    path = args.run/"final_model.pt"
    state = torch.load(path if path.exists() else io.BytesIO(gzip.open(args.run/"final_model.pt.gz").read()), map_location="cpu")
    model.load_state_dict(state, strict=True)
    masks = torch.load(args.run/"masks.pt", map_location="cpu")

    convs = {n: m for n, m in model.named_modules() if isinstance(m, nn.Conv2d)}
    if set(convs) != set(masks):
        raise SystemExit("masks.pt layer names do not match the model")
    kept = total = 0
    for name, conv in convs.items():
        weight = conv.weight.detach()
        full = masks[name].bool().repeat_interleave(8, 0).reshape_as(weight)
        if torch.count_nonzero(weight[~full]):
            raise SystemExit(f"{name}: nonzero weight outside the mask")
        kept += int(full.sum())
        total += weight.numel()
    print(f"state_dict keys {len(state)}, conv layers {len(convs)}, clusters {sum(len(m) for m in masks.values())}")
    print(f"weights outside the masks are all zero; kept {kept} of {total}, pruning ratio {1-kept/total:.6f}")

    if args.q_json:
        layers = json.loads(args.q_json.read_text())["layers"]
        same = [l["name"] for l in layers] == list(convs) and all(l["q"] == masks[l["name"]].sum(1).tolist() for l in layers)
        print("q per cluster matches", args.q_json.name, ":", same)
        if not same:
            raise SystemExit("q mismatch")

    test = datasets.CIFAR100(args.data, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)]))
    loader = torch.utils.data.DataLoader(test, batch_size=500, num_workers=2)
    model.to(device).eval()
    for label, amp in (("FP32", False), ("AMP float16 (as in result.json)", True)):
        if amp and device.type != "cuda":
            continue
        correct = 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            for x, y in loader:
                correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
        print(f"test top-1 {label}: {correct}/{len(test)} = {correct/len(test):.4f}")


if __name__ == "__main__":
    main()
