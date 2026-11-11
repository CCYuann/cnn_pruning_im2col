#!/usr/bin/env python3
"""Hardware ranges under the frozen Q15 flow (MSE scale, CLE k<=8, bias correction), for the accelerator team.

Per conv layer, over real validation images (never test):
  S1  multiplier and right_shift per output channel (raw before the 48 clamp, and as sent to RTL)
  S2  acc (conv sum before bias), bias, acc+bias and (acc+bias)*multiplier ranges, with signed bit widths
  S4  share of zero input activations, and share of kept-weight MACs whose activation is zero (padding excluded)
Also reports identity requantization and classifier multipliers/shifts, and the Q15 validation accuracy as a sanity check.

Usage: hw_range_report.py [--images N]   Writes runs/hw_ranges_2026-11-11/{layers.csv,summary.csv}.
"""
import argparse, csv, sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
import resnet34_pruning_experiments_path  # noqa: F401
import resnet34_pruning_experiments as core
from dataset_pruning_worker import DATA
from int8_eval_pruned import MEAN, STD
from int8_hw_spec_eval import FRACTION, IntConv, IntegerResNet, blocks, calibrate, equalize, quantize_multiplier, requant

OUT = Path("runs/hw_ranges_2026-11-11")
MODELS = [  # (label, dataset, path, note)
    ("c10_fs98_seed6", "cifar10", "runs/confirmation_r0p95/seed6_gradual_r34r98_fs", "RTL bit-exact verified (V2.3)"),
    ("c100_kd98_seed3", "cifar100", "runs/other_datasets/cifar100_resnet34_seed3_gradual_r98_floor_kd", "RTL bit-exact verified (V2.3)"),
] + [(f"c100_floor90_seed{s}", "cifar100", f"runs/other_datasets/cifar100_resnet34_seed{s}_gradual_r90_floor", "90% main line") for s in range(3)]


def bits(lo, hi):
    """Signed two's-complement width that holds [lo, hi]."""
    m = max(-lo-1, hi, 0)
    return int(m).bit_length()+1


def data(dataset, n):
    if dataset == "cifar10":
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
        train = datasets.CIFAR10("data", train=True, transform=tf)
        order = torch.randperm(50000, generator=torch.Generator().manual_seed(0)).tolist()
    else:
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*DATA["cifar100"]["norm"])])
        train = datasets.CIFAR100("data", train=True, transform=tf)
        order = torch.randperm(50000, generator=torch.Generator().manual_seed(0)).tolist()
    kw = dict(batch_size=250, shuffle=False, num_workers=2)
    # Same splits as the frozen flow: calibration order[5000:7048], validation order[:5000].
    return DataLoader(Subset(train, order[5000:7048]), **kw), DataLoader(Subset(train, order[:n]), **kw)


def name_layers(model, integer):
    integer.stem.name = "conv1"
    for (prefix, block), (ops, last, short, _, _) in zip(blocks(model), integer.items):
        for i, op in enumerate(ops):
            op.name = f"{prefix}.conv{i+1}"
        last.name = f"{prefix}.conv{len(ops)+1}"
        if short is not None:
            short.name = f"{prefix}.downsample"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=5000)
    args = parser.parse_args()
    device = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    layer_rows, summary = [], []
    original_call = IntConv.__call__
    for label, dataset, path, note in MODELS:
        model = core.make_model("resnet34", 10 if dataset == "cifar10" else 100, 32).to(device).eval()
        model.load_state_dict(torch.load(Path(path)/"final_model.pt", map_location=device, weights_only=False))
        calibration, validation = data(dataset, args.images)
        with torch.inference_mode():
            equalize(model, calibration, device, 8.)
            scale, logit = calibrate(model, calibration, device, .5)
            integer = IntegerResNet(model, scale, logit, device)
            integer.correct_bias(model, next(iter(calibration))[0][:256])
        name_layers(model, integer)
        stats = {}

        def record(self, x):
            s = stats.setdefault(self.name, dict(acc=[0, 0], bias=[0, 0], biased=[0, 0], product=[0, 0], zeros=0, values=0, zero_macs=0, macs=0, signed_input=False))
            plain = F.conv2d(x.double(), self.weight, stride=self.stride, padding=self.padding).round().long()
            biased = plain+self.bias
            product = biased*self.multiplier
            for key, v in (("acc", plain), ("bias", self.bias), ("biased", biased), ("product", product)):
                s[key][0], s[key][1] = min(s[key][0], int(v.min())), max(s[key][1], int(v.max()))
            s["signed_input"] |= bool((x < 0).any())
            s["zeros"] += int((x == 0).sum())
            s["values"] += x.numel()
            kept = (self.weight != 0).double()
            s["zero_macs"] += float(F.conv2d((x == 0).double(), kept, stride=self.stride, padding=self.padding).sum())
            s["macs"] += float(F.conv2d(torch.ones_like(x, dtype=torch.double), kept, stride=self.stride, padding=self.padding).sum())
            return original_call(self, x)

        IntConv.__call__ = record
        good = total = 0
        try:
            with torch.inference_mode():
                for x, y in validation:
                    good += int((integer(x).argmax(1).cpu() == y).sum())
                    total += len(y)
        finally:
            IntConv.__call__ = original_call
        convs = [integer.stem]+[c for ops, last, short, _, _ in integer.items for c in [*ops, last]+([short] if short is not None else [])]
        for conv in convs:
            s = stats[conv.name]
            raw = [FRACTION-quantize_multiplier(float(u)/conv.s_out, FRACTION).shift for u in conv.unit.flatten().tolist()]
            rs, mul = conv.right_shift.flatten(), conv.multiplier.flatten()
            layer_rows.append(dict(model=label, layer=conv.name, out_channels=len(rs),
                                   multiplier_min=int(mul.min()), multiplier_max=int(mul.max()),
                                   right_shift_raw_min=min(raw), right_shift_raw_max=max(raw), right_shift_min=int(rs.min()), right_shift_max=int(rs.max()),
                                   channels_raw_shift_gt48=sum(r > 48 for r in raw),
                                   acc_min=s["acc"][0], acc_max=s["acc"][1], acc_bits=bits(*s["acc"]),
                                   bias_min=s["bias"][0], bias_max=s["bias"][1], bias_bits=bits(*s["bias"]),
                                   acc_bias_min=s["biased"][0], acc_bias_max=s["biased"][1], acc_bias_bits=bits(*s["biased"]),
                                   product_min=s["product"][0], product_max=s["product"][1], product_bits=bits(*s["product"]),
                                   signed_input=s["signed_input"], input_zero_share=s["zeros"]/s["values"], mac_zero_share=s["zero_macs"]/s["macs"]))
        identity = [(int(m), int(r)) for _, _, short, m, r in integer.items if short is None]
        head_shift = (FRACTION-integer.head.shifts).tolist()
        rows = [r for r in layer_rows if r["model"] == label]
        pr = [r for r in rows if not r["signed_input"]]
        summary.append(dict(model=label, note=note, images=total, q15_validation=good/total,
                            conv_right_shift_raw_max=max(r["right_shift_raw_max"] for r in rows), conv_right_shift_min=min(r["right_shift_min"] for r in rows),
                            channels_raw_shift_gt48=sum(r["channels_raw_shift_gt48"] for r in rows), channels_total=sum(r["out_channels"] for r in rows),
                            identity_right_shift=f"{min(r for _, r in identity)}..{max(r for _, r in identity)}", identity_multiplier=f"{min(m for m, _ in identity)}..{max(m for m, _ in identity)}",
                            head_right_shift=f"{min(head_shift)}..{max(head_shift)}",
                            acc_bits_max=max(r["acc_bits"] for r in rows), bias_bits_max=max(r["bias_bits"] for r in rows),
                            acc_bias_bits_max=max(r["acc_bias_bits"] for r in rows), product_bits_max=max(r["product_bits"] for r in rows),
                            layers_bias_over_28bit=sum(r["bias_bits"] > 28 for r in rows), layers_acc_bias_over_28bit=sum(r["acc_bias_bits"] > 28 for r in rows),
                            mac_zero_share_postrelu_mean=sum(r["mac_zero_share"]*1 for r in pr)/len(pr)))
        print(summary[-1], flush=True)
        for name, rows_ in (("layers.csv", layer_rows), ("summary.csv", summary)):
            with (OUT/name).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows_[0]))
                writer.writeheader()
                writer.writerows(rows_)


if __name__ == "__main__":
    main()
