#!/usr/bin/env python3
"""Provable constant-channel rewrite (roadmap A5) and the V3 cycles it frees (roadmap B2), on the S1-S4 models.

Interval analysis over the integer network: every tensor channel gets a [lo, hi] that holds for ANY input image
(pixels in [0, 1] before normalization). A conv channel whose output interval is a single value is provably
constant and is rewritten as weights 0, bias = v, multiplier 2^14, right_shift 14 (output exactly v).
Branch convs are judged after their ReLU clamp; the last conv of a block and the shortcut conv before the residual add.

Cycles (V3, 31 + tiles*max(q,15) per cluster, 50 MHz, conv only), two lossless reductions:
  (a) a cluster whose 8 filters are all provably constant becomes bias-only (q=0);
  (b) kept positions that read an input channel provably equal to 0 everywhere (padding is 0 too) are dropped.
Validation only (5000 images): rewritten logits must equal the original logits bit for bit. Test is not read.
Writes runs/hw_ranges_2026-11-11/provable/{summary.csv,layers.csv,report.txt}.
"""
import csv, json, sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hw_range_report import MODELS, bits, data, name_layers
import resnet34_pruning_experiments as core
from dataset_pruning_worker import DATA
from int8_hw_spec_eval import FRACTION, blocks, calibrate, equalize, IntegerResNet, quantize_multiplier, requant, ties_away

OUT = Path("runs/hw_ranges_2026-11-11/provable")
GEOMETRY = json.loads(Path("runs/other_datasets/geometry_cifar100_resnet34.json").read_text())
CIFAR10 = ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))


def conv_interval(op, lo, hi):
    """Output interval of an IntConv (before any clamp) given per-input-channel integer intervals."""
    if op.padding[0] > 0:  # border taps read 0
        lo, hi = lo.clamp(max=0), hi.clamp(min=0)
    w = op.weight.double()
    wp, wn = w.clamp(min=0), w.clamp(max=0)
    l, h = lo.double()[None, :, None, None], hi.double()[None, :, None, None]
    acc_hi = (wp*h+wn*l).sum((1, 2, 3)).round().long()+op.bias.flatten()
    acc_lo = (wp*l+wn*h).sum((1, 2, 3)).round().long()+op.bias.flatten()
    shape = lambda t: t[None, :, None, None]
    out_lo = requant(shape(acc_lo), op.multiplier, op.right_shift).flatten()
    out_hi = requant(shape(acc_hi), op.multiplier, op.right_shift).flatten()
    return out_lo, out_hi, acc_lo, acc_hi


def rewrite(op, constant, value):
    idx = constant.nonzero().flatten()
    op.weight[idx] = 0
    op.bias[0, idx, 0, 0] = value[idx]
    op.multiplier = op.multiplier.expand_as(op.bias).clone()
    op.right_shift = op.right_shift.expand_as(op.bias).clone()
    op.multiplier[0, idx, 0, 0] = 1 << 14
    op.right_shift[0, idx, 0, 0] = 14


def raw_shift(op):
    return torch.tensor([FRACTION-quantize_multiplier(float(u)/op.s_out, FRACTION).shift for u in op.unit.flatten().tolist()])


def analyse(model, integer, mean, std, device):
    """Interval pass. Returns per-op records; rewrites provably constant channels in place."""
    s_in = integer.scale["input"]
    q = lambda v: ties_away(torch.tensor(v, dtype=torch.float64)/s_in).clamp(-127, 127).long().to(device)
    lo, hi = q([(0-m)/s for m, s in zip(mean, std)]), q([(1-m)/s for m, s in zip(mean, std)])
    records = []

    def visit(op, lo, hi, clamp):
        shift_before = raw_shift(op)
        out_lo, out_hi, _, _ = conv_interval(op, lo, hi)
        if clamp:
            out_lo, out_hi = out_lo.clamp(0, 127), out_hi.clamp(0, 127)
        constant = out_lo == out_hi
        zero_inputs = (lo == 0) & (hi == 0)
        rec = dict(op=op, name=op.name, constant=constant.cpu(), zero_inputs=zero_inputs.cpu(), shift_before=shift_before)
        rewrite(op, constant, out_lo)
        _, _, acc_lo, acc_hi = conv_interval(op, lo, hi)  # worst case after the rewrite
        rec.update(acc_bound=bits(int(acc_lo.min()), int(acc_hi.max())), live=(~constant).cpu())
        records.append(rec)
        return out_lo, out_hi

    lo, hi = visit(integer.stem, lo, hi, True)
    for ops, last, short, multiplier, right_shift in integer.items:
        id_lo, id_hi = lo, hi
        for op in ops:
            lo, hi = visit(op, lo, hi, True)
        a_lo, a_hi = visit(last, lo, hi, False)
        if short is not None:
            s_lo, s_hi = visit(short, id_lo, id_hi, False)
        else:
            s_lo = requant(id_lo[None, :, None, None], multiplier, right_shift).flatten()
            s_hi = requant(id_hi[None, :, None, None], multiplier, right_shift).flatten()
        lo, hi = (a_lo+s_lo).clamp(0, 127), (a_hi+s_hi).clamp(0, 127)
    return records


def float_cluster_masks(model):
    """Per conv: [clusters, Cin, kh, kw] kept positions, from the pruned float weights (pruned weights are exactly 0)."""
    out = {}
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            w = m.weight.detach()
            out[name] = (w != 0).view(w.shape[0]//8, 8, *w.shape[1:]).any(1).cpu()
    return out


def v3_ms(layers):
    total = 0
    for name, q in layers.items():
        g = GEOMETRY[name]
        tiles = g["Hout"]*((g["Wout"]+7)//8)
        total += (31+tiles*q.clamp(min=15)).sum().item()
    return total/50e6*1e3


@torch.inference_mode()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    summary, layer_rows, report = [], [], []
    for label, dataset, path, _ in MODELS:
        torch.manual_seed(0)
        calibration, validation = data(dataset, 5000)
        model = core.make_model("resnet34", 10 if dataset == "cifar10" else 100, 32).to(device).eval()
        model.load_state_dict(torch.load(Path(path)/"final_model.pt", map_location=device, weights_only=False))
        masks = float_cluster_masks(model)
        equalize(model, calibration, device, 8.)
        scale, logit = calibrate(model, calibration, device, .5)
        integer = IntegerResNet(model, scale, logit, device)
        integer.correct_bias(model, next(iter(calibration))[0][:256])
        name_layers(model, integer)
        reference = torch.cat([integer(x).cpu() for x, _ in validation])
        integer.stats["peak"] = 0
        observed_before = None
        mean, std = (CIFAR10 if dataset == "cifar10" else DATA["cifar100"]["norm"])
        records = analyse(model, integer, mean, std, device)
        integer.stats["peak"] = 0
        rewritten = torch.cat([integer(x).cpu() for x, _ in validation])
        identical = bool(torch.equal(reference, rewritten))
        peak_after = integer.stats["peak"]

        base, after_a, after_ab = {}, {}, {}
        for rec in records:
            key = rec["name"] if rec["name"] in GEOMETRY else rec["name"]+".0"
            m = masks[key]
            q = m.flatten(1).sum(1)
            all_const = rec["constant"].view(-1, 8).all(1)
            keep_b = m & ~rec["zero_inputs"][None, :, None, None]
            base[key] = q
            after_a[key] = torch.where(all_const, torch.zeros_like(q), q)
            after_ab[key] = torch.where(all_const, torch.zeros_like(q), keep_b.flatten(1).sum(1))
            live = rec["live"]
            layer_rows.append(dict(model=label, layer=key, channels=len(live), provably_constant=int(rec["constant"].sum()),
                                   zero_input_channels=int(rec["zero_inputs"].sum()), bias_only_clusters=int(all_const.sum()),
                                   positions=int(q.sum()), positions_dropped=int(q.sum()-after_ab[key].sum()),
                                   raw_shift_max_live=int(rec["shift_before"][live].max()) if live.any() else "",
                                   acc_bias_bound_bits=rec["acc_bound"]))
        live_shift = torch.cat([r["shift_before"][r["live"]] for r in records])
        all_shift = torch.cat([r["shift_before"] for r in records])
        live_bias = torch.cat([r["op"].bias.flatten().cpu()[r["live"]] for r in records])
        ms0, ms_a, ms_ab = v3_ms(base), v3_ms(after_a), v3_ms(after_ab)
        row = dict(model=label, channels=sum(len(r["live"]) for r in records),
                   provably_constant=sum(int(r["constant"].sum()) for r in records),
                   val_logits_identical=identical,
                   raw_shift_max_all=int(all_shift.max()), raw_shift_max_live=int(live_shift.max()),
                   live_channels_shift_over_36=int((live_shift > 36).sum()), live_channels_shift_over_32=int((live_shift > 32).sum()),
                   bias_bits_live=bits(int(live_bias.min()), int(live_bias.max())),
                   acc_bias_observed_bits=bits(-peak_after, peak_after),
                   acc_bias_bound_bits_max=max(r["acc_bound"] for r in records),
                   layers_bound_over_28bit=sum(r["acc_bound"] > 28 for r in records),
                   bias_only_clusters=sum(int(r["constant"].view(-1, 8).all(1).sum()) for r in records),
                   ms_v3=round(ms0, 3), ms_after_bias_only=round(ms_a, 3), ms_after_both=round(ms_ab, 3),
                   saving_pct=round(100*(1-ms_ab/ms0), 2))
        summary.append(row)
        report.append(json.dumps(row))
        print(row, flush=True)
    for name, rows in (("summary.csv", summary), ("layers.csv", layer_rows)):
        with (OUT/name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (OUT/"report.txt").write_text("\n".join(report)+"\n")


if __name__ == "__main__":
    main()
