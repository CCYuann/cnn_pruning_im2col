#!/usr/bin/env python3
"""Q15 integer simulation of ResNet34/50 under the accelerator's frozen numeric spec.

Spec (mask_aware_conv_accelerator/docs/QUANTIZATION_V*.md): symmetric [-127,127], zero-point 0,
ties-away rounding, Conv+BN folded, per-output-channel weight scale, per-tensor activation scale from
calibration maxima, wide residual add before ReLU clamp, integer pooling and INT8 classifier.
Calibrates on 2048 training images outside validation; reports the 5000-image validation split.
"""
import argparse, csv, json, sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import resnet34_pruning_experiments_path  # noqa: F401
import resnet34_pruning_experiments as core
from int8_eval_pruned import accuracy, loaders

HW = Path(__file__).resolve().parents[2]/"mask_aware_conv_accelerator/scripts"
sys.path.insert(0, str(HW))
from evaluate_bitexact_resnet34_backbone import IntegerHead  # noqa: E402
from quantization import quantize_multiplier  # noqa: E402

FRACTION = 15
BIAS_BITS = 28


def ties_away(value):
    return torch.where(value >= 0, torch.floor(value+.5), torch.ceil(value-.5))


def requant(value, multiplier, right_shift):
    product = value*multiplier
    left = (-right_shift).clamp_min(0)
    right = right_shift.clamp_min(0)
    half = torch.where(right > 0, torch.ones_like(right) << (right-1).clamp_min(0), torch.zeros_like(right))
    rounded = ((product.abs()+half) >> right) << left
    return torch.where(product < 0, -rounded, rounded)


class IntConv:
    def __init__(self, conv, bn, s_in, s_out, device, stats):
        gain = bn.weight.detach().double()/(bn.running_var.detach().double()+bn.eps).sqrt()
        folded = conv.weight.detach().double()*gain[:, None, None, None]
        bias = bn.bias.detach().double()-gain*bn.running_mean.detach().double()
        maxima = folded.flatten(1).abs().amax(1)
        scale = torch.where(maxima > 0, maxima/127, torch.ones_like(maxima))
        # Near-dead channels (BN gain ~ 0) keep a real bias; raise the scale so bias fits INT32.
        scale = torch.maximum(scale, bias.abs()/(s_in*2**BIAS_BITS))
        self.weight = ties_away(folded/scale[:, None, None, None]).clamp(-127, 127).to(device)
        self.bias = ties_away(bias/(s_in*scale)).long().to(device)[None, :, None, None]
        params = [quantize_multiplier(s_in*float(x)/s_out, FRACTION) for x in scale.tolist()]
        self.multiplier = torch.tensor([p.multiplier for p in params], device=device)[None, :, None, None]
        # RTL accepts shifts up to 48; |acc*multiplier| < 2^46 rounds to 0 for any shift >= 48, so the clamp is bit-exact.
        self.right_shift = torch.tensor([min(FRACTION-p.shift, 48) for p in params], device=device)[None, :, None, None]
        self.stride, self.padding, self.stats = conv.stride, conv.padding, stats
        self.unit, self.s_out = (s_in*scale).to(device)[None, :, None, None], s_out

    def correct(self, x, target):
        """Move the mean pre-ReLU offset into the bias, on data that already carries upstream correction."""
        drift = (target-self(x).double()*self.s_out).mean((0, 2, 3), keepdim=True)
        self.bias = self.bias+ties_away(drift/self.unit).long()

    def __call__(self, x):
        acc = F.conv2d(x.double(), self.weight, stride=self.stride, padding=self.padding).round().long()+self.bias
        self.stats["peak"] = max(self.stats["peak"], int(acc.abs().amax()))
        return requant(acc, self.multiplier, self.right_shift)


def blocks(model):
    for number in range(1, 5):
        for index, block in enumerate(getattr(model, f"layer{number}")):
            yield f"layer{number}.{index}", block


def branch_convs(block):
    if hasattr(block, "conv3"):
        return [(block.conv1, block.bn1), (block.conv2, block.bn2), (block.conv3, block.bn3)]
    return [(block.conv1, block.bn1), (block.conv2, block.bn2)]


@torch.inference_mode()
def calibrate(model, loader, device, clip=1.):
    peak, samples = {}, {}

    def track(name, value):
        level = float(value.abs().amax())
        peak[name] = max(peak.get(name, 0.), level)
        if clip < 1 and name != "input":
            sample = value.flatten()
            samples.setdefault(name, []).append(sample[torch.randperm(len(sample), device=sample.device)[:50000]].cpu())
        return value
    logit = 0.
    for images, _ in loader:
        x = track("input", images.to(device))
        x = track("stem", F.relu(model.bn1(model.conv1(x))))
        for prefix, block in blocks(model):
            identity = x
            convs = branch_convs(block)
            for i, (conv, bn) in enumerate(convs[:-1]):
                x = track(f"{prefix}.a{i}", F.relu(bn(conv(x))))
            x = convs[-1][1](convs[-1][0](x))
            if block.downsample is not None:
                identity = block.downsample(identity)
            x = track(f"{prefix}.out", F.relu(x+identity))
        logit = max(logit, float(model.fc(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)).abs().amax()))
    if clip < 1:  # clip<1 selects the per-tensor scale with least quantization MSE on calibration data
        for name, parts in samples.items():
            value = torch.cat(parts).to(device).double()
            candidates = [peak[name]*f for f in torch.linspace(.2, 1, 41).tolist()]
            error = [float((ties_away(value/(c/127)).clamp(-127, 127)*(c/127)-value).square().mean()) for c in candidates]
            peak[name] = candidates[min(range(len(error)), key=error.__getitem__)]
    return {k: v/127 for k, v in peak.items()}, logit/127


@torch.inference_mode()
def equalize(model, loader, device, limit):
    """Cross-layer equalization of branch-internal activations; the float function and every zero weight stay put.

    Channel c of a branch activation is multiplied by k[c] (BN gain and bias), and input column c of the next
    conv is divided by k[c]. Only branch-internal tensors qualify: nothing else reads them, residual tensors do.
    """
    peak = {}
    for images, _ in loader:
        x = F.relu(model.bn1(model.conv1(images.to(device))))
        for prefix, block in blocks(model):
            identity, convs = x, branch_convs(block)
            for i, (conv, bn) in enumerate(convs[:-1]):
                x = F.relu(bn(conv(x)))
                level = x.transpose(0, 1).flatten(1).amax(1)
                key = (prefix, i)
                peak[key] = torch.maximum(peak[key], level) if key in peak else level
            x = convs[-1][1](convs[-1][0](x))
            if block.downsample is not None:
                identity = block.downsample(identity)
            x = F.relu(x+identity)
    named = dict(blocks(model))
    for (prefix, i), level in peak.items():
        convs = branch_convs(named[prefix])
        top = level.amax()
        gain = torch.where(level > 0, (top/level).clamp_min(1), torch.ones_like(level)).clamp_max(limit)
        bn, following = convs[i][1], convs[i+1][0]
        bn.weight.mul_(gain)
        bn.bias.mul_(gain)
        following.weight.div_(gain[None, :, None, None])


class IntegerResNet:
    def __init__(self, model, scale, logit_scale, device):
        self.device, self.scale, self.stats = device, scale, dict(peak=0)
        self.stem = IntConv(model.conv1, model.bn1, scale["input"], scale["stem"], device, self.stats)
        self.items, previous = [], "stem"
        for prefix, block in blocks(model):
            convs = branch_convs(block)
            ops, s_in = [], scale[previous]
            for i, (conv, bn) in enumerate(convs[:-1]):
                ops.append(IntConv(conv, bn, s_in, scale[f"{prefix}.a{i}"], device, self.stats))
                s_in = scale[f"{prefix}.a{i}"]
            s_out = scale[f"{prefix}.out"]
            last = IntConv(*convs[-1], s_in, s_out, device, self.stats)
            short = None
            if block.downsample is not None:
                short = IntConv(block.downsample[0], block.downsample[1], scale[previous], s_out, device, self.stats)
            p = quantize_multiplier(scale[previous]/s_out, FRACTION)
            self.items.append((ops, last, short, torch.tensor(p.multiplier, device=device), torch.tensor(FRACTION-p.shift, device=device)))
            previous = f"{prefix}.out"
        self.head = IntegerHead(model.fc, scale[previous], logit_scale, FRACTION, device)

    @torch.inference_mode()
    def correct_bias(self, model, images):
        """Sequential per-channel bias correction (Nagel et al., ICCV 2019) using the float network as reference."""
        fx = images.to(self.device)
        ix = ties_away(fx.double()/self.scale["input"]).clamp(-127, 127).long()
        target = model.bn1(model.conv1(fx)).double()
        self.stem.correct(ix, target)
        ix, fx = self.stem(ix).clamp(0, 127), F.relu(target.float())
        for (_, block), (ops, last, short, multiplier, right_shift) in zip(blocks(model), self.items):
            identity_f, identity_i, convs = fx, ix, branch_convs(block)
            for (conv, bn), op in zip(convs[:-1], ops):
                target = bn(conv(fx)).double()
                op.correct(ix, target)
                ix, fx = op(ix).clamp(0, 127), F.relu(target.float())
            target = convs[-1][1](convs[-1][0](fx)).double()
            last.correct(ix, target)
            if short is not None:
                reference = block.downsample(identity_f)
                short.correct(identity_i, reference.double())
                side = short(identity_i)
            else:
                reference, side = identity_f, requant(identity_i, multiplier, right_shift)
            ix, fx = (last(ix)+side).clamp(0, 127), F.relu(target.float()+reference)

    def __call__(self, images):
        x = ties_away(images.to(self.device).double()/self.scale["input"]).clamp(-127, 127).long()
        x = self.stem(x).clamp(0, 127)
        for ops, last, short, multiplier, right_shift in self.items:
            identity = x
            for op in ops:
                x = op(x).clamp(0, 127)
            wide = last(x)+(short(identity) if short is not None else requant(identity, multiplier, right_shift))
            x = wide.clamp(0, 127)
        return self.head(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", help="label=model_name=path=expected_validation")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clip", type=float, default=1., help="1 uses calibration maxima; below 1 uses the least-MSE per-tensor scale")
    parser.add_argument("--bias-correct", action="store_true", help="Sequential bias correction on calibration data")
    parser.add_argument("--equalize", type=float, default=0., help="Largest channel gain for cross-layer equalization; 0 disables it")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(6)
    calibration, validation = loaders("data")
    done = list(csv.DictReader(args.out.open())) if args.out.exists() else []
    finished = {r["label"] for r in done}
    for item in args.models:
        label, name, path, expected = item.split("=")
        if label in finished:
            continue
        model = core.make_model(name, 10, 32).to(device).eval()
        state = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(state.get("model_state_dict", state))
        if args.equalize:
            equalize(model, calibration, device, args.equalize)
        scale, logit = calibrate(model, calibration, device, args.clip)
        with torch.inference_mode():
            fp32 = sum(int((model(x.to(device)).argmax(1).cpu() == y).sum()) for x, y in validation)/5000
            if abs(fp32-float(expected)) > 3e-3:
                raise ValueError(f"{label}: FP32 {fp32} disagrees with {expected}")
            integer = IntegerResNet(model, scale, logit, device)
            if args.bias_correct:
                integer.correct_bias(model, next(iter(calibration))[0][:256])
            q15 = sum(int((integer(x).argmax(1).cpu() == y).sum()) for x, y in validation)/5000
        peak = integer.stats["peak"]
        done.append(dict(label=label, fp32=fp32, q15=q15, loss_pp=(fp32-q15)*100, peak_acc=peak, int32_overflow=peak >= 2**31))
        with args.out.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(done[0]))
            writer.writeheader()
            writer.writerows(done)
        print(done[-1], flush=True)


if __name__ == "__main__":
    main()
