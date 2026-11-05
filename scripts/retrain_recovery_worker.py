#!/usr/bin/env python3
"""Compare training-side recovery methods for shared-mask pruning at equal budget."""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import compensation_ratio_worker as ratio_worker
import weight_compensation_worker as base

common, core = base.common, base.core
PROJECT = common.PROJECT
ARMS = ["dense", "unstructured", "ft_const", "rewind", "rewind_r90", "rewind_B", "rewind_kd", "gradual", "group"]
CHANGING_MASK = {"gradual", "group"}


def lr_at(arm, epoch, epochs):
    if arm == "ft_const":
        return .001
    return 1e-4+(.1-1e-4)*.5*(1+math.cos(math.pi*(100-epochs+epoch)/100))


def gradual_target(epoch, end, ratio):
    """Cubic schedule reaching `ratio` at the last update before `end`."""
    return ratio*(1-(1-(epoch+2)/end)**3)


def cluster_view(weight):
    if weight.shape[0] % 8:
        raise ValueError("Group penalty expects a multiple of 8 filters")
    return weight.view(weight.shape[0]//8, 8, -1)


@torch.no_grad()
def group_prox(specs, step, lam):
    for spec in specs:
        view = cluster_view(spec.module.weight)
        norm = view.square().sum(1, keepdim=True).sqrt()
        view.mul_((1-step*lam/norm.clamp_min(1e-12)).clamp_min(0))


def kd_loss(logits, teacher, targets, temperature=4., alpha=.9):
    soft = F.kl_div(F.log_softmax(logits/temperature, 1), F.softmax(teacher/temperature, 1), reduction="batchmean")
    return alpha*temperature*temperature*soft+(1-alpha)*F.cross_entropy(logits, targets)


def expand(specs, masks, device):
    return {s.name: base.full_mask(s.module, masks[s.name]).to(device) for s in specs}


@torch.no_grad()
def apply(specs, full, grads=False):
    for spec in specs:
        (spec.module.weight.grad if grads else spec.module.weight).mul_(full[spec.name])


def unstructured_masks(specs, ratio, device):
    reference = core.global_unstructured_reference(specs, 8, ratio)
    return {s.name: reference[s.name]["full_mask"].to(s.module.weight.device).bool() for s in specs}


def cost(specs, full):
    kept, macs, total, dense_macs = 0, 0, 0, 0
    for spec in specs:
        h, w = core.output_hw(spec.input_hw, spec.module)
        count = int(full[spec.name].sum())
        kept += count
        macs += count*h*w
        total += spec.module.weight.numel()
        dense_macs += spec.module.weight.numel()*h*w
    return dict(kept=kept, sparsity=1-kept/total, macs=macs, mac_reduction=1-macs/dense_macs)


def run(args, device):
    directory = args.root/f"seed{args.seed}_{args.arm}{args.tag}"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/"result.json").exists():
        return
    model, specs, payload = base.load_model(args.seed, device)
    teacher = None
    if args.arm == "rewind_kd":
        teacher, _, _ = base.load_model(args.seed, device)
        teacher.requires_grad_(False)
    calibration = PROJECT/"runs/compensation_r0p95/calibration"/f"seed{args.seed}"
    full = None
    if args.arm == "unstructured":
        full = unstructured_masks(specs, .95, device)
    elif args.arm == "rewind_r90":
        full = expand(specs, torch.load(base.ROOT/"calibration"/f"seed{args.seed}/masks.pt", map_location="cpu", weights_only=False), device)
    elif args.arm in {"ft_const", "rewind", "rewind_kd", "rewind_B"}:
        full = expand(specs, torch.load(calibration/"masks.pt", map_location="cpu", weights_only=False), device)
    if args.arm == "rewind_B":
        source = PROJECT/"runs/compensation_r0p95"/f"seed{args.seed}_dense_input/initialization.pt"
        model.load_state_dict(torch.load(source, map_location=device, weights_only=False)["model"])
    del payload
    if full:
        apply(specs, full)
    (train_loader, val_loader, _), _, metadata = core.build_cifar10(
        PROJECT/"data", 128, 32, args.train_limit, None, None, 2, True, 0, args.seed, 5000)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=5e-4)
    scaler = core.make_scaler(device, "float16")
    core.reset_training_rng(args.seed+100000, train_loader)
    end = round(args.epochs*2/3)
    history, started = [], time.monotonic()
    for epoch in range(args.epochs):
        lr = lr_at(args.arm, epoch, args.epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr
        if args.arm == "gradual" and epoch < end and epoch % 2 == 0:
            previous = full
            full = expand(specs, ratio_worker.make_masks(specs, gradual_target(epoch, end, args.ratio)), device)
            if previous is not None and any((full[n] & ~previous[n]).any() for n in full):
                raise ValueError("Gradual mask regrew a pruned position")
            apply(specs, full)
        if args.arm == "group" and epoch == end:
            full = expand(specs, ratio_worker.make_masks(specs, args.ratio), device)
            apply(specs, full)
        model.train()
        loss_sum, correct, seen = 0., 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with core.autocast_context(device, "float16"):
                logits = model(x)
                if teacher is None:
                    loss = F.cross_entropy(logits, y)
                else:
                    with torch.no_grad():
                        reference = teacher(x)
                    loss = kd_loss(logits, reference, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if full:
                apply(specs, full, grads=True)
            scaler.step(optimizer)
            scaler.update()
            if args.arm == "group" and epoch < end:
                group_prox(specs, lr, args.group_lambda)
            if full:
                apply(specs, full)
            loss_sum += loss.item()*len(x)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(x)
        val_loss, val_acc = core.evaluate(model, val_loader, device, use_amp="float16")
        if not math.isfinite(val_loss) or not math.isfinite(loss_sum):
            raise FloatingPointError("Non-finite loss")
        row = dict(epoch=epoch+1, lr=lr, train_loss=loss_sum/seen, train_accuracy=correct/seen,
                   validation_loss=val_loss, validation_accuracy=val_acc, seconds=time.monotonic()-started)
        history.append(row)
        print(json.dumps(row), flush=True)
    common.write_csv(directory/"training_history.csv", history)
    result = dict(seed=args.seed, arm=args.arm, group_lambda=args.group_lambda if args.arm == "group" else None,
                  epochs=args.epochs, final_validation=history[-1]["validation_accuracy"],
                  final_train=history[-1]["train_accuracy"], split_id=metadata["split_id"],
                  seconds=time.monotonic()-started, test_evaluated=False)
    if full:
        for spec in specs:
            if torch.count_nonzero(spec.module.weight[~full[spec.name].bool()]):
                raise ValueError("Forbidden weights are nonzero")
        result.update(cost(specs, full))
        if args.arm != "unstructured":
            result["q_zero"] = sum(int((full[s.name].flatten(1)[::8].sum(1) == 0).sum()) for s in specs)
        if args.arm != "unstructured":
            torch.save({s.name: full[s.name][::8].flatten(1).cpu() for s in specs}, directory/"masks.pt")
    torch.save({n: v.detach().cpu() for n, v in model.state_dict().items()}, directory/"final_model.pt")
    common.write_json(directory/"result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ratio", type=float, default=.95)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--group-lambda", type=float, default=1e-4)
    parser.add_argument("--tag", default="")
    parser.add_argument("--train-limit", type=int, default=None)
    args = parser.parse_args()
    args.root = args.root.resolve()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    core.reset_training_rng(args.seed)
    run(args, torch.device("cuda" if torch.cuda.is_available() else "cpu"))


if __name__ == "__main__":
    main()
