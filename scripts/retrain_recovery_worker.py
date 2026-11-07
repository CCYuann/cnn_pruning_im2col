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
ARMS = ["dense", "unstructured", "ft_const", "rewind", "rewind_r90", "rewind_B", "rewind_kd", "gradual", "group", "group_gradual",
        "dpf", "dpf_group"]
PENALIZED = {"group", "group_gradual", "dpf_group"}
SOFT = {"dpf", "dpf_group"}


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


def load_model(name, seed, device):
    if name == "resnet50":
        return base.load_model(seed, device)
    payload = torch.load(common.checkpoint(name, seed), map_location="cpu", weights_only=False)
    if payload["model"] != name or payload["seed"] != seed or payload["split_metadata"]["split_id"] != "42fbb7440cc27c71":
        raise ValueError("Wrong Dense checkpoint")
    model = core.make_model(name, 10, 32).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    specs = core.collect_conv_specs(model, 32)
    if any(s.module.groups != 1 for s in specs):
        raise ValueError("Standard convolutions only")
    return model, specs, payload


def split_keep(specs, keep):
    masks, offset = {}, 0
    for spec in specs:
        view = cluster_view(spec.module.weight)
        size = view.shape[0]*view.shape[2]
        masks[spec.name] = keep[offset:offset+size].view(view.shape[0], -1).cpu()
        offset += size
    return masks


@torch.no_grad()
def lamp_masks(specs, ratio):
    """LAMP (Lee et al., ICLR 2021) over shared positions; a position is 8 weights."""
    scores = []
    for spec in specs:
        value = cluster_view(spec.module.weight).square().sum(1).flatten().double()
        order = torch.argsort(value)
        ordered = value[order]
        tail = ordered.flip(0).cumsum(0).flip(0)
        score = torch.zeros_like(value)
        score[order] = ordered/tail.clamp_min(1e-300)
        scores.append(score)
    merged = torch.cat(scores)
    keep = torch.ones(len(merged), dtype=torch.bool, device=merged.device)
    keep[torch.argsort(merged)[:round(ratio*len(merged))]] = False
    return split_keep(specs, keep)


@torch.no_grad()
def mac_masks(specs, ratio, beta):
    """Global-cluster quota from |w|/(output pixels)^beta; beta=0 is the frozen allocator."""
    scaled = []
    for spec in specs:
        h, w = core.output_hw(spec.input_hw, spec.module)
        scaled.append(spec.module.weight.abs().flatten()/(h*w)**beta)
    merged = torch.cat(scaled)
    keep = torch.ones(len(merged), dtype=torch.bool, device=merged.device)
    keep[torch.argsort(merged)[:round(ratio*len(merged))]] = False
    masks, offset = {}, 0
    for spec in specs:
        weight = spec.module.weight
        kept = keep[offset:offset+weight.numel()].view_as(weight)
        offset += weight.numel()
        parts = []
        for cluster in range(weight.shape[0]//8):
            group = weight[cluster*8:cluster*8+8]
            part = kept[cluster*8:cluster*8+8]
            removed = round((1-float(part.sum().item())/part.numel())*group[0].numel())
            score = core.compute_importance(group, "l1_sum").flatten()
            mask = torch.ones(len(score), dtype=torch.bool, device=score.device)
            mask[torch.argsort(score)[:removed]] = False
            parts.append(mask.cpu())
        masks[spec.name] = torch.stack(parts)
    return masks


@torch.no_grad()
def protected_masks(specs, masks, floor=False, stem=False, floor_n=1, free=False):
    """Protect positions under a fixed budget, which comes from the globally weakest unprotected kept positions.

    floor_n sets the positions each cluster keeps. free adds the protected positions on top of the budget.
    """
    if stem and not any(s.name == "conv1" for s in specs):
        raise ValueError("Stem layer conv1 not found")
    out, scores, added = {}, [], 0
    for spec in specs:
        score = cluster_view(spec.module.weight).abs().sum(1).flatten(1).cpu()
        mask = masks[spec.name].bool().clone()
        before, safe = int(mask.sum()), torch.zeros_like(mask)
        if floor and floor_n > 1:
            rank = score+(score.max()+1)*mask
            top = rank.topk(min(floor_n, mask.shape[1]), 1).indices
            mask.scatter_(1, top, True)
            safe.scatter_(1, top, True)
        elif floor:
            live = mask.any(1)
            best = torch.where(live, score.masked_fill(~mask, -1).argmax(1), score.argmax(1))
            mask[torch.arange(len(mask)), best] = True
            safe[torch.arange(len(mask)), best] = True
        if stem and spec.name == "conv1":
            mask[:], safe[:] = True, True
        added += int(mask.sum())-before
        scores.append(score.masked_fill(~mask | safe, math.inf).flatten())
        out[spec.name] = mask
    drop = torch.argsort(torch.cat(scores))[:0 if free else added]
    keep = torch.ones(sum(len(s) for s in scores), dtype=torch.bool)
    keep[drop] = False
    offset = 0
    for spec in specs:
        size = out[spec.name].numel()
        out[spec.name] = out[spec.name] & keep[offset:offset+size].view_as(out[spec.name])
        offset += size
    return out


def build_masks(specs, ratio, allocator, beta=.5, floor=False, stem=False, floor_n=1, free=False):
    if allocator == "lamp":
        masks = lamp_masks(specs, ratio)
    elif allocator == "mac":
        masks = mac_masks(specs, ratio, beta)
    else:
        masks = ratio_worker.make_masks(specs, ratio)
    return protected_masks(specs, masks, floor, stem, floor_n, free) if floor or stem else masks


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
    model, specs, payload = load_model(args.model, args.seed, device)
    teacher = None
    if args.arm == "rewind_kd" or args.kd:
        teacher, _, _ = load_model(args.model, args.seed, device)
        teacher.requires_grad_(False)
    calibration = PROJECT/"runs/compensation_r0p95/calibration"/f"seed{args.seed}"
    full = None
    if args.arm == "unstructured":
        full = unstructured_masks(specs, args.ratio, device)
    elif args.arm == "rewind_r90":
        full = expand(specs, torch.load(base.ROOT/"calibration"/f"seed{args.seed}/masks.pt", map_location="cpu", weights_only=False), device)
    elif args.arm in {"ft_const", "rewind", "rewind_kd", "rewind_B"}:
        if args.model == "resnet50" and args.ratio == .95 and args.allocator == "cluster" and (calibration/"masks.pt").exists():
            full = expand(specs, torch.load(calibration/"masks.pt", map_location="cpu", weights_only=False), device)
        else:
            full = expand(specs, build_masks(specs, args.ratio, args.allocator, args.mac_beta, args.q_floor, args.stem_keep, args.q_floor_n, args.free_fill), device)
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
    end = round(args.epochs*args.ramp_frac)
    history, started = [], time.monotonic()
    for epoch in range(args.epochs):
        lr = lr_at(args.arm, epoch, args.epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr
        soft = args.arm in SOFT and epoch < end
        if soft and epoch % 2 == 0:
            full = expand(specs, build_masks(specs, gradual_target(epoch, end, args.ratio), args.allocator, args.mac_beta, args.q_floor, args.stem_keep, args.q_floor_n, args.free_fill), device)
        if args.arm in SOFT and epoch == end:
            full = expand(specs, build_masks(specs, args.ratio, args.allocator, args.mac_beta, args.q_floor, args.stem_keep, args.q_floor_n, args.free_fill), device)
            apply(specs, full)
        if args.arm in {"gradual", "group_gradual"} and epoch < end and epoch % 2 == 0:
            previous = full
            full = expand(specs, build_masks(specs, gradual_target(epoch, end, args.ratio), args.allocator, args.mac_beta, args.q_floor, args.stem_keep, args.q_floor_n, args.free_fill), device)
            if previous is not None:
                if args.arm == "group_gradual":
                    # Prox creates exact-zero ties, so the global rank can under-prune a cluster.
                    full = {n: full[n] & previous[n] for n in full}
                elif any((full[n] & ~previous[n]).any() for n in full):
                    raise ValueError("Gradual mask regrew a pruned position")
            apply(specs, full)
        if args.arm == "group" and epoch == end:
            full = expand(specs, build_masks(specs, args.ratio, args.allocator, args.mac_beta, args.q_floor, args.stem_keep, args.q_floor_n, args.free_fill), device)
            apply(specs, full)
        model.train()
        loss_sum, correct, seen = 0., 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if soft:
                # DPF: gradient of the masked network updates the dense weights, so positions can recover.
                saved = {s.name: s.module.weight.data.clone() for s in specs}
                apply(specs, full)
            with core.autocast_context(device, "float16"):
                logits = model(x)
                if teacher is None:
                    loss = F.cross_entropy(logits, y)
                else:
                    with torch.no_grad():
                        reference = teacher(x)
                    loss = kd_loss(logits, reference, y, alpha=args.kd_alpha)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if soft:
                for spec in specs:
                    spec.module.weight.data.copy_(saved[spec.name])
            elif full:
                apply(specs, full, grads=True)
            scaler.step(optimizer)
            scaler.update()
            if args.arm in PENALIZED and epoch < end:
                group_prox(specs, lr, args.group_lambda)
            if full and not soft:
                apply(specs, full)
            loss_sum += loss.item()*len(x)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(x)
        if soft:
            saved = {s.name: s.module.weight.data.clone() for s in specs}
            apply(specs, full)
        val_loss, val_acc = core.evaluate(model, val_loader, device, use_amp="float16")
        if soft:
            for spec in specs:
                spec.module.weight.data.copy_(saved[spec.name])
        if not math.isfinite(val_loss) or not math.isfinite(loss_sum):
            raise FloatingPointError("Non-finite loss")
        row = dict(epoch=epoch+1, lr=lr, train_loss=loss_sum/seen, train_accuracy=correct/seen,
                   validation_loss=val_loss, validation_accuracy=val_acc, seconds=time.monotonic()-started)
        history.append(row)
        print(json.dumps(row), flush=True)
    common.write_csv(directory/"training_history.csv", history)
    result = dict(seed=args.seed, arm=args.arm, model=args.model, ratio=args.ratio, allocator=args.allocator, q_floor=args.q_floor, stem_keep=args.stem_keep, q_floor_n=args.q_floor_n, free_fill=args.free_fill, kd=bool(teacher), kd_alpha=args.kd_alpha if teacher else None, ramp_frac=args.ramp_frac,
                  mac_beta=args.mac_beta if args.allocator == "mac" else None, group_lambda=args.group_lambda if args.arm in PENALIZED else None,
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
            result["zero_layers"] = [s.name for s in specs if not full[s.name].any()]
        if args.arm != "unstructured":
            torch.save({s.name: full[s.name][::8].flatten(1).cpu() for s in specs}, directory/"masks.pt")
    if not args.discard_model:
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
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--allocator", choices=["cluster", "lamp", "mac"], default="cluster")
    parser.add_argument("--mac-beta", type=float, default=.5)
    parser.add_argument("--q-floor", action="store_true", help="Keep at least one position in every cluster")
    parser.add_argument("--stem-keep", action="store_true", help="Keep every conv1 position under the same budget")
    parser.add_argument("--q-floor-n", type=int, default=1, help="Positions each cluster keeps under --q-floor")
    parser.add_argument("--free-fill", action="store_true", help="Add protected positions on top of the budget")
    parser.add_argument("--ramp-frac", type=float, default=2/3, help="Fraction of epochs for the soft or gradual phase")
    parser.add_argument("--kd", action="store_true", help="Dense teacher with T=4 on top of any arm")
    parser.add_argument("--kd-alpha", type=float, default=.9)
    parser.add_argument("--discard-model", action="store_true", help="Development runs keep masks and history only, to save disk")
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
