#!/usr/bin/env python3
"""Two adaptation epochs, fixed-quota shared-gate proposals, eight remaining epochs."""

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import run_fair_criterion_r0p9 as common
import run_shared_mask_repair as previous
from shared_gate_gradient import gate_derivative, matched_random, propose
from task_loss_mask_search import assert_quotas, full_mask, measure

core = previous.core
PROJECT = common.PROJECT
ROOT = PROJECT / "runs/adapted_gradient_mask_study"
VARIANTS = ["continuous_control", "gradient_all", "gradient_gate", "matched_random"]


def event(directory, kind, **details):
    value = dict(time=common.stamp(), kind=kind, **details)
    with (directory / "journal.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    print(json.dumps(value, ensure_ascii=False), flush=True)


def rng_state(loader):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                loader=loader.generator.get_state())


def restore_rng(saved, loader):
    random.setstate(saved["python"])
    np.random.set_state(saved["numpy"])
    torch.set_rng_state(saved["torch"].cpu())
    if saved["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda"]])
    loader.generator.set_state(saved["loader"].cpu())


def calibration(directory, device):
    loaders, chosen = previous.calibration_loaders(type("Config", (), dict(images=512, calibration_seed=20261104))())
    dataset = loaders[0].dataset.dataset
    split = torch.randperm(50000, generator=torch.Generator().manual_seed(0))
    used = set(chosen)
    available = [i for i in split[5000:].tolist() if i not in used]
    order = torch.randperm(len(available), generator=torch.Generator().manual_seed(20261105))[:1024].tolist()
    indices = dict(search=chosen[:512], check=chosen[512:], audit=[available[i] for i in order])
    sets = [set(indices[name]) for name in indices]
    validation = set(split[:5000].tolist())
    if len(set.union(*sets)) != 2048 or any(s & validation for s in sets):
        raise ValueError("Calibration split overlap")
    common.write_json(directory / "data_indices.json", dict(**indices, split_id="42fbb7440cc27c71", test_used=False))
    data = {}
    for name, values in indices.items():
        samples = [dataset[i] for i in values]
        data[name] = (torch.stack([s[0] for s in samples]).to(device), torch.tensor([s[1] for s in samples], device=device))
    return data


def build_stats(specs, masks):
    stats = []
    for spec in specs:
        conv = spec.module
        saved = masks[spec.name]
        full = full_mask(conv, saved).to(conv.weight.device)
        with torch.no_grad():
            conv.weight.mul_(full)
        shaped = [m.reshape(conv.weight.shape[1:]) for m in saved]
        hw = core.output_hw(spec.input_hw, conv)
        count = conv.weight.numel()
        stat = core.build_stats(spec.name, "adapted_gradient", spec, shaped, count, int(full.sum()),
                                count*hw[0]*hw[1], core.macs_from_masks(shaped, conv.out_channels, hw, 8), 8, 8)
        stat.full_mask = full
        stats.append(stat)
    return stats


def update_masks(model, specs, stats, initial, selected, shadow, optimizer):
    assert_quotas(initial, selected)
    for spec, stat in zip(specs, stats):
        conv = spec.module
        old = full_mask(conv, initial[spec.name]).to(conv.weight.device)
        new = full_mask(conv, selected[spec.name]).to(conv.weight.device)
        with torch.no_grad():
            conv.weight.copy_(torch.where(old, conv.weight, shadow[spec.name]) * new)
            momentum = optimizer.state.get(conv.weight, {}).get("momentum_buffer")
            if momentum is not None:
                momentum.mul_(old & new)
        stat.masks = [m.reshape(conv.weight.shape[1:]) for m in selected[spec.name]]
        stat.full_mask = new
    core.apply_masks_(model, stats, 8)


def search(model, specs, masks, shadow, directory, device, seed):
    started = time.monotonic()
    data = calibration(directory, device)
    model.eval()
    # No forward mask multiplication: dormant zero-valued parameters retain dL/dW.
    model.zero_grad(set_to_none=True)
    images, labels = data["search"]
    for x, y in zip(images.split(64), labels.split(64)):
        loss = F.cross_entropy(model(x), y, reduction="sum") / len(images)
        loss.backward()
    proposals, swap_rows = {}, []
    for spec in specs:
        conv = spec.module
        derivative = gate_derivative(conv.weight.detach(), conv.weight.grad, shadow[spec.name], masks[spec.name])
        candidate, swaps = propose(derivative, masks[spec.name])
        proposals[spec.name] = candidate.cpu()
        swap_rows.extend(dict(layer=spec.name, **row) for row in swaps)
    model.zero_grad(set_to_none=True)
    original_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    before = {name: measure(model, value) for name, value in data.items()}
    current = {name: before[name] for name in ["search", "check"]}
    gated = copy.deepcopy(masks)
    decisions = []
    for spec in specs:
        name, conv = spec.name, spec.module
        old = full_mask(conv, masks[name]).to(device)
        candidate = full_mask(conv, proposals[name]).to(device)
        row = dict(layer=name, candidate_swaps=int((masks[name] & ~proposals[name]).sum()), accepted=False,
                   before_search_ce=current["search"]["loss"], before_check_ce=current["check"]["loss"])
        if row["candidate_swaps"]:
            with torch.no_grad():
                conv.weight.copy_(torch.where(old, original_state[name+".weight"], shadow[name]) * candidate)
            first = measure(model, data["search"])
            second = measure(model, data["check"])
            row.update(candidate_search_ce=first["loss"], candidate_check_ce=second["loss"])
            if first["loss"] < current["search"]["loss"]-1e-5 and second["loss"] < current["check"]["loss"]-1e-5:
                gated[name] = proposals[name].clone()
                current = dict(search=first, check=second)
                row["accepted"] = True
            else:
                with torch.no_grad():
                    conv.weight.copy_(original_state[name+".weight"])
        row.update(after_search_ce=current["search"]["loss"], after_check_ce=current["check"]["loss"])
        decisions.append(row)
        event(directory, "layer_search", **row)
    after = {name: measure(model, value) for name, value in data.items()}
    random_masks = {spec.name: matched_random(masks[spec.name], gated[spec.name], seed+20261104+i)
                    for i, spec in enumerate(specs)}
    variants = dict(continuous_control=masks, gradient_all=proposals, gradient_gate=gated, matched_random=random_masks)
    for selected in variants.values():
        assert_quotas(masks, selected)
    model.load_state_dict(original_state)
    if swap_rows:
        common.write_csv(directory / "gradient_swaps.csv", swap_rows)
    torch.save(variants, directory / "mask_plans.pt")
    report = dict(before=before, after_gated=after, seconds=time.monotonic()-started, decisions=decisions,
                  proposed_swaps=len(swap_rows), gated_swaps=sum(int((masks[n] & ~gated[n]).sum()) for n in masks),
                  accepted_layers=sum(r["accepted"] for r in decisions), test_evaluated=False)
    common.write_json(directory / "search.json", report)
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    directory = args.root.resolve() / f"seed{args.seed}_{args.variant}"
    directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    core.reset_training_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = PROJECT / "runs/spatial_quota_baseline_study" / f"resnet50_seed{args.seed}_l1_sum_flat/masks.pt"
    config = dict(seed=args.seed, variant=args.variant, checkpoint_sha256=common.sha(common.checkpoint("resnet50", args.seed)),
                  initial_masks_sha256=common.sha(source), core_sha256=common.sha(common.CORE),
                  worker_sha256=common.sha(Path(__file__)), helper_sha256=common.sha(Path(__file__).with_name("shared_gate_gradient.py")),
                  evaluator_sha256=common.sha(Path(__file__).with_name("task_loss_mask_search.py")),
                  tm=8, total_epochs=10, adaptation_epochs=2, max_swaps_per_cluster=1, search_images=512, check_images=512)
    manifest = directory / "configuration.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError("Configuration changed; use another root")
    common.write_json(manifest, config)
    payload = torch.load(common.checkpoint("resnet50", args.seed), map_location="cpu", weights_only=False)
    if payload["split_metadata"]["split_id"] != "42fbb7440cc27c71" or payload["model"] != "resnet50" or payload["seed"] != args.seed:
        raise ValueError("Wrong Dense checkpoint")
    model = core.make_model("resnet50", 10, 32).to(device)
    model.load_state_dict(payload["model_state_dict"])
    del payload
    model.eval()
    specs = core.collect_conv_specs(model, 32)
    shadow = {s.name: s.module.weight.detach().clone() for s in specs}
    initial = torch.load(source, map_location="cpu", weights_only=False)["l1_sum"]
    masks = copy.deepcopy(initial)
    stats = build_stats(specs, masks)
    (train_loader, val_loader, _unused_test_loader), _, metadata = core.build_cifar10(PROJECT/"data", 128, 32, None, None, None, 2, True, 0, args.seed, 5000)
    _, direct = core.evaluate(model, val_loader, device, use_amp="float16")
    optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9, weight_decay=.0005)
    scaler = core.make_scaler(device, "float16")
    core.reset_training_rng(args.seed+100000, train_loader)
    epoch, searched, history = 0, False, []
    resume = directory / "training_state.pt"
    if resume.exists():
        saved = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        masks = {n: m.cpu() for n, m in saved["masks"].items()}
        for spec, stat in zip(specs, stats):
            stat.masks = [m.reshape(spec.module.weight.shape[1:]) for m in masks[spec.name]]
            stat.full_mask = full_mask(spec.module, masks[spec.name]).to(device)
        restore_rng(saved["rng"], train_loader)
        epoch, searched, history = saved["epoch"], saved["searched"], saved["history"]
        event(directory, "resume_training", epoch=epoch, searched=searched)
    def save():
        value = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), masks=masks,
                     rng=rng_state(train_loader), epoch=epoch, searched=searched, history=history)
        torch.save(value, directory/"training_state.tmp")
        (directory/"training_state.tmp").replace(resume)
    while epoch < 10:
        if epoch == 2 and not searched:
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                            rng=rng_state(train_loader), epoch=epoch), directory/"post_adaptation.pt")
            if args.variant != "continuous_control":
                rng = rng_state(train_loader)
                plans = search(model, specs, initial, shadow, directory, device, args.seed)
                masks = plans[args.variant]
                update_masks(model, specs, stats, initial, masks, shadow, optimizer)
                restore_rng(rng, train_loader)
            searched = True
            torch.save(dict(initial=initial, selected=masks), directory/"masks.pt")
            save()
        train_loss, train_acc = core.train_one_epoch(model, train_loader, optimizer, device, str(epoch+1),
                                                     prune_stats=stats, tm=8, scaler=scaler, use_amp="float16")
        val_loss, val_acc = core.evaluate(model, val_loader, device, use_amp="float16")
        epoch += 1
        row = dict(epoch=epoch, train_loss=train_loss, train_accuracy=train_acc, validation_loss=val_loss, validation_accuracy=val_acc)
        history.append(row)
        event(directory, "epoch_complete", **row)
        save()
    assert_quotas(initial, masks)
    common.write_csv(directory/"training_history.csv", history)
    common.write_json(directory/"result.json", dict(model="resnet50", seed=args.seed, variant=args.variant,
                      direct_validation_acc=direct, final_validation_acc=history[-1]["validation_accuracy"],
                      validation_loss=history[-1]["validation_loss"], total_epochs=10, test_evaluated=False,
                      sparsity=core.summarize(stats)["pruning_ratio"], split_id=metadata["split_id"]))
    torch.save(dict(model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}, masks=masks,
                    variant=args.variant, test_evaluated=False), directory/"final_model.pt")
    event(directory, "worker_complete", variant=args.variant, validation_accuracy=history[-1]["validation_accuracy"])


if __name__ == "__main__":
    main()
