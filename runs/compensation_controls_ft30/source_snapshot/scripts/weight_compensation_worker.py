#!/usr/bin/env python3
"""Prepare aligned teacher patches, compensate fixed masks, and fine-tune fairly."""

import argparse
import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch

import run_fair_criterion_r0p9 as common
import run_shared_mask_repair as previous
from adapted_gradient_mask import build_stats, restore_rng, rng_state
from fixed_support_ridge import compensate, positions_for_layer, sampled_output
from shared_mask_repair import sample_patches
from task_loss_mask_search import full_mask, recalibrate_bn

core = previous.core
PROJECT = common.PROJECT
ROOT = PROJECT/"runs/fixed_mask_weight_compensation"
VARIANTS = ["original", "dense_input", "sequential"]


def event(directory, kind, **details):
    value = dict(time=common.stamp(), kind=kind, **details)
    with (directory/"journal.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False)+"\n")
    print(json.dumps(value, ensure_ascii=False), flush=True)


def cache_dir(args):
    return args.root/"calibration"/f"seed{args.seed}"


def fingerprint(args):
    return dict(seed=args.seed, model="resnet50", checkpoint_sha256=common.sha(common.checkpoint("resnet50", args.seed)),
                core_sha256=common.sha(common.CORE), worker_sha256=common.sha(Path(__file__)),
                helper_sha256=common.sha(Path(__file__).with_name("fixed_support_ridge.py")),
                sampler_sha256=common.sha(Path(__file__).with_name("shared_mask_repair.py")),
                training_helper_sha256=common.sha(Path(__file__).with_name("adapted_gradient_mask.py")),
                eval_helper_sha256=common.sha(Path(__file__).with_name("task_loss_mask_search.py")),
                images_per_split=512, max_patches=4096, calibration_seed=20261107, tm=8, target_sparsity=.9)


def freeze(path, config):
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError(f"Frozen configuration changed: {path}")
    common.write_json(path, config)


def load_model(seed, device):
    payload = torch.load(common.checkpoint("resnet50", seed), map_location="cpu", weights_only=False)
    if payload["model"] != "resnet50" or payload["seed"] != seed or payload["split_metadata"]["split_id"] != "42fbb7440cc27c71":
        raise ValueError("Wrong Dense checkpoint")
    model = core.make_model("resnet50", 10, 32).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    specs = core.collect_conv_specs(model, 32)
    if any(s.module.groups != 1 or s.module.bias is not None for s in specs):
        raise ValueError("This pilot supports the bias-free standard ResNet50 convolutions only")
    return model, specs, payload


@torch.no_grad()
def make_masks(specs, seed):
    reference = core.global_unstructured_reference(specs, 8, .9)
    masks = {}
    for spec in specs:
        parts = []
        for i, group in enumerate(reference[spec.name]["filter_groups"]):
            score = core.compute_importance(spec.module.weight[group], "l1_sum").flatten()
            removed = round(reference[spec.name]["cluster_ratios"][i]*len(score))
            mask = torch.ones(len(score), dtype=torch.bool, device=score.device)
            mask[torch.argsort(score)[:removed]] = False
            parts.append(mask.cpu())
        masks[spec.name] = torch.stack(parts)
    source = PROJECT/"runs/spatial_quota_baseline_study"/f"resnet50_seed{seed}_l1_sum_flat/masks.pt"
    if source.exists():
        old = torch.load(source, map_location="cpu", weights_only=False)["l1_sum"]
        if old.keys() != masks.keys() or any(not torch.equal(old[n], m) for n, m in masks.items()):
            raise ValueError("Mask differs from existing formal flat l1")
    return masks


@torch.no_grad()
def prepare(args, device):
    directory = cache_dir(args)
    directory.mkdir(parents=True, exist_ok=True)
    freeze(directory/"configuration.json", fingerprint(args))
    if (directory/"ready.json").exists():
        return
    model, specs, payload = load_model(args.seed, device)
    common.write_json(directory/"dense_training_config.json", payload["training_config"])
    masks = make_masks(specs, args.seed)
    torch.save(masks, directory/"masks.pt")
    loaders, indices = previous.calibration_loaders(SimpleNamespace(images=512, calibration_seed=20261107))
    dataset = loaders[0].dataset.dataset
    validation = set(torch.randperm(50000, generator=torch.Generator().manual_seed(0))[:5000].tolist())
    if len(set(indices)) != 1024 or set(indices)&validation:
        raise ValueError("Fit/check overlap or validation contamination")
    common.write_json(directory/"data_indices.json", dict(fit=indices[:512], check=indices[512:], test_used=False))
    images = {name: torch.stack([dataset[i][0] for i in values])
              for name, values in [("fit", indices[:512]), ("check", indices[512:])]} 
    torch.save(images, directory/"images.pt")
    buffers = {s.name: {} for s in specs}
    metadata = []
    for index, spec in enumerate(specs):
        hw = core.output_hw(spec.input_hw, spec.module)
        metadata.append(dict(index=index, name=spec.name, file=f"{index:03d}.pt", output_hw=list(hw),
                             kernel=list(spec.module.kernel_size), stride=list(spec.module.stride),
                             max_q=int(masks[spec.name].sum(1).max())))
    started = time.monotonic()
    for split_id, name in enumerate(["fit", "check"]):
        handles = []
        for record, spec in zip(metadata, specs):
            digest = int.from_bytes(hashlib.sha256(spec.name.encode()).digest()[:4], "little")
            positions = positions_for_layer(512, math.prod(record["output_hw"]), 4096, 20261107+digest+split_id)
            buffers[spec.name][name] = dict(x=[], y=[], positions=positions)
            def hook(module, inputs, output, layer=spec.name):
                part = buffers[layer][name]
                pos = part["positions"][offset:offset+len(inputs[0])].to(device)
                part["x"].append(sample_patches(inputs[0], module, pos).cpu().clone())
                part["y"].append(sampled_output(output, pos).cpu().clone())
            handles.append(spec.module.register_forward_hook(hook))
        try:
            for offset in range(0, 512, 32):
                model(images[name][offset:offset+32].to(device))
        finally:
            for handle in handles:
                handle.remove()
        event(directory, "teacher_split_collected", split=name)
    files = []
    for record in metadata:
        saved = dict(name=record["name"])
        for name in ["fit", "check"]:
            part = buffers[record["name"]][name]
            saved[name] = dict(x=torch.cat(part["x"], 1), y=torch.cat(part["y"], 1), positions=part["positions"])
        path = directory/record["file"]
        torch.save(saved, path)
        files.append(dict(file=record["file"], sha256=common.sha(path)))
    del buffers
    selected = list(dict.fromkeys([0,
        next(i for i,s in enumerate(specs) if s.module.kernel_size==(1,1)),
        next(i for i,s in enumerate(specs) if s.module.stride==(2,2)),
        max(range(len(metadata)), key=lambda i:metadata[i]["max_q"])]))
    benchmarks = []
    for i in selected:
        spec = specs[i]
        data = torch.load(directory/metadata[i]["file"], map_location="cpu", weights_only=False)
        torch.testing.assert_close((spec.module.weight.flatten(1) @ data["fit"]["x"][:, :32].to(device)).cpu(),
                                   data["fit"]["y"][:, :32], atol=2e-4, rtol=2e-4)
        cluster = int(masks[spec.name].sum(1).argmax())
        start, end = cluster*8, min(cluster*8+8, spec.module.out_channels)
        _, report = compensate(spec.module.weight[start:end].flatten(1), masks[spec.name][cluster].to(device),
                               data["fit"]["x"].to(device), data["fit"]["y"][start:end].to(device),
                               data["check"]["x"].to(device), data["check"]["y"][start:end].to(device))
        benchmarks.append(dict(layer=spec.name, cluster=cluster, **report))
    common.write_csv(directory/"benchmark.csv", benchmarks)
    total = sum(s.module.weight.numel() for s in specs)
    kept = sum(int(m.sum())*8 for m in masks.values())
    ready = dict(seed=args.seed, layers=metadata, cache_files=files, total_weights=total, kept_weights=kept,
                 mask_sha256=common.sha(directory/"masks.pt"), image_sha256=common.sha(directory/"images.pt"),
                 zero_clusters=sum(int((m.sum(1)==0).sum()) for m in masks.values()),
                 max_q=max(int(m.sum(1).max()) for m in masks.values()),
                 cache_bytes=sum((directory/r["file"]).stat().st_size for r in metadata),
                 peak_gpu_bytes=torch.cuda.max_memory_allocated() if device.type=="cuda" else 0,
                 seconds=time.monotonic()-started, test_evaluated=False)
    common.write_json(directory/"ready.json", ready)
    event(directory, "preparation_complete", seconds=ready["seconds"], max_q=ready["max_q"], zero_clusters=ready["zero_clusters"])


@torch.no_grad()
def student_patches(model, conv, images, positions, device):
    parts = []
    def hook(module, inputs, output):
        pos = positions[offset:offset+len(inputs[0])].to(device)
        parts.append(sample_patches(inputs[0], module, pos).cpu().clone())
    handle = conv.register_forward_hook(hook)
    model.eval()
    try:
        for offset in range(0, len(images), 32):
            model(images[offset:offset+32].to(device))
    finally:
        handle.remove()
    return torch.cat(parts, 1)


@torch.no_grad()
def initialize(model, specs, dense_state, masks, args, directory, device):
    cache = cache_dir(args)
    ready = json.loads((cache/"ready.json").read_text())
    conv_keys = {s.name+".weight" for s in specs}
    nonconv = {n:v.detach().cpu().clone() for n,v in model.state_dict().items() if n not in conv_keys}
    for spec in specs:
        spec.module.weight.mul_(full_mask(spec.module, masks[spec.name]).to(device))
    if args.variant == "original":
        common.write_json(directory/"compensation.json", dict(seconds=0., variant=args.variant, nonconv_unchanged=True))
        return
    images = torch.load(cache/"images.pt", map_location="cpu", weights_only=False) if args.variant=="sequential" else None
    rows, start_layer, elapsed = [], 0, 0.
    resume = directory/"compensation_state.pt"
    if resume.exists():
        saved = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        rows, start_layer, elapsed = saved["rows"], saved["next_layer"], saved["seconds"]
        event(directory, "resume_compensation", next_layer=start_layer)
    started = time.monotonic()
    for i, spec in enumerate(specs):
        if i < start_layer:
            continue
        path = cache/ready["layers"][i]["file"]
        if common.sha(path) != ready["cache_files"][i]["sha256"]:
            raise ValueError("Teacher cache checksum changed")
        data = torch.load(path, map_location="cpu", weights_only=False)
        if data["name"] != spec.name:
            raise ValueError("Teacher cache layer order changed")
        inputs, input_deltas = {}, {}
        for name in ["fit", "check"]:
            if args.variant=="sequential":
                inputs[name] = student_patches(model, spec.module, images[name], data[name]["positions"], device)
            else:
                inputs[name] = data[name]["x"]
            input_deltas[name] = float((inputs[name]-data[name]["x"]).square().mean().sqrt())
        xx = {n:v.to(device) for n,v in inputs.items()}
        teacher_weight = dense_state[spec.name+".weight"].flatten(1).to(device)
        layer_output = torch.zeros_like(teacher_weight)
        for cluster, mask in enumerate(masks[spec.name]):
            a, b = cluster*8, min(cluster*8+8, len(teacher_weight))
            value, report = compensate(teacher_weight[a:b], mask.to(device), xx["fit"], data["fit"]["y"][a:b].to(device),
                                       xx["check"], data["check"]["y"][a:b].to(device))
            layer_output[a:b] = value
            rows.append(dict(layer=spec.name, cluster=cluster, input_fit_delta_rms=input_deltas["fit"],
                             input_check_delta_rms=input_deltas["check"], **report))
        spec.module.weight.copy_(layer_output.reshape_as(spec.module.weight))
        for name, value in model.state_dict().items():
            if name in nonconv and not torch.equal(value.cpu(), nonconv[name]):
                raise ValueError("Offline compensation modified BN/bias/classifier")
        for cluster, mask in enumerate(masks[spec.name]):
            values = spec.module.weight[cluster*8:cluster*8+8].flatten(1)
            if torch.count_nonzero(values[:, ~mask.to(device)]):
                raise ValueError("Forbidden weights became nonzero")
        torch.save(dict(model=model.state_dict(), rows=rows, next_layer=i+1, seconds=elapsed+time.monotonic()-started), directory/"compensation_state.tmp")
        (directory/"compensation_state.tmp").replace(resume)
        event(directory, "layer_compensated", layer=spec.name, index=i+1, total=len(specs), input_delta=input_deltas)
    common.write_csv(directory/"cluster_reconstruction.csv", rows)
    summary = dict(variant=args.variant, seconds=elapsed+time.monotonic()-started, clusters=len(rows),
                   accepted=sum(r["status"]=="accepted" for r in rows), fallback=sum(r["status"]=="fallback" for r in rows),
                   empty=sum(r["status"]=="empty" for r in rows), full=sum(r["status"]=="full" for r in rows),
                   nonconv_unchanged=True, max_weight_after=max(r["compensated_abs_max"] for r in rows),
                   max_weight_before=max(r["original_abs_max"] for r in rows))
    common.write_json(directory/"compensation.json", summary)
    event(directory, "compensation_complete", **summary)


def train(args, device):
    directory = args.root/f"seed{args.seed}_{args.variant}"
    directory.mkdir(parents=True, exist_ok=True)
    cache = cache_dir(args)
    if not (cache/"ready.json").exists():
        raise ValueError("Prepare cache first")
    freeze(cache/"configuration.json", fingerprint(args))
    freeze(directory/"configuration.json", dict(**fingerprint(args), variant=args.variant,
           cache_manifest_sha256=common.sha(cache/"ready.json"), total_ft_epochs=10))
    ready = json.loads((cache/"ready.json").read_text())
    if common.sha(cache/"masks.pt") != ready["mask_sha256"] or common.sha(cache/"images.pt") != ready["image_sha256"]:
        raise ValueError("Calibration masks/images checksum changed")
    model, specs, payload = load_model(args.seed, device)
    masks = torch.load(cache/"masks.pt", map_location="cpu", weights_only=False)
    torch.save(masks, directory/"masks.pt")
    init = directory/"initialization.pt"
    if init.exists():
        model.load_state_dict(torch.load(init, map_location=device, weights_only=False)["model"])
    else:
        initialize(model, specs, payload["model_state_dict"], masks, args, directory, device)
        torch.save(dict(model=model.state_dict()), init)
    del payload
    stats = build_stats(specs, masks)
    (train_loader, val_loader, _unused_test), _, metadata = core.build_cifar10(PROJECT/"data", 128, 32, None, None, None, 2, True, 0, args.seed, 5000)
    diagnostic = directory/"initial_validation.json"
    if diagnostic.exists():
        initial = json.loads(diagnostic.read_text())
    else:
        loss, accuracy = core.evaluate(model, val_loader, device, use_amp="float16")
        if not math.isfinite(loss):
            raise FloatingPointError("Non-finite initial AMP validation loss")
        clone = copy.deepcopy(model)
        images = torch.load(cache/"images.pt", map_location="cpu", weights_only=False)["fit"].to(device)
        recalibrate_bn(clone, (images, None))
        bn_loss, bn_accuracy = core.evaluate(clone, val_loader, device, use_amp="float16")
        del clone, images
        initial = dict(raw_loss=loss, raw_accuracy=accuracy, bn_clone_loss=bn_loss, bn_clone_accuracy=bn_accuracy)
        common.write_json(diagnostic, initial)
    optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9, weight_decay=.0005)
    scaler = core.make_scaler(device, "float16")
    core.reset_training_rng(args.seed+100000, train_loader)
    epoch, history = 0, []
    resume = directory/"training_state.pt"
    if resume.exists():
        saved = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        restore_rng(saved["rng"], train_loader)
        epoch, history = saved["epoch"], saved["history"]
        event(directory, "resume_training", epoch=epoch)
    started = time.monotonic()
    for epoch in range(epoch+1, 11):
        train_loss, train_acc = core.train_one_epoch(model, train_loader, optimizer, device, str(epoch),
                                                     prune_stats=stats, tm=8, scaler=scaler, use_amp="float16")
        val_loss, val_acc = core.evaluate(model, val_loader, device, use_amp="float16")
        if not math.isfinite(val_loss):
            raise FloatingPointError("Non-finite FT validation loss")
        row = dict(epoch=epoch, train_loss=train_loss, train_accuracy=train_acc, validation_loss=val_loss, validation_accuracy=val_acc)
        history.append(row)
        event(directory, "epoch_complete", **row)
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                        rng=rng_state(train_loader), epoch=epoch, history=history), directory/"training_state.tmp")
        (directory/"training_state.tmp").replace(resume)
    common.write_csv(directory/"training_history.csv", history)
    for spec in specs:
        values = spec.module.weight.detach().flatten(1)
        expanded = masks[spec.name].repeat_interleave(8, 0)[:len(values)].to(device)
        if torch.count_nonzero(values[~expanded]):
            raise ValueError("Fine-tuning regrew forbidden connections")
    torch.save(dict(model_state_dict={n:v.detach().cpu() for n,v in model.state_dict().items()}, variant=args.variant), directory/"final_model.pt")
    result = dict(seed=args.seed, variant=args.variant, ft0_accuracy=initial["raw_accuracy"],
                  ft0_bn_clone_accuracy=initial["bn_clone_accuracy"], ft2_accuracy=history[1]["validation_accuracy"],
                  ft10_accuracy=history[-1]["validation_accuracy"], validation_loss=history[-1]["validation_loss"],
                  sparsity=core.summarize(stats)["pruning_ratio"], training_seconds=time.monotonic()-started,
                  peak_gpu_bytes=torch.cuda.max_memory_allocated() if device.type=="cuda" else 0,
                  epochs=10, split_id=metadata["split_id"], mask_sha256=common.sha(cache/"masks.pt"), test_evaluated=False)
    common.write_json(directory/"result.json", result)
    event(directory, "worker_complete", **result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "train"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variant", choices=VARIANTS, default="original")
    parser.add_argument("--root", type=Path, default=ROOT)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (prepare if args.mode=="prepare" else train)(args, device)


if __name__ == "__main__":
    main()
