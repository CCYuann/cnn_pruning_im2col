#!/usr/bin/env python3
"""Prepare fixed layer quotas, regroup offline, and train paired branches."""

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
from adapted_gradient_mask import restore_rng, rng_state
from fixed_quota_grouping import VARIANTS, closure, discarded, exchange_search, grouping_loss, shared_masks, validate_groups
from task_loss_mask_search import recalibrate_bn
from weight_compensation_worker import freeze, load_model

core = previous.core
PROJECT = common.PROJECT
ROOT = PROJECT/"runs/fixed_quota_filter_regrouping"


def event(directory, kind, **details):
    value = dict(time=common.stamp(), kind=kind, **details)
    with (directory/"journal.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False)+"\n")
    print(json.dumps(value, ensure_ascii=False), flush=True)


def fingerprint(seed):
    return dict(seed=seed, checkpoint_sha256=common.sha(common.checkpoint("resnet50", seed)),
                source_sha256={name:common.sha(PROJECT/name) for name in [
                    "scripts/fixed_quota_grouping_worker.py", "scripts/fixed_quota_grouping.py",
                    "scripts/weight_compensation_worker.py", "scripts/adapted_gradient_mask.py",
                    "scripts/run_shared_mask_repair.py", "scripts/task_loss_mask_search.py",
                    "scripts/run_fair_criterion_r0p9.py", "resnet34_pruning_experiments.py"]},
                tm=8, target_sparsity=.9, quota_source="global_layer_uniform_integer",
                ft_epochs=10, batch_size=128, lr=.001, momentum=.9, weight_decay=.0005,
                amp="float16", bn_calibration_images=512, bn_calibration_seed=20261109)


def cache_dir(args):
    return args.root/"prepared"/f"seed{args.seed}"


@torch.no_grad()
def prepare(args, device):
    directory = cache_dir(args)
    directory.mkdir(parents=True, exist_ok=True)
    freeze(directory/"configuration.json", fingerprint(args.seed))
    if (directory/"ready.json").exists():
        return
    started = time.monotonic()
    model, specs, payload = load_model(args.seed, device)
    reference = core.global_unstructured_reference(specs, 8, .9)
    packs = {v:{} for v in VARIANTS}
    rows, searches = [], []
    total, kept, dense_macs, kept_macs = 0, 0, 0, 0
    for spec in specs:
        layer_started = time.monotonic()
        weights = spec.module.weight.detach().flatten(1)
        n, k = weights.shape
        if n % 8 or spec.module.groups != 1:
            raise ValueError("Only complete clusters of standard convolutions are supported")
        q = k-round(reference[spec.name]["layer_ratio"]*k)
        contiguous = torch.arange(n, device=device).reshape(-1, 8)
        jaccard = torch.tensor(core.similarity_filter_groups(reference[spec.name]["full_mask"].to(device), 8), device=device)
        digest = int.from_bytes(hashlib.sha256(spec.name.encode()).digest()[:4], "little")
        swapped, search = exchange_search(weights, jaccard, q, 20261110+digest+args.seed*100)
        searches.append(dict(layer=spec.name, **search))
        groups = dict(contiguous=contiguous, jaccard=jaccard, jaccard_swap=swapped,
                      norm_sorted=weights.double().abs().sum(1).argsort(stable=True).reshape(-1, 8))
        for index in range(3):
            generator = torch.Generator().manual_seed(20261110+digest+args.seed*100+index)
            groups[f"random{index}"] = torch.randperm(n, generator=generator).reshape(-1, 8).to(device)
        lower = float(discarded(weights.double().abs(), q).sum())
        baseline = float(grouping_loss(weights, contiguous, q))
        hw = core.output_hw(spec.input_hw, spec.module)
        total += weights.numel()
        kept += n*q
        dense_macs += weights.numel()*math.prod(hw)
        kept_macs += n*q*math.prod(hw)
        for variant in VARIANTS:
            group = groups[variant]
            mask, full = shared_masks(weights, group, q)
            cost = float((weights.double().abs()*~full).sum())
            torch.testing.assert_close(torch.tensor(cost, dtype=torch.float64), grouping_loss(weights, group, q).cpu(), rtol=1e-10, atol=1e-8)
            if cost < lower-max(1e-8, lower*1e-10):
                raise ValueError("Cost below independent-filter lower bound")
            packs[variant][spec.name] = dict(groups=group.cpu(), masks=mask.cpu(), q=q,
                shape=list(spec.module.weight.shape), output_hw=list(hw))
            rows.append(dict(layer=spec.name, variant=variant, filters=n, k=k, q=q, clusters=n//8,
                reference_sparsity=reference[spec.name]["layer_ratio"], actual_sparsity=1-q/k,
                discarded_l1=cost, independent_lower_l1=lower, contiguous_l1=baseline,
                gap_closure=closure(baseline,cost,lower), kept_weights=n*q, kept_macs=n*q*math.prod(hw)))
        event(directory, "layer_prepared", layer=spec.name, q=q, k=k, swaps=search["accepted"], seconds=time.monotonic()-layer_started)
    for variant in VARIANTS:
        torch.save(packs[variant], directory/f"{variant}.pt")
    common.write_csv(directory/"layer_metrics.csv", rows)
    common.write_json(directory/"search_history.json", searches)
    loaders, chosen = previous.calibration_loaders(SimpleNamespace(images=512, calibration_seed=20261109))
    dataset = loaders[0].dataset.dataset
    validation = set(torch.randperm(50000, generator=torch.Generator().manual_seed(0))[:5000].tolist())
    indices = chosen[:512]
    if len(set(indices)) != 512 or set(indices)&validation:
        raise ValueError("BN diagnostic images overlap validation")
    torch.save(torch.stack([dataset[i][0] for i in indices]), directory/"bn_images.pt")
    common.write_json(directory/"data_indices.json", dict(bn_train_indices=indices, split_id="42fbb7440cc27c71", test_evaluated=False))
    common.write_json(directory/"ready.json", dict(seed=args.seed, variants=VARIANTS, total_weights=total, kept_weights=kept,
        sparsity=1-kept/total, dense_macs=dense_macs, kept_macs=kept_macs, seconds=time.monotonic()-started,
        files_sha256={name:common.sha(directory/name) for name in [*(v+".pt" for v in VARIANTS), "bn_images.pt", "data_indices.json", "layer_metrics.csv", "search_history.json"]},
        dense_training_config=payload["training_config"], test_evaluated=False))
    event(directory, "prepare_complete", seconds=time.monotonic()-started, sparsity=1-kept/total)


@torch.no_grad()
def install_masks(model, specs, pack):
    stats = []
    nonconv = {n:v.clone() for n,v in model.state_dict().items() if n not in {s.name+".weight" for s in specs}}
    for spec in specs:
        record = pack[spec.name]
        groups, masks = record["groups"].to(spec.module.weight.device), record["masks"].to(spec.module.weight.device)
        validate_groups(groups, spec.module.out_channels)
        if not bool((masks.sum(1)==record["q"]).all()):
            raise ValueError("Wrong per-cluster quota")
        full = torch.zeros_like(spec.module.weight.flatten(1), dtype=torch.bool)
        full[groups] = masks[:,None,:].expand(-1, 8, -1)
        full = full.reshape_as(spec.module.weight)
        spec.module.weight.mul_(full)
        stats.append(SimpleNamespace(name=spec.name, full_mask=full, masks=[]))
    for name, value in nonconv.items():
        if not torch.equal(value, model.state_dict()[name]):
            raise ValueError("Initialization changed a non-convolution parameter or buffer")
    return stats


def train(args, device):
    directory = args.root/f"seed{args.seed}_{args.variant}"
    directory.mkdir(parents=True, exist_ok=True)
    cache = cache_dir(args)
    freeze(cache/"configuration.json", fingerprint(args.seed))
    ready = json.loads((cache/"ready.json").read_text())
    freeze(directory/"configuration.json", dict(**fingerprint(args.seed), variant=args.variant,
        preparation_sha256=common.sha(cache/"ready.json")))
    for name, digest in ready["files_sha256"].items():
        if common.sha(cache/name) != digest:
            raise ValueError("Prepared data changed: "+name)
    model, specs, payload = load_model(args.seed, device)
    pack = torch.load(cache/f"{args.variant}.pt", map_location="cpu", weights_only=False)
    stats = install_masks(model, specs, pack)
    del payload
    (train_loader, val_loader, _unused_test), _, metadata = core.build_cifar10(PROJECT/"data",128,32,None,None,None,2,True,0,args.seed,5000)
    if metadata["split_id"] != "42fbb7440cc27c71":
        raise ValueError("Wrong data split")
    diagnostic = directory/"initial_validation.json"
    if diagnostic.exists():
        initial = json.loads(diagnostic.read_text())
    else:
        loss, accuracy = core.evaluate(model, val_loader, device, use_amp="float16")
        clone = copy.deepcopy(model)
        images = torch.load(cache/"bn_images.pt", map_location=device, weights_only=False)
        recalibrate_bn(clone, (images,None))
        bn_loss, bn_accuracy = core.evaluate(clone, val_loader, device, use_amp="float16")
        del clone, images
        if not math.isfinite(loss) or not math.isfinite(bn_loss):
            raise FloatingPointError("Non-finite FT0 validation")
        initial = dict(raw_loss=loss, raw_accuracy=accuracy, bn_clone_loss=bn_loss, bn_clone_accuracy=bn_accuracy)
        common.write_json(diagnostic, initial)
        event(directory, "initial_evaluation", **initial)
    optimizer = torch.optim.SGD(model.parameters(), lr=.001,momentum=.9,weight_decay=.0005)
    scaler = core.make_scaler(device,"float16")
    core.reset_training_rng(args.seed+100000,train_loader)
    epoch, history, elapsed = 0, [], 0.
    resume = directory/"training_state.pt"
    if resume.exists():
        saved = torch.load(resume,map_location=device,weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        restore_rng(saved["rng"],train_loader)
        epoch, history, elapsed = saved["epoch"], saved["history"], saved["elapsed"]
        event(directory,"resume_training",epoch=epoch)
    started = time.monotonic()
    for epoch in range(epoch+1,11):
        train_loss, train_acc = core.train_one_epoch(model,train_loader,optimizer,device,str(epoch),prune_stats=stats,tm=8,scaler=scaler,use_amp="float16")
        val_loss, val_acc = core.evaluate(model,val_loader,device,use_amp="float16")
        if not math.isfinite(val_loss):
            raise FloatingPointError("Non-finite validation loss")
        row = dict(epoch=epoch,train_loss=train_loss,train_accuracy=train_acc,validation_loss=val_loss,validation_accuracy=val_acc)
        history.append(row)
        event(directory,"epoch_complete",**row)
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
            rng=rng_state(train_loader),epoch=epoch,history=history,elapsed=elapsed+time.monotonic()-started),directory/"training_state.tmp")
        (directory/"training_state.tmp").replace(resume)
    for spec, stat in zip(specs,stats):
        if torch.count_nonzero(spec.module.weight.detach()[stat.full_mask==0]):
            raise ValueError("Forbidden weights regrew")
    common.write_csv(directory/"training_history.csv",history)
    torch.save(dict(model_state_dict={n:v.detach().cpu() for n,v in model.state_dict().items()},variant=args.variant),directory/"final_model.pt")
    result = dict(seed=args.seed,variant=args.variant,ft0_accuracy=initial["raw_accuracy"],
        ft0_bn_clone_accuracy=initial["bn_clone_accuracy"],ft1_accuracy=history[0]["validation_accuracy"],
        ft2_accuracy=history[1]["validation_accuracy"],ft10_accuracy=history[-1]["validation_accuracy"],
        validation_loss=history[-1]["validation_loss"],sparsity=ready["sparsity"],kept_weights=ready["kept_weights"],
        kept_macs=ready["kept_macs"],training_seconds=elapsed+time.monotonic()-started,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),epochs=10,split_id=metadata["split_id"],
        grouping_sha256=common.sha(cache/f"{args.variant}.pt"),test_evaluated=False)
    common.write_json(directory/"result.json",result)
    event(directory,"worker_complete",**result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["prepare","train"])
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--variant",choices=VARIANTS,default="contiguous")
    parser.add_argument("--root",type=Path,default=ROOT)
    args = parser.parse_args()
    args.root = args.root.resolve()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True,warn_only=True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    core.reset_training_rng(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; do not silently run this study on CPU")
    (prepare if args.mode=="prepare" else train)(args,torch.device("cuda"))


if __name__ == "__main__":
    main()
