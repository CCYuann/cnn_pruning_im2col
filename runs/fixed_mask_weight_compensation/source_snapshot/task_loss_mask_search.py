#!/usr/bin/env python3
"""Fixed-quota shared-mask proposals selected by the current network's task loss."""

import argparse
import copy
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

import run_fair_criterion_r0p9 as common
import run_shared_mask_repair as previous

core = previous.core
PROJECT = common.PROJECT
ROOT = PROJECT / "runs/task_loss_mask_study"


def emit(directory, kind, **details):
    value = dict(time=common.stamp(), kind=kind, **details)
    with (directory / "journal.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    print(json.dumps(value, ensure_ascii=False), flush=True)


def assert_quotas(reference, candidate):
    if reference.keys() != candidate.keys():
        raise ValueError("Layer sets differ")
    for name, mask in reference.items():
        other = candidate[name]
        if other.dtype != torch.bool or other.shape != mask.shape:
            raise ValueError(f"Invalid shared mask: {name}")
        if not torch.equal(mask.sum(1), other.sum(1)):
            raise ValueError(f"Per-cluster kept_count changed: {name}")


def improves(candidate, baseline, minimum=1e-5):
    return math.isfinite(candidate) and candidate < baseline - minimum


def accept(search, check, baseline_search, baseline_check, policy):
    return improves(search, baseline_search) and (
        policy == "search_only" or improves(check, baseline_check)
    )


def full_mask(conv, mask):
    return mask.repeat_interleave(8, 0)[:conv.out_channels].reshape_as(conv.weight)


@torch.no_grad()
def assign(conv, dense, mask):
    conv.weight.copy_(dense * full_mask(conv, mask).to(conv.weight.device))


def bn_state(model):
    return {name: value.detach().clone() for name, value in model.named_buffers()}


@torch.no_grad()
def restore_buffers(model, state):
    for name, value in model.named_buffers():
        value.copy_(state[name])


@torch.no_grad()
def recalibrate_bn(model, data):
    model.eval()
    modules = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    momenta = [m.momentum for m in modules]
    try:
        for module in modules:
            module.reset_running_stats()
            module.momentum = None
            module.train()
        images, _ = data
        for batch in images.split(32):
            model(batch)
    finally:
        for module, momentum in zip(modules, momenta):
            module.momentum = momentum
        model.eval()


@torch.no_grad()
def measure(model, data):
    model.eval()
    images, labels = data
    loss, correct = 0.0, 0
    for image, label in zip(images.split(64), labels.split(64)):
        logits = model(image)
        loss += float(F.cross_entropy(logits, label, reduction="sum"))
        correct += int((logits.argmax(1) == label).sum())
    answer = dict(loss=loss / len(images), accuracy=correct / len(images), images=len(images))
    if not math.isfinite(answer["loss"]):
        raise ValueError("Non-finite whole-network classification loss")
    return answer


def data_splits(args, directory, device):
    loaders, chosen = previous.calibration_loaders(SimpleNamespace(images=512, calibration_seed=20261101))
    dataset = loaders[0].dataset.dataset
    split = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(0))
    available = [i for i in split[5000:].tolist() if i not in set(chosen)]
    order = torch.randperm(len(available), generator=torch.Generator().manual_seed(20261102))
    audit = [available[i] for i in order[:1024].tolist()]
    indices = dict(search=chosen[:512], check=chosen[512:], audit=audit)
    validation = set(split[:5000].tolist())
    if any(set(v) & validation for v in indices.values()):
        raise ValueError("Calibration overlaps validation")
    if len(set(sum(indices.values(), []))) != 2048:
        raise ValueError("Search/check/audit overlap")
    common.write_json(directory / "data_indices.json", dict(**indices, split_id="42fbb7440cc27c71", test_used=False))
    result = {}
    for name, values in indices.items():
        samples = [dataset[i] for i in values]
        result[name] = (torch.stack([s[0] for s in samples]).to(device),
                        torch.tensor([s[1] for s in samples], device=device))
    return result


def source_dir(model, seed):
    return PROJECT / "runs/shared_mask_repair_study/v1" / f"{model}_seed{seed}"


def mask_search(model, specs, args, directory, proposals, device):
    baseline = proposals["l1"]
    for variant in ["mean_repair", "worst_repair"]:
        assert_quotas(baseline, proposals[variant])
    data = data_splits(args, directory, device)
    dense = {s.name: s.module.weight.detach().clone() for s in specs}
    checkpoint_buffers = bn_state(model)
    masks = copy.deepcopy(baseline)
    for spec in specs:
        assign(spec.module, dense[spec.name], masks[spec.name])
    resume = directory / "search_state.pt"
    if resume.exists():
        saved = torch.load(resume, map_location=device, weights_only=False)
        masks = {name: value.cpu() for name, value in saved["masks"].items()}
        assert_quotas(baseline, masks)
        for spec in specs:
            assign(spec.module, dense[spec.name], masks[spec.name])
        restore_buffers(model, saved["buffers"])
        before, current, start, decisions = saved["before"], saved["current"], saved["next_layer"], saved["decisions"]
        emit(directory, "resume_search", next_layer=start)
    else:
        if args.bn == "recalibrate":
            recalibrate_bn(model, data["search"])
        before = {name: measure(model, values) for name, values in data.items()}
        current = {name: before[name] for name in ["search", "check"]}
        start, decisions = 0, []
        emit(directory, "baseline_proxy", values=before)
    for layer, spec in enumerate(specs):
        if layer < start:
            continue
        old_mask = masks[spec.name]
        old_buffers = bn_state(model)
        candidates = []
        seen = [old_mask]
        for variant in ["mean_repair", "worst_repair"]:
            candidate = proposals[variant][spec.name]
            if any(torch.equal(candidate, known) for known in seen):
                continue
            seen.append(candidate)
            assign(spec.module, dense[spec.name], candidate)
            restore_buffers(model, old_buffers)
            if args.bn == "recalibrate":
                recalibrate_bn(model, data["search"])
            score = measure(model, data["search"])
            candidates.append(dict(variant=variant, search=score))
        assign(spec.module, dense[spec.name], old_mask)
        restore_buffers(model, old_buffers)
        chosen = min(candidates, key=lambda r: r["search"]["loss"]) if candidates else None
        record = dict(layer=spec.name, layer_index=layer, candidates=candidates,
                      before_search_loss=current["search"]["loss"], before_check_loss=current["check"]["loss"],
                      accepted=False, selected="l1")
        if chosen and improves(chosen["search"]["loss"], current["search"]["loss"]):
            candidate = proposals[chosen["variant"]][spec.name]
            assign(spec.module, dense[spec.name], candidate)
            if args.bn == "recalibrate":
                recalibrate_bn(model, data["search"])
            check = measure(model, data["check"])
            record.update(proposed=chosen["variant"], proposed_check=check)
            if accept(chosen["search"]["loss"], check["loss"], current["search"]["loss"], current["check"]["loss"], args.policy):
                masks[spec.name] = candidate.clone()
                current = dict(search=chosen["search"], check=check)
                record.update(accepted=True, selected=chosen["variant"])
            else:
                assign(spec.module, dense[spec.name], old_mask)
                restore_buffers(model, old_buffers)
        record.update(after_search_loss=current["search"]["loss"], after_check_loss=current["check"]["loss"])
        decisions.append(record)
        emit(directory, "layer_decision", **record)
        saved = dict(masks=masks, buffers=bn_state(model), before=before, current=current,
                     next_layer=layer+1, decisions=decisions)
        torch.save(saved, directory / "search_state.tmp")
        (directory / "search_state.tmp").replace(resume)
    after = {name: measure(model, values) for name, values in data.items()}
    assert_quotas(baseline, masks)
    torch.save({"l1": baseline, "loss_gate": masks}, directory / "masks.pt")
    report = dict(model=args.model, seed=args.seed, policy=args.policy, bn=args.bn, before=before, after=after,
                  accepted_layers=sum(r["accepted"] for r in decisions), total_layers=len(specs),
                  changed_clusters=sum(int((baseline[n] != m).any(1).sum()) for n, m in masks.items()),
                  exact_per_cluster_quotas=True, test_evaluated=False, decisions=decisions)
    common.write_json(directory / "search.json", report)
    restore_buffers(model, checkpoint_buffers)
    emit(directory, "search_complete", accepted_layers=report["accepted_layers"], before=before, after=after)


def pilot(model, specs, args, directory, device):
    # Only masks cross this boundary: proxy BN statistics never enter training.
    call = SimpleNamespace(root=directory, model=args.model, seed=args.seed, variant="loss_gate")
    previous.pilot(model, specs, call, directory, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["search", "pilot"])
    parser.add_argument("--model", default="resnet50", choices=common.MODELS)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--policy", choices=["dual", "search_only"], default="dual")
    parser.add_argument("--bn", choices=["frozen", "recalibrate"], default="frozen")
    args = parser.parse_args()
    directory = args.root.resolve() / (args.bn + "_" + args.policy) / f"{args.model}_seed{args.seed}"
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
    source = source_dir(args.model, args.seed) / "masks.pt"
    config = dict(model=args.model, seed=args.seed, policy=args.policy, bn=args.bn, minimum_ce_gain=1e-5,
                  source_masks=str(source), source_sha256=common.sha(source),
                  checkpoint_sha256=common.sha(common.checkpoint(args.model, args.seed)),
                  worker_sha256=common.sha(Path(__file__)), core_sha256=common.sha(common.CORE),
                  pilot_sha256=common.sha(Path(previous.__file__)), epochs=10)
    manifest = directory / "configuration.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError("Configuration or source changed; use another root")
    common.write_json(manifest, config)
    payload = torch.load(common.checkpoint(args.model, args.seed), map_location="cpu", weights_only=False)
    if payload["model"] != args.model or payload["seed"] != args.seed or payload["split_metadata"]["split_id"] != "42fbb7440cc27c71":
        raise ValueError("Wrong checkpoint")
    model = core.make_model(args.model, 10, 32).to(device)
    model.load_state_dict(payload["model_state_dict"])
    del payload
    model.eval()
    specs = core.collect_conv_specs(model, 32)
    emit(directory, "worker_start", mode=args.mode, config=config)
    if args.mode == "search":
        masks = torch.load(source, map_location="cpu", weights_only=False)
        mask_search(model, specs, args, directory, masks, device)
    else:
        pilot(model, specs, args, directory, device)


if __name__ == "__main__":
    main()
