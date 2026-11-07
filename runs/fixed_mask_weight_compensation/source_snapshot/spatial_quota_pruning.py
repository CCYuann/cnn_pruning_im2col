#!/usr/bin/env python3
"""Interpretation audit: flattened K versus per-spatial-position quotas."""

import argparse
import copy
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import torch

import run_fair_criterion_r0p9 as common
import run_shared_mask_repair as previous

core = previous.core
PROJECT = common.PROJECT
ROOT = PROJECT / "runs/spatial_quota_baseline_study"


def spatial_counts(reference, kept):
    """Largest-remainder rounding of unstructured survivors / cluster width."""
    if reference.ndim != 4 or not torch.all((reference == 0) | (reference == 1)):
        raise ValueError("Expected a binary [filters, Cin/groups, Kh, Kw] reference")
    filters, channels, kh, kw = reference.shape
    ideal = reference.double().sum((0, 1)).flatten().cpu() / filters
    if not 0 <= kept <= channels * kh * kw or abs(float(ideal.sum()) - kept) > .500001:
        raise ValueError("Target must be the nearest whole-cluster survivor budget")
    counts = [int(math.floor(float(value))) for value in ideal]
    remainder = kept - sum(counts)
    eligible = sorted(range(len(counts)), key=lambda i: (-(float(ideal[i]) - counts[i]), i))
    for i in eligible:
        if remainder == 0:
            break
        if counts[i] < channels:
            counts[i] += 1
            remainder -= 1
    if remainder or sum(counts) != kept:
        raise ValueError("Cannot allocate spatial quotas exactly")
    return torch.tensor(counts, dtype=torch.long).reshape(kh, kw), ideal.reshape(kh, kw)


def cluster_masks(weight, reference, criterion):
    positions = weight[0].numel()
    local_ratio = 1 - float(reference.sum()) / reference.numel()
    removed = min(max(round(local_ratio * positions), 0), positions)
    kept = positions - removed
    importance = core.compute_importance(weight, criterion)
    flat = torch.ones(positions, dtype=torch.bool, device=weight.device)
    flat[torch.argsort(importance.flatten())[:removed]] = False
    counts, ideal = spatial_counts(reference, kept)
    channels, kh, kw = importance.shape
    spatial = torch.ones_like(importance, dtype=torch.bool)
    for r in range(kh):
        for s in range(kw):
            drop = channels - int(counts[r, s])
            spatial[torch.argsort(importance[:, r, s])[:drop], r, s] = False
    flat = flat.reshape(channels, kh, kw)
    if int(flat.sum()) != kept or int(spatial.sum()) != kept:
        raise ValueError("Cluster budget mismatch")
    if (kh, kw) == (1, 1) and not torch.equal(flat, spatial):
        raise ValueError("1x1 interpretation must produce identical masks")
    return flat, spatial, counts, ideal


def event(directory, kind, **details):
    value = dict(time=common.stamp(), kind=kind, **details)
    with (directory / "journal.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    print(json.dumps(value, ensure_ascii=False), flush=True)


def output(args):
    return args.root / f"{args.model}_seed{args.seed}_{args.criterion}_{args.granularity}"


@torch.no_grad()
def generate(model, specs, args, directory):
    reference = core.global_unstructured_reference(specs, 8, .9)
    variants = {"flat": {}, "spatial": {}}
    cluster_rows, position_rows = [], []
    original, kept_weights, changed_weights = 0, 0, 0
    one_by_one = 0
    for spec in specs:
        conv = spec.module
        original += conv.weight.numel()
        layer = {"flat": [], "spatial": []}
        for cluster, group in enumerate(reference[spec.name]["filter_groups"]):
            ww = conv.weight[group]
            rr = reference[spec.name]["full_mask"][group]
            flat, spatial, counts, ideal = cluster_masks(ww, rr, args.criterion)
            changed = int((flat != spatial).sum())
            kept_weights += int(flat.sum()) * len(group)
            changed_weights += changed * len(group)
            if conv.kernel_size == (1, 1):
                one_by_one += 1
            for label, mask in [("flat", flat), ("spatial", spatial)]:
                layer[label].append(mask.flatten().cpu())
            naive_kept = sum(round(float(value)) for value in ideal.flatten())
            cluster_rows.append(dict(layer=spec.name, cluster=cluster, filters=len(group), K=flat.numel(),
                                     kept_count=int(flat.sum()), changed_k_positions=changed,
                                     spatial_positions=counts.numel(), naive_spatial_kept=naive_kept,
                                     integer_correction=int(flat.sum())-naive_kept))
            for r in range(conv.kernel_size[0]):
                for s in range(conv.kernel_size[1]):
                    position_rows.append(dict(layer=spec.name, cluster=cluster, r=r, s=s,
                                              ideal_reference_kept=float(ideal[r, s]), flat_kept=int(flat[:, r, s].sum()),
                                              spatial_kept=int(counts[r, s])))
        for label in variants:
            variants[label][spec.name] = torch.stack(layer[label])
        event(directory, "layer_masks", layer=spec.name, kernel=list(conv.kernel_size))
    # Audit exact equality to the existing production l1/top2 implementation.
    other = copy.deepcopy(model)
    other_specs = core.collect_conv_specs(other, 32)
    stats = core.cross_filter_prune_(other, other_specs, 8, .9, args.criterion,
                                    ratio_source="global_cluster", sparsity_reference=reference)
    for stat in stats:
        conv = dict(other.named_modules())[stat.name]
        ours = variants["flat"][stat.name].repeat_interleave(8, 0)[:conv.out_channels].reshape_as(conv.weight)
        if not torch.equal(ours, stat.full_mask.bool().cpu()):
            raise ValueError("Flat masks do not reproduce existing core")
    del other
    baseline = common.baselines()[(args.model, args.seed)]
    if not math.isclose(1-kept_weights/original, float(baseline["pruning_ratio"]), abs_tol=1e-6):
        raise ValueError("Does not match the formal l1 weight budget")
    torch.save({args.criterion: variants[args.granularity]}, directory / "masks.pt")
    torch.save(variants, directory / "mask_audit.pt")
    common.write_csv(directory / "clusters.csv", cluster_rows)
    common.write_csv(directory / "spatial_counts.csv", position_rows)
    report = dict(model=args.model, seed=args.seed, criterion=args.criterion, granularity=args.granularity,
                  original_weights=original, kept_weights=kept_weights, sparsity=1-kept_weights/original,
                  changed_weight_mask_entries=changed_weights, changed_clusters=sum(r["changed_k_positions"] > 0 for r in cluster_rows),
                  clusters=len(cluster_rows), identical_1x1_clusters=one_by_one, flat_matches_existing_core=True,
                  per_cluster_budget_equal=True, test_evaluated=False,
                  claim="Per-spatial interpretation with exact whole-cluster budget, not verified author implementation")
    common.write_json(directory / "mask_report.json", report)
    event(directory, "mask_audit_complete", **report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["resnet34", "resnet50"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--criterion", choices=["l1_sum", "top2_mean_abs"], required=True)
    parser.add_argument("--granularity", choices=["flat", "spatial"], required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    args.root = args.root.resolve()
    directory = output(args)
    directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    core.reset_training_rng(args.seed)
    checkpoint = common.checkpoint(args.model, args.seed)
    config = dict(model=args.model, seed=args.seed, criterion=args.criterion, granularity=args.granularity,
                  checkpoint_sha256=common.sha(checkpoint), core_sha256=common.sha(common.CORE),
                  worker_sha256=common.sha(Path(__file__)), pilot_sha256=common.sha(Path(previous.__file__)),
                  tm=8, target_sparsity=.9, fine_tune_epochs=10, split_id="42fbb7440cc27c71")
    manifest = directory / "configuration.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError("Frozen configuration changed; use another root")
    common.write_json(manifest, config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["model"] != args.model or payload["seed"] != args.seed or payload["split_metadata"]["split_id"] != config["split_id"]:
        raise ValueError("Wrong Dense checkpoint")
    common.write_json(directory / "dense_training_config.json", payload["training_config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = core.make_model(args.model, 10, 32).to(device)
    model.load_state_dict(payload["model_state_dict"])
    del payload
    model.eval()
    specs = core.collect_conv_specs(model, 32)
    generate(model, specs, args, directory)
    if not (directory / "pilots" / (args.criterion + ".json")).exists():
        previous.pilot(model, specs, SimpleNamespace(root=directory, model=args.model, seed=args.seed, variant=args.criterion), directory, device)


if __name__ == "__main__":
    main()
