#!/usr/bin/env python3
"""Re-run the frozen A/B fixed-support comparison at another sparsity.

Wraps weight_compensation_worker.py without editing it, because its SHA-256 is
pinned in the frozen 90% configurations. Teacher patches do not depend on the
mask, so they are symlinked from the 90% calibration cache instead of copied.
"""

import argparse
import json
import os
from pathlib import Path

import torch

import weight_compensation_worker as base

common, core = base.common, base.core
SOURCE = base.ROOT/"calibration"
SHARED = ["images.pt", "data_indices.json", "dense_training_config.json"]
RATIO = None


def fingerprint(args):
    return dict(base_fingerprint(args), target_sparsity=RATIO, source_calibration=str(SOURCE),
                ratio_worker_sha256=common.sha(Path(__file__)))


base_fingerprint, base.fingerprint = base.fingerprint, fingerprint


@torch.no_grad()
def make_masks(specs, ratio):
    reference = core.global_unstructured_reference(specs, 8, ratio)
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
    return masks


def prepare(args, device):
    directory = base.cache_dir(args)
    directory.mkdir(parents=True, exist_ok=True)
    base.freeze(directory/"configuration.json", fingerprint(args))
    if (directory/"ready.json").exists():
        return
    source = SOURCE/f"seed{args.seed}"
    ready = json.loads((source/"ready.json").read_text())
    _, specs, _ = base.load_model(args.seed, device)
    frozen = torch.load(source/"masks.pt", map_location="cpu", weights_only=False)
    rebuilt = make_masks(specs, .9)
    if frozen.keys() != rebuilt.keys() or any(not torch.equal(frozen[n], rebuilt[n]) for n in frozen):
        raise ValueError("Mask builder does not reproduce the frozen 90% masks")
    masks = make_masks(specs, RATIO)
    torch.save(masks, directory/"masks.pt")
    for name in SHARED+[r["file"] for r in ready["layers"]]:
        link = directory/name
        if not link.exists():
            link.symlink_to(source/name)
    for layer in ready["layers"]:
        layer["max_q"] = int(masks[layer["name"]].sum(1).max())
    ready.update(kept_weights=sum(int(m.sum())*8 for m in masks.values()), mask_sha256=common.sha(directory/"masks.pt"),
                 zero_clusters=sum(int((m.sum(1)==0).sum()) for m in masks.values()),
                 max_q=max(int(m.sum(1).max()) for m in masks.values()), reproduced_r0p9_masks=True, target_sparsity=RATIO)
    common.write_json(directory/"ready.json", ready)
    base.event(directory, "preparation_complete", zero_clusters=ready["zero_clusters"], kept_weights=ready["kept_weights"])


def main():
    global RATIO
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "train"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", choices=["original", "dense_input"], default="original")
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    args.root, RATIO = args.root.resolve(), args.ratio
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    core.reset_training_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (prepare if args.mode=="prepare" else base.train)(args, device)


if __name__ == "__main__":
    main()
