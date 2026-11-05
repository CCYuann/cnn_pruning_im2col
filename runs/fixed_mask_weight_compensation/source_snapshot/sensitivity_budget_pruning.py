#!/usr/bin/env python3
"""Run the existing pruning experiment with a fixed-size sensitivity-aware budget."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def allocate_counts(clusters, target_weights):
    """Bounded water filling; this experiment uses full Tm8 clusters only."""
    widths = {c["filters"] for c in clusters}
    if len(widths) != 1:
        raise ValueError("Mixed-width/partial clusters need a separate integer budget solver")
    width = widths.pop()
    if target_weights % width:
        raise ValueError("The requested budget is not representable with whole shared K positions")
    target = target_weights // width
    upper = [c["max_prune"] for c in clusters]
    initial = [min(round(c["reference_ratio"] * c["positions"]), limit) for c, limit in zip(clusters, upper)]
    if not sum(initial) <= target <= sum(upper):
        raise ValueError(f"Budget infeasible: {sum(initial)*width} <= {target_weights} <= {sum(upper)*width} fails")
    if target == sum(upper):
        return upper
    rates = [count / c["positions"] for c, count in zip(clusters, initial)]
    mobility = [1 / (1 + max(0, c["sensitivity"])) for c in clusters]

    def continuous(level):
        return [min(limit, c["positions"] * (rate + level * weight))
                for c, limit, rate, weight in zip(clusters, upper, rates, mobility)]

    low, high = 0.0, 1.0
    while sum(continuous(high)) < target:
        high *= 2
    for _ in range(80):
        middle = (low + high) / 2
        if sum(continuous(middle)) < target:
            low = middle
        else:
            high = middle
    ideal = continuous(high)
    counts = [max(start, min(limit, math.floor(value))) for start, limit, value in zip(initial, upper, ideal)]
    remainder = target - sum(counts)
    candidates = sorted(range(len(counts)), key=lambda i: (-(ideal[i] - counts[i]), clusters[i]["sensitivity"], i))
    for i in candidates:
        if remainder <= 0:
            break
        if counts[i] < upper[i]:
            counts[i] += 1
            remainder -= 1
    if remainder or sum(counts) * width != target_weights:
        raise ValueError("Integer allocation failed to exactly match the requested weight budget")
    return counts


def main():
    options = argparse.ArgumentParser(add_help=False)
    options.add_argument("--ranking-csv", type=Path, required=True)
    options.add_argument("--target-pruned-weights", type=int, required=True)
    options.add_argument("--allocation-dir", type=Path, required=True)
    own, remainder = options.parse_known_args()
    sys.path.insert(0, str(PROJECT))
    import resnet34_pruning_experiments as core

    with own.ranking_csv.open(encoding="utf-8-sig", newline="") as stream:
        ranking = list(csv.DictReader(stream))
    ranking.sort(key=lambda r: float(r["mean_drop_pp"]), reverse=True)
    if not ranking or any(r["criterion"] != "top2_mean_abs" or r["evaluation_split"] != "validation" for r in ranking):
        raise ValueError("Expected formal validation-only top2 sensitivity ranking")
    sensitivities = {r["layer_name"]: float(r["mean_drop_pp"]) for r in ranking}
    caps = {r["layer_name"]: 0.0 if i == 0 else 0.2 for i, r in enumerate(ranking[:4])}
    original_reference = core.global_unstructured_reference

    def redistributed_reference(specs, tm, target_sparsity, include_1x1=True, filter_grouping="contiguous"):
        reference = original_reference(specs, tm, target_sparsity, include_1x1, filter_grouping)
        clusters, total = [], 0
        for spec in specs:
            if spec.name not in reference:
                continue
            positions = spec.module.weight[0].numel()
            total += spec.module.weight.numel()
            if spec.name not in sensitivities:
                raise ValueError(f"Missing sensitivity for {spec.name}")
            for cluster_id, group in enumerate(reference[spec.name]["filter_groups"]):
                maximum = math.floor(positions * caps[spec.name] + 1e-12) if spec.name in caps else positions - 1
                clusters.append({"layer_name": spec.name, "cluster_id": cluster_id, "filters": len(group),
                                 "positions": positions, "reference_ratio": reference[spec.name]["cluster_ratios"][cluster_id],
                                 "sensitivity": sensitivities[spec.name], "max_prune": maximum,
                                 "protected": spec.name in caps})
        counts = allocate_counts(clusters, own.target_pruned_weights)
        for cluster, count in zip(clusters, counts):
            reference[cluster["layer_name"]]["cluster_ratios"][cluster["cluster_id"]] = count / cluster["positions"]
            cluster["allocated_prune_k"] = count
            cluster["allocated_prune_ratio"] = count / cluster["positions"]
            cluster["allocated_pruned_weights"] = count * cluster["filters"]
        own.allocation_dir.mkdir(parents=True, exist_ok=True)
        with (own.allocation_dir / "cluster_allocation.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(clusters[0]))
            writer.writeheader()
            writer.writerows(clusters)
        (own.allocation_dir / "budget.json").write_text(json.dumps({
            "target_pruned_weights": own.target_pruned_weights, "total_weights": total,
            "actual_planned_pruning_ratio": own.target_pruned_weights / total,
            "allocated_pruned_weights": sum(c["allocated_pruned_weights"] for c in clusters),
            "protected_layer_caps": caps, "non_sensitive_min_kept_k": 1,
            "allocation_rule": "r=min(cap,initial_r+lambda/(1+max(0,mean_drop_pp))); exact integer correction",
        }, indent=2) + "\n", encoding="utf-8")
        print(f"Sensitivity redistribution: {own.target_pruned_weights}/{total} weights pruned = {own.target_pruned_weights/total:.8%}", flush=True)
        return reference

    core.global_unstructured_reference = redistributed_reference
    core.cross_ratio_config_label = lambda *args, **kwargs: "sensitivity_redistributed_exact_budget"
    sys.argv = [sys.argv[0]] + remainder
    core.main()


if __name__ == "__main__":
    main()
