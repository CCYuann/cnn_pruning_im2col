#!/usr/bin/env python3
"""Layer-wise pruning sensitivity analysis.

This script trains or loads one dense model, then prunes one convolution layer at
a time at multiple ratios and evaluates validation/test accuracy without
fine-tuning. It produces CSV summaries and paper-style sensitivity plots.
"""

import argparse
import csv
import copy
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from resnet34_pruning_experiments import (
    SUPPORTED_MODELS,
    apply_masks_,
    build_animals10,
    build_cifar10,
    build_imagefolder_dataset,
    choose_device,
    collect_conv_specs,
    compute_importance,
    evaluate,
    load_dense_checkpoint,
    make_model,
    make_scaler,
    output_hw,
    parse_models,
    resolve_checkpoint_path,
    train_one_epoch,
)


def parse_ratios(value):
    ratios = []
    for item in value.split(","):
        item = item.strip()
        if item:
            ratios.append(float(item))
    if not ratios:
        raise ValueError("--ratios must contain at least one value")
    return ratios


def layer_stage(model_name, layer_name):
    if model_name in {"resnet34", "resnet50"}:
        for stage in ["layer1", "layer2", "layer3", "layer4"]:
            if layer_name.startswith(stage + "."):
                return stage
        return "stem" if layer_name == "conv1" else "other"
    if model_name == "densenet121":
        parts = layer_name.split(".")
        return parts[1] if len(parts) > 2 and parts[0] == "features" else "other"
    if model_name in {"mobilenet_v2", "efficientnet_v2_m"}:
        parts = layer_name.split(".")
        return ".".join(parts[:2]) if len(parts) >= 2 else layer_name
    if model_name == "googlenet":
        return layer_name.split(".")[0]
    return "other"


def is_depthwise(conv):
    return conv.groups == conv.in_channels and conv.out_channels == conv.in_channels


@torch.no_grad()
def prune_one_layer_cross_filter_(spec, ratio, criterion, tm):
    conv = spec.module
    weight = conv.weight
    out_ch, in_ch, kh, kw = weight.shape
    positions = in_ch * kh * kw
    masks = []
    for start in range(0, out_ch, tm):
        end = min(start + tm, out_ch)
        cluster = weight[start:end]
        num_prune = min(max(round(ratio * positions), 0), positions)
        importance = compute_importance(cluster, criterion).reshape(-1)
        prune_idx = torch.argsort(importance)[:num_prune]
        flat_mask = torch.ones(positions, device=weight.device, dtype=weight.dtype)
        flat_mask[prune_idx] = 0
        mask = flat_mask.reshape(in_ch, kh, kw)
        weight[start:end].mul_(mask)
        masks.append(mask.detach().cpu())
    return masks


@torch.no_grad()
def prune_one_layer_filter_l1_(spec, ratio):
    conv = spec.module
    weight = conv.weight
    out_ch = weight.shape[0]
    num_prune = min(max(round(ratio * out_ch), 0), out_ch)
    mask = torch.ones(out_ch, device=weight.device, dtype=weight.dtype)
    if num_prune > 0:
        scores = weight.abs().reshape(out_ch, -1).sum(dim=1)
        prune_idx = torch.argsort(scores)[:num_prune]
        mask[prune_idx] = 0
        weight.mul_(mask.reshape(out_ch, 1, 1, 1))
    return mask.detach().cpu()


def train_or_load_dense(
    args,
    model_name,
    train_loader,
    eval_loader,
    num_classes,
    split_metadata,
    device,
    use_amp,
):
    model = make_model(model_name, num_classes, args.image_size).to(device)
    fixed_checkpoint = resolve_checkpoint_path(args.dense_checkpoint_in, model_name, args.seed)
    if fixed_checkpoint is not None:
        checkpoint_id, _ = load_dense_checkpoint(
            fixed_checkpoint,
            model,
            args,
            model_name,
            num_classes,
            split_metadata,
            device,
        )
        _, dense_acc = evaluate(model, eval_loader, device, use_amp=use_amp)
        print(f"Validation baseline: {dense_acc:.2%}")
        return model, dense_acc, checkpoint_id

    ckpt_path = Path(args.checkpoint_dir) / f"{args.dataset}_{model_name}_seed{args.seed}_e{args.epochs}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    if args.load_checkpoint and ckpt_path.exists():
        payload = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(payload["model"] if isinstance(payload, dict) and "model" in payload else payload)
        _, dense_acc = evaluate(model, eval_loader, device, use_amp=use_amp)
        print(f"Loaded checkpoint: {ckpt_path} dense_acc={dense_acc:.2%}")
        return model, dense_acc, str(ckpt_path)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = make_scaler(device, use_amp)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, epoch, scaler=scaler, use_amp=use_amp
        )
        scheduler.step()
        _, test_acc = evaluate(model, eval_loader, device, use_amp=use_amp)
        print(
            f"{model_name} epoch {epoch}: train_loss={train_loss:.4f}, "
            f"train_acc={train_acc:.2%}, test_acc={test_acc:.2%}"
        )

    _, dense_acc = evaluate(model, eval_loader, device, use_amp=use_amp)
    if args.save_checkpoint:
        torch.save({"model": model.state_dict(), "dense_acc": dense_acc, "args": vars(args)}, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")
    return model, dense_acc, str(ckpt_path) if ckpt_path.exists() else ""


def sensitivity_rows(args, model_name, dense_model, dense_acc, eval_loader, device, use_amp):
    base_specs = collect_conv_specs(dense_model, args.image_size)
    rows = []
    for layer_index, spec in enumerate(base_specs):
        conv = spec.module
        kh, kw = conv.kernel_size
        if args.no_prune_1x1 and (kh, kw) == (1, 1):
            continue
        if args.skip_depthwise and is_depthwise(conv):
            continue
        if args.only_first_residual_conv and model_name in {"resnet34", "resnet50"}:
            if not (".conv1" in spec.name or spec.name == "conv1"):
                continue

        for ratio in args.ratios:
            model = copy.deepcopy(dense_model)
            specs = collect_conv_specs(model, args.image_size)
            target = next(s for s in specs if s.name == spec.name)
            if args.mode == "cross_filter":
                masks = prune_one_layer_cross_filter_(target, ratio, args.criterion, args.tm)
                pruned_units = sum(int(m.numel() - m.sum().item()) for m in masks)
                total_units = sum(m.numel() for m in masks)
            else:
                mask = prune_one_layer_filter_l1_(target, ratio)
                pruned_units = int(mask.numel() - mask.sum().item())
                total_units = int(mask.numel())

            _, acc = evaluate(model, eval_loader, device, use_amp=use_amp)
            out_h, out_w = output_hw(target.input_hw, target.module)
            weight_count = int(target.module.weight.numel())
            rows.append(
                {
                    "dataset": args.dataset,
                    "model": model_name,
                    "mode": args.mode,
                    "criterion": args.criterion if args.mode == "cross_filter" else "filter_l1",
                    "layer_index": layer_index,
                    "layer_name": spec.name,
                    "stage": layer_stage(model_name, spec.name),
                    "kernel": f"{kh}x{kw}",
                    "groups": conv.groups,
                    "is_1x1": int((kh, kw) == (1, 1)),
                    "is_depthwise": int(is_depthwise(conv)),
                    "out_channels": conv.out_channels,
                    "in_channels": conv.in_channels,
                    "input_hw": f"{spec.input_hw[0]}x{spec.input_hw[1]}",
                    "output_hw": f"{out_h}x{out_w}",
                    "weight_count": weight_count,
                    "prune_ratio": ratio,
                    "pruned_units": pruned_units,
                    "total_units": total_units,
                    "dense_acc": dense_acc,
                    "pruned_acc": acc,
                    "acc_drop_pp": (dense_acc - acc) * 100.0,
                    "evaluation_split": "validation",
                    "seed": args.seed,
                    "tm": args.tm,
                }
            )
            print(f"{model_name} {spec.name} ratio={ratio:.2f} acc={acc:.2%} drop={(dense_acc-acc)*100:.2f}pp")
    return rows


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No sensitivity rows were produced")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def rank_sensitivity_rows(rows):
    grouped = {}
    for row in rows:
        key = (row["model"], row["criterion"], int(row["layer_index"]), row["layer_name"])
        grouped.setdefault(key, []).append(row)

    ranked = []
    for (model, criterion, layer_index, layer_name), items in grouped.items():
        drops = [float(item["acc_drop_pp"]) for item in items]
        first = items[0]
        at_half = [
            float(item["acc_drop_pp"])
            for item in items
            if abs(float(item["prune_ratio"]) - 0.5) < 1e-9
        ]
        ranked.append(
            {
                "model": model,
                "criterion": criterion,
                "layer_index": layer_index,
                "layer_name": layer_name,
                "stage": first["stage"],
                "kernel": first["kernel"],
                "is_1x1": first["is_1x1"],
                "is_depthwise": first["is_depthwise"],
                "mean_drop_pp": sum(drops) / len(drops),
                "max_drop_pp": max(drops),
                "drop_at_50pp": at_half[0] if at_half else "",
                "dense_acc": first["dense_acc"],
                "evaluation_split": first["evaluation_split"],
                "seed": first["seed"],
                "tm": first["tm"],
            }
        )
    return sorted(ranked, key=lambda row: row["mean_drop_pp"], reverse=True)


def plot_model(rows, out_dir, model_name):
    out_dir.mkdir(parents=True, exist_ok=True)
    model_rows = [r for r in rows if r["model"] == model_name]
    if not model_rows:
        return
    ratios = sorted({float(r["prune_ratio"]) for r in model_rows})
    layers = []
    seen = set()
    for r in model_rows:
        key = (int(r["layer_index"]), r["layer_name"])
        if key not in seen:
            seen.add(key)
            layers.append(key)
    layers.sort()

    fig_w = max(11, 0.35 * len(layers) + 5)
    fig, ax = plt.subplots(figsize=(fig_w, 5.8), constrained_layout=True)
    for ratio in ratios:
        ys = []
        xs = []
        for idx, name in layers:
            found = [r for r in model_rows if int(r["layer_index"]) == idx and r["layer_name"] == name and float(r["prune_ratio"]) == ratio]
            if found:
                xs.append(len(xs))
                ys.append(float(found[0]["acc_drop_pp"]))
        ax.plot(xs, ys, marker="o", linewidth=1.6, markersize=3.5, label=f"{ratio:.0%}")

    labels = [f"{idx}:{name}" for idx, name in layers]
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set_title(f"Layer-wise pruning sensitivity - {model_name}")
    ax.set_xlabel("Convolution layer")
    ax.set_ylabel("Accuracy drop vs dense (percentage points)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Prune ratio", ncols=min(4, len(ratios)))
    fig.savefig(out_dir / f"{model_name}_layer_sensitivity.png", dpi=180)
    plt.close(fig)

    heat = []
    for ratio in ratios:
        row = []
        for idx, name in layers:
            found = [r for r in model_rows if int(r["layer_index"]) == idx and r["layer_name"] == name and float(r["prune_ratio"]) == ratio]
            row.append(float(found[0]["acc_drop_pp"]) if found else float("nan"))
        heat.append(row)
    fig_h = max(3.2, 0.45 * len(ratios) + 2.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    im = ax.imshow(heat, aspect="auto", cmap="magma_r")
    ax.set_title(f"Sensitivity heatmap - {model_name}")
    ax.set_xlabel("Convolution layer")
    ax.set_ylabel("Prune ratio")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    ax.set_yticks(range(len(ratios)))
    ax.set_yticklabels([f"{r:.0%}" for r in ratios])
    fig.colorbar(im, ax=ax, label="Accuracy drop (pp)")
    fig.savefig(out_dir / f"{model_name}_layer_sensitivity_heatmap.png", dpi=180)
    plt.close(fig)


def write_readme(path, args, csv_path, rows):
    models = sorted({r["model"] for r in rows})
    text = [
        "# Layer-wise pruning sensitivity",
        "",
        "這個資料夾是單層剪枝敏感度分析：每次只剪一個 convolution layer，只用 validation accuracy 排名。",
        "",
        "## 設定",
        "",
        f"- dataset: `{args.dataset}`",
        f"- models: `{', '.join(models)}`",
        f"- mode: `{args.mode}`",
        f"- ratios: `{', '.join(str(r) for r in args.ratios)}`",
        f"- epochs for dense model: `{args.epochs}`",
        f"- CSV: `{csv_path.name}`",
        "",
        "## 怎麼看",
        "",
        "- `acc_drop_pp` 越大，代表該層越敏感，剪那一層 accuracy 掉越多。",
        "- 曲線圖是不同 prune ratio 下，每一層的 accuracy drop。",
        "- heatmap 顏色越深，代表該 layer/ratio 組合越危險。",
        "- 如果要模仿 *Pruning Filters for Efficient ConvNets*，可用 `--mode filter_l1`。如果要對應目前 cross-filter 實驗，用 `--mode cross_filter`。",
        "",
        "## 圖表",
        "",
    ]
    for model in models:
        text.append(f"- `{model}_layer_sensitivity.png`")
        text.append(f"- `{model}_layer_sensitivity_heatmap.png`")
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["cifar10", "animals10", "imagenette", "imagefolder"], default="cifar10")
    p.add_argument("--models", default="resnet34", help="Comma-separated model list or all.")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--animals-dir", default=None)
    p.add_argument("--imagenette-dir", default=None)
    p.add_argument("--imagefolder-dir", default=None)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--ratios", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7")
    p.add_argument("--mode", choices=["cross_filter", "filter_l1"], default="cross_filter")
    p.add_argument("--criterion", default="top4_mean_abs")
    p.add_argument("--tm", type=int, default=8)
    p.add_argument("--only-first-residual-conv", action="store_true", help="For ResNets, only test conv1 in each residual block, like the paper figures.")
    p.add_argument("--no-prune-1x1", action="store_true")
    p.add_argument("--skip-depthwise", action="store_true")
    p.add_argument("--train-limit", type=int, default=None)
    p.add_argument("--test-limit", type=int, default=None)
    p.add_argument("--val-limit", type=int, default=None)
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--no-cudnn-benchmark", action="store_true")
    p.add_argument("--save-dir", default="runs/layer_sensitivity")
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint-dir", default="runs/layer_sensitivity/checkpoints")
    p.add_argument("--dense-checkpoint-in", default=None)
    p.add_argument("--load-checkpoint", action="store_true")
    p.add_argument("--save-checkpoint", action="store_true")
    args = p.parse_args()
    args.ratios = parse_ratios(args.ratios)
    return args


def build_loaders(args):
    pin_memory = args.pin_memory
    if args.dataset == "cifar10":
        return build_cifar10(
            args.data_dir,
            args.batch_size,
            args.image_size,
            args.train_limit,
            args.val_limit,
            args.test_limit,
            args.num_workers,
            pin_memory,
            args.split_seed,
            args.seed,
            args.val_size,
        )
    if args.dataset == "animals10":
        if args.animals_dir is None:
            raise ValueError("--animals-dir is required for --dataset animals10")
        return build_animals10(args.animals_dir, args.batch_size, args.image_size, args.train_limit, args.test_limit, args.num_workers, pin_memory)
    if args.dataset == "imagenette":
        if args.imagenette_dir is None:
            raise ValueError("--imagenette-dir is required for --dataset imagenette")
        return build_imagefolder_dataset(args.imagenette_dir, args.batch_size, args.image_size, args.train_limit, args.test_limit, args.num_workers, pin_memory)
    if args.imagefolder_dir is None:
        raise ValueError("--imagefolder-dir is required for --dataset imagefolder")
    return build_imagefolder_dataset(args.imagefolder_dir, args.batch_size, args.image_size, args.train_limit, args.test_limit, args.num_workers, pin_memory)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    if device.type == "cuda" and not args.no_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    use_amp = args.amp_dtype if args.amp and device.type == "cuda" else False
    models = parse_models(args.models)
    (train_loader, val_loader, test_loader), num_classes, split_metadata = build_loaders(args)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.save_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for model_name in models:
        print(f"=== Layer sensitivity: {model_name} ===")
        dense_model, dense_acc, ckpt = train_or_load_dense(
            args,
            model_name,
            train_loader,
            val_loader,
            num_classes,
            split_metadata,
            device,
            use_amp,
        )
        rows = sensitivity_rows(args, model_name, dense_model, dense_acc, val_loader, device, use_amp)
        for row in rows:
            row["checkpoint"] = ckpt
        all_rows.extend(rows)
        plot_model(rows, out_dir, model_name)
        del dense_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_path = out_dir / "layer_sensitivity.csv"
    ranked_path = out_dir / "layer_sensitivity_ranked.csv"
    write_csv(all_rows, csv_path)
    write_csv(rank_sensitivity_rows(all_rows), ranked_path)
    write_readme(out_dir / "README.md", args, csv_path, all_rows)
    print(f"Wrote {csv_path}")
    print(f"Wrote {ranked_path}")
    print(f"Wrote charts to {out_dir}")


if __name__ == "__main__":
    main()
