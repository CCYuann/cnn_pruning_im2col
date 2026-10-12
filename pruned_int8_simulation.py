import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

import resnet34_pruning_experiments as pruning
from quantization_experiments import checkpoint_path, make_loaders


@torch.inference_mode()
def evaluate_cpu(model, loader):
    correct = 0
    total = 0
    for images, labels in loader:
        correct += int((model(images).argmax(dim=1) == labels).sum().item())
        total += labels.numel()
    return correct / total


@torch.inference_mode()
def fx_int8_ptq(model, calibration_loader):
    torch.backends.quantized.engine = "onednn"
    prepared = prepare_fx(
        model.cpu().eval(),
        get_default_qconfig_mapping("onednn"),
        (torch.randn(1, 3, 32, 32),),
    )
    for images, _ in calibration_loader:
        prepared(images)
    return convert_fx(prepared)


def save_result(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Cross-filter pruning plus FX INT8 PTQ")
    parser.add_argument("--model", default="resnet34")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prune-ratio", type=float, default=0.8)
    parser.add_argument("--tm", type=int, default=8)
    parser.add_argument(
        "--filter-grouping",
        choices=("contiguous", "similarity_jaccard"),
        default="similarity_jaccard",
    )
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--finetune-lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--calibration-samples", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--save-dir", default="runs/pruned_int8_simulation")
    parser.add_argument(
        "--save-pruned-checkpoint",
        default=None,
        help="Optional path for the fine-tuned pruned model and hardware masks.",
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    checkpoint = checkpoint_path(args.runs_dir, args.model, args.seed)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = pruning.make_model(args.model, payload["num_classes"], payload["image_size"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model = model.to(device)

    (train_loader, val_loader, test_loader), _, _ = pruning.build_cifar10(
        args.data_dir,
        args.batch_size,
        payload["image_size"],
        None,
        None,
        None,
        args.num_workers,
        True,
        0,
        args.seed,
        5000,
    )
    _, dense_accuracy = pruning.evaluate(model, test_loader, device, use_amp="float16")
    specs = pruning.collect_conv_specs(model, payload["image_size"])
    reference = pruning.global_unstructured_reference(
        specs,
        tm=args.tm,
        target_sparsity=args.prune_ratio,
        include_1x1=True,
        filter_grouping=args.filter_grouping,
    )
    stats = pruning.cross_filter_prune_(
        model,
        specs,
        tm=args.tm,
        prune_ratio=args.prune_ratio,
        criterion="l1_sum",
        include_1x1=True,
        ratio_source="global_cluster",
        sparsity_reference=reference,
        macs_per_pe=8,
        filter_grouping=args.filter_grouping,
    )
    _, direct_accuracy = pruning.evaluate(model, test_loader, device, use_amp="float16")
    pruning.reset_training_rng(args.seed + 100000, train_loader)
    train_args = SimpleNamespace(
        finetune_lr=args.finetune_lr,
        weight_decay=5e-4,
        finetune_epochs=args.finetune_epochs,
        tm=args.tm,
    )
    pruning.fine_tune(
        model,
        stats,
        train_loader,
        val_loader,
        train_args,
        device,
        f"{args.model} {args.filter_grouping} l1 r{args.prune_ratio:g}",
        "float16",
    )
    _, pruned_fp32_accuracy = pruning.evaluate(
        model,
        test_loader,
        device,
        use_amp="float16",
    )

    if args.save_pruned_checkpoint:
        checkpoint_output = Path(args.save_pruned_checkpoint)
        checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
        stats_by_name = {stat.name: stat for stat in stats}
        pruning_layers = {}
        for spec in specs:
            if spec.name not in stats_by_name:
                continue
            stat = stats_by_name[spec.name]
            pruning_layers[spec.name] = {
                "full_mask": stat.full_mask.cpu(),
                "cluster_masks": [mask.cpu() for mask in stat.masks],
                "filter_groups": reference[spec.name]["filter_groups"],
            }
        torch.save(
            {
                "model": args.model,
                "model_state_dict": {
                    name: value.detach().cpu() for name, value in model.state_dict().items()
                },
                "num_classes": payload["num_classes"],
                "image_size": payload["image_size"],
                "source_checkpoint": str(checkpoint),
                "source_checkpoint_id": payload.get("checkpoint_id", checkpoint.stem),
                "prune_ratio": args.prune_ratio,
                "tm": args.tm,
                "criterion": "l1_sum",
                "ratio_source": "global_cluster",
                "filter_grouping": args.filter_grouping,
                "finetune_epochs": args.finetune_epochs,
                "pruned_fp32_accuracy": pruned_fp32_accuracy,
                "pruning_layers": pruning_layers,
            },
            checkpoint_output,
        )
        print(f"Saved pruned checkpoint to {checkpoint_output}", flush=True)

    calibration_loader, cpu_test_loader = make_loaders(
        args.data_dir,
        payload["image_size"],
        256,
        max(args.num_workers, 2),
        args.calibration_samples,
    )
    quantized_model = fx_int8_ptq(model, calibration_loader)
    int8_accuracy = evaluate_cpu(quantized_model, cpu_test_loader)
    summary = pruning.summarize(stats)
    row = {
        "model": args.model,
        "seed": args.seed,
        "checkpoint_id": payload.get("checkpoint_id", checkpoint.stem),
        "prune_ratio_target": args.prune_ratio,
        "prune_ratio_actual": summary["pruning_ratio"],
        "filter_grouping": args.filter_grouping,
        "criterion": "l1_sum",
        "finetune_epochs": args.finetune_epochs,
        "calibration_samples": args.calibration_samples,
        "dense_fp32_accuracy": dense_accuracy,
        "pruned_direct_accuracy": direct_accuracy,
        "pruned_finetuned_fp32_accuracy": pruned_fp32_accuracy,
        "pruned_finetuned_int8_accuracy": int8_accuracy,
        "int8_loss_vs_pruned_fp32_pp": 100.0 * (pruned_fp32_accuracy - int8_accuracy),
    }
    output = Path(args.save_dir) / (
        f"{args.model}_seed{args.seed}_r{str(args.prune_ratio).replace('.', 'p')}_results.csv"
    )
    save_result(output, row)
    print("Final result:")
    for key, value in row.items():
        print(f"  {key}: {value}")
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
