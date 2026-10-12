import argparse
import csv
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from resnet34_pruning_experiments import make_model, parse_models


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def checkpoint_path(root, model_name, seed):
    root = Path(root)
    if model_name == "resnet34":
        return root / "fixed_checkpoint_resnet34" / "checkpoints" / f"resnet34_seed{seed}_dense.pt"
    return root / "fixed_checkpoint_multi_model" / "checkpoints" / f"{model_name}_seed{seed}_dense.pt"


def load_checkpoint(path, model_name, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = make_model(model_name, payload["num_classes"], payload["image_size"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval(), payload


def make_loaders(data_dir, image_size, batch_size, workers, calibration_samples):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ]
    )
    train_set = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
    indices = torch.randperm(len(train_set), generator=torch.Generator().manual_seed(0))
    calibration_set = Subset(train_set, indices[:calibration_samples].tolist())
    kwargs = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": True,
    }
    return (
        DataLoader(calibration_set, shuffle=False, **kwargs),
        DataLoader(test_set, shuffle=False, **kwargs),
    )


def quantized_modules(model):
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    }


@torch.inference_mode()
def calibrate_activation_maxima(model, loader, device):
    maxima = {
        name: torch.zeros((), device=device)
        for name in quantized_modules(model)
    }
    handles = []

    def make_hook(name):
        def hook(_module, inputs):
            maxima[name] = torch.maximum(maxima[name], inputs[0].detach().abs().amax())

        return hook

    for name, module in quantized_modules(model).items():
        handles.append(module.register_forward_pre_hook(make_hook(name)))
    for images, _ in loader:
        model(images.to(device, non_blocking=True))
    for handle in handles:
        handle.remove()
    return {name: float(value.item()) for name, value in maxima.items()}


def signed_qmax(bits):
    if bits < 2:
        raise ValueError("Quantization bit width must be at least 2")
    return (1 << (bits - 1)) - 1


def fake_quantize_symmetric(tensor, scale, qmax):
    if not torch.is_tensor(scale) and scale == 0:
        return tensor
    return torch.clamp(torch.round(tensor / scale), -qmax, qmax) * scale


@torch.no_grad()
def quantize_weights_(model, bits):
    qmax = signed_qmax(bits)
    for module in quantized_modules(model).values():
        weight = module.weight
        reduce_dims = tuple(range(1, weight.ndim))
        channel_max = weight.abs().amax(dim=reduce_dims, keepdim=True)
        scale = torch.where(channel_max > 0, channel_max / qmax, torch.ones_like(channel_max))
        module.weight.copy_(fake_quantize_symmetric(weight, scale, qmax))


def install_activation_fake_quant(model, maxima, bits):
    qmax = signed_qmax(bits)
    handles = []

    def make_hook(name):
        scale = maxima[name] / qmax if maxima[name] > 0 else 0.0

        def hook(_module, inputs):
            quantized = fake_quantize_symmetric(inputs[0], scale, qmax)
            return (quantized, *inputs[1:])

        return hook

    for name, module in quantized_modules(model).items():
        handles.append(module.register_forward_pre_hook(make_hook(name)))
    return handles


@torch.inference_mode()
def evaluate(model, loader, device):
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        predictions = model(images).argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += labels.numel()
    return correct / total


def save_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Hardware-oriented PTQ accuracy simulation")
    parser.add_argument("--models", default="resnet34,resnet50,googlenet,densenet121,mobilenet_v2")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--bits", default="8,9,10")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--save-dir", default="runs/quantization_dense_ptq")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--calibration-samples", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    models = parse_models(args.models)
    seeds = parse_int_list(args.seeds)
    bits_list = parse_int_list(args.bits)
    rows = []
    loaders = {}

    for model_name in models:
        for seed in seeds:
            path = checkpoint_path(args.runs_dir, model_name, seed)
            base_model, payload = load_checkpoint(path, model_name, device)
            image_size = int(payload["image_size"])
            if image_size not in loaders:
                loaders[image_size] = make_loaders(
                    args.data_dir,
                    image_size,
                    args.batch_size,
                    args.num_workers,
                    args.calibration_samples,
                )
            calibration_loader, test_loader = loaders[image_size]
            fp32_acc = evaluate(base_model, test_loader, device)
            maxima = calibrate_activation_maxima(base_model, calibration_loader, device)
            print(f"{model_name} seed={seed} FP32={fp32_acc:.2%}", flush=True)

            for bits in bits_list:
                model, _ = load_checkpoint(path, model_name, device)
                quantize_weights_(model, bits)
                handles = install_activation_fake_quant(model, maxima, bits)
                quantized_acc = evaluate(model, test_loader, device)
                for handle in handles:
                    handle.remove()
                loss_pp = 100.0 * (fp32_acc - quantized_acc)
                print(
                    f"  INT{bits}={quantized_acc:.2%}, loss={loss_pp:+.3f} pp",
                    flush=True,
                )
                rows.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "checkpoint_id": payload.get("checkpoint_id", path.stem),
                        "weight_bits": bits,
                        "activation_bits": bits,
                        "weight_quantization": "per_output_channel_symmetric",
                        "activation_quantization": "per_layer_symmetric_static_ptq",
                        "calibration_samples": args.calibration_samples,
                        "fp32_accuracy": fp32_acc,
                        "quantized_accuracy": quantized_acc,
                        "accuracy_loss_pp": loss_pp,
                    }
                )
            del base_model
            torch.cuda.empty_cache()

    save_rows(Path(args.save_dir) / "results.csv", rows)
    print(f"Saved {len(rows)} rows to {Path(args.save_dir) / 'results.csv'}")


if __name__ == "__main__":
    main()
