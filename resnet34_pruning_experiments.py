import argparse
import copy
import csv
import hashlib
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from torchvision.models import densenet121, efficientnet_v2_m, googlenet, mobilenet_v2, resnet34, resnet50


@dataclass
class LayerSpec:
    name: str
    module: nn.Conv2d
    input_hw: Tuple[int, int]


@dataclass
class LayerStats:
    name: str
    method: str
    kernel: str
    original_weights: int
    nonzero_weights: int
    original_macs: int
    pruned_macs: int
    original_cycles: int
    pruned_cycles: int
    pe_utilization: float
    cycle_pe_utilization: float
    avg_work_per_pe: float
    std_work_per_pe: float
    masks: List[torch.Tensor]
    full_mask: Optional[torch.Tensor] = None


@dataclass
class Result:
    name: str
    direct_acc: float
    finetuned_acc: Optional[float]
    stats: List[LayerStats]
    criterion: str = ""
    ratio_source: str = ""


SUPPORTED_MODELS = [
    "resnet34",
    "resnet50",
    "googlenet",
    "densenet121",
    "mobilenet_v2",
    "efficientnet_v2_m",
]


def adapt_small_image_stem(model, model_name):
    # CIFAR/Animals-10 images are smaller than ImageNet. These stem changes
    # keep more spatial detail and avoid collapsing 32x32 inputs too early.
    if model_name in {"resnet34", "resnet50"}:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    elif model_name == "googlenet":
        model.conv1.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool1 = nn.Identity()
    elif model_name == "densenet121":
        model.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.features.pool0 = nn.Identity()
    return model


def make_model(model_name, num_classes, image_size):
    if model_name == "resnet34":
        model = resnet34(weights=None, num_classes=num_classes)
    elif model_name == "resnet50":
        model = resnet50(weights=None, num_classes=num_classes)
    elif model_name == "googlenet":
        model = googlenet(weights=None, aux_logits=False, init_weights=True, num_classes=num_classes)
    elif model_name == "densenet121":
        model = densenet121(weights=None, num_classes=num_classes)
    elif model_name == "mobilenet_v2":
        model = mobilenet_v2(weights=None, num_classes=num_classes)
    elif model_name == "efficientnet_v2_m":
        model = efficientnet_v2_m(weights=None, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choices: {SUPPORTED_MODELS}")

    if image_size <= 64:
        model = adapt_small_image_stem(model, model_name)
    return model


def make_subset(dataset, limit):
    if limit is None or limit >= len(dataset):
        return dataset
    return Subset(dataset, range(limit))


def build_cifar10(
    data_dir,
    batch_size,
    image_size,
    train_limit,
    val_limit,
    test_limit,
    num_workers,
    pin_memory,
    split_seed,
    loader_seed,
    val_size,
):
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    train_full = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
    val_full = datasets.CIFAR10(data_dir, train=True, download=True, transform=test_tf)
    test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_tf)

    if val_size <= 0 or val_size >= len(train_full):
        raise ValueError(f"--val-size must be between 1 and {len(train_full) - 1}")
    split_generator = torch.Generator().manual_seed(split_seed)
    indices = torch.randperm(len(train_full), generator=split_generator)
    val_indices = indices[:val_size].tolist()
    train_indices = indices[val_size:].tolist()
    if train_limit is not None:
        train_indices = train_indices[:train_limit]
    if val_limit is not None:
        val_indices = val_indices[:val_limit]

    train_set = Subset(train_full, train_indices)
    val_set = Subset(val_full, val_indices)
    test_set = make_subset(test_set, test_limit)
    split_bytes = np.asarray(indices.tolist(), dtype=np.int32).tobytes()
    split_metadata = {
        "split_seed": split_seed,
        "val_size": val_size,
        "train_size": len(train_set),
        "validation_size": len(val_set),
        "test_size": len(test_set),
        "split_id": hashlib.sha256(split_bytes).hexdigest()[:16],
    }
    loaders = make_loaders(
        train_set,
        val_set,
        test_set,
        batch_size,
        num_workers,
        pin_memory,
        loader_seed,
    )
    return loaders, 10, split_metadata


def build_animals10(
    root,
    batch_size,
    image_size,
    train_limit,
    test_limit,
    num_workers,
    pin_memory,
):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"Animals-10 path not found: {root}. Expected ImageFolder data."
        )

    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    train_dir = root / "train"
    test_dir = root / "test"
    val_dir = root / "val"

    if train_dir.exists() and (test_dir.exists() or val_dir.exists()):
        eval_dir = test_dir if test_dir.exists() else val_dir
        train_set = datasets.ImageFolder(train_dir, transform=train_tf)
        test_set = datasets.ImageFolder(eval_dir, transform=test_tf)
    else:
        full_train = datasets.ImageFolder(root, transform=train_tf)
        full_test = datasets.ImageFolder(root, transform=test_tf)
        train_len = int(0.8 * len(full_train))
        test_len = len(full_train) - train_len
        generator = torch.Generator().manual_seed(0)
        train_indices, test_indices = random_split(
            range(len(full_train)),
            [train_len, test_len],
            generator=generator,
        )
        train_set = Subset(full_train, list(train_indices))
        test_set = Subset(full_test, list(test_indices))

    train_set = make_subset(train_set, train_limit)
    test_set = make_subset(test_set, test_limit)
    num_classes = len(train_set.dataset.classes) if isinstance(train_set, Subset) else len(train_set.classes)
    loaders = make_loaders(
        train_set,
        test_set,
        test_set,
        batch_size,
        num_workers,
        pin_memory,
    )
    split_metadata = {
        "split_seed": 0,
        "val_size": len(test_set),
        "train_size": len(train_set),
        "validation_size": len(test_set),
        "test_size": len(test_set),
        "split_id": "external-eval",
    }
    return loaders, num_classes, split_metadata


def build_imagefolder_dataset(
    root,
    batch_size,
    image_size,
    train_limit,
    test_limit,
    num_workers,
    pin_memory,
):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"ImageFolder path not found: {root}")

    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    train_dir = root / "train"
    test_dir = root / "test"
    val_dir = root / "val"

    if train_dir.exists() and (test_dir.exists() or val_dir.exists()):
        eval_dir = test_dir if test_dir.exists() else val_dir
        train_set = datasets.ImageFolder(train_dir, transform=train_tf)
        test_set = datasets.ImageFolder(eval_dir, transform=test_tf)
    else:
        full_train = datasets.ImageFolder(root, transform=train_tf)
        full_test = datasets.ImageFolder(root, transform=test_tf)
        train_len = int(0.8 * len(full_train))
        test_len = len(full_train) - train_len
        generator = torch.Generator().manual_seed(0)
        train_indices, test_indices = random_split(
            range(len(full_train)),
            [train_len, test_len],
            generator=generator,
        )
        train_set = Subset(full_train, list(train_indices))
        test_set = Subset(full_test, list(test_indices))

    train_set = make_subset(train_set, train_limit)
    test_set = make_subset(test_set, test_limit)
    num_classes = len(train_set.dataset.classes) if isinstance(train_set, Subset) else len(train_set.classes)
    loaders = make_loaders(
        train_set,
        test_set,
        test_set,
        batch_size,
        num_workers,
        pin_memory,
    )
    split_metadata = {
        "split_seed": 0,
        "val_size": len(test_set),
        "train_size": len(train_set),
        "validation_size": len(test_set),
        "test_size": len(test_set),
        "split_id": "external-eval",
    }
    return loaders, num_classes, split_metadata


def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loaders(
    train_set,
    val_set,
    test_set,
    batch_size,
    num_workers,
    pin_memory,
    loader_seed=0,
):
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    generator = torch.Generator().manual_seed(loader_seed)
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        generator=generator,
        worker_init_fn=seed_worker,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def reset_training_rng(seed, train_loader=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if train_loader is not None:
        generator = getattr(train_loader, "generator", None)
        if generator is not None:
            generator.manual_seed(seed)
        sampler_generator = getattr(getattr(train_loader, "sampler", None), "generator", None)
        if sampler_generator is not None and sampler_generator is not generator:
            sampler_generator.manual_seed(seed)


def autocast_context(device, use_amp):
    if use_amp and device.type == "cuda":
        amp_dtype = torch.bfloat16 if use_amp == "bfloat16" else torch.float16
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        return torch.cuda.amp.autocast(dtype=amp_dtype)
    return nullcontext()


def make_scaler(device, use_amp):
    enabled = bool(use_amp) and use_amp != "bfloat16" and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch,
    prune_stats=None,
    tm=8,
    scaler=None,
    use_amp=False,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for batch_idx, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, use_amp):
            logits = model(x)
            loss = F.cross_entropy(logits, y)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if prune_stats is not None:
                scaler.unscale_(optimizer)
                apply_masks_(model, prune_stats, tm=tm, include_grad=True)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if prune_stats is not None:
                apply_masks_(model, prune_stats, tm=tm, include_grad=True)
            optimizer.step()

        if prune_stats is not None:
            apply_masks_(model, prune_stats, tm=tm)

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
        if batch_idx % 100 == 0:
            print(
                f"Epoch {epoch} batch {batch_idx:4d}: "
                f"loss={total_loss / total:.4f}, acc={total_correct / total:.2%}"
            )

    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, device, use_amp=False):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast_context(device, use_amp):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, total_correct / total


def collect_conv_specs(model, image_size):
    specs = []
    hooks = []

    def make_hook(name, module):
        def hook(_module, inputs, _output):
            _, _, h, w = inputs[0].shape
            specs.append(LayerSpec(name, module, (h, w)))

        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(make_hook(name, module)))

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        model(torch.zeros(1, 3, image_size, image_size, device=device))
    if was_training:
        model.train()
    for hook in hooks:
        hook.remove()
    return specs


def output_hw(input_hw, conv):
    h, w = input_hw
    kh, kw = conv.kernel_size
    sh, sw = conv.stride
    ph, pw = conv.padding
    dh, dw = conv.dilation
    out_h = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    out_w = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    return out_h, out_w


def compute_importance(cluster, criterion):
    abs_cluster = cluster.abs()
    if criterion == "l1_sum":
        return abs_cluster.sum(dim=0)
    if criterion == "l2_sum":
        return torch.sqrt((cluster * cluster).sum(dim=0))
    if criterion == "max_abs":
        return abs_cluster.max(dim=0).values
    if criterion == "top2_mean_abs":
        k = min(2, abs_cluster.shape[0])
        return abs_cluster.topk(k, dim=0).values.mean(dim=0)
    if criterion == "top4_mean_abs":
        k = min(4, abs_cluster.shape[0])
        return abs_cluster.topk(k, dim=0).values.mean(dim=0)
    if criterion == "min_abs":
        return abs_cluster.min(dim=0).values
    if criterion == "random":
        return torch.rand(cluster.shape[1:], dtype=cluster.dtype, device=cluster.device)
    raise ValueError(f"Unknown criterion: {criterion}")


@torch.no_grad()
def apply_masks_(model, stats, tm, include_grad=False):
    modules = dict(model.named_modules())
    for stat in stats:
        layer = modules[stat.name]
        if stat.full_mask is None:
            expanded_masks = []
            for cluster_id, mask in enumerate(stat.masks):
                start = cluster_id * tm
                end = min(start + tm, layer.weight.shape[0])
                mask = mask.to(device=layer.weight.device, dtype=layer.weight.dtype)
                if mask.dim() == 3:
                    mask = mask.unsqueeze(0).expand(end - start, -1, -1, -1)
                expanded_masks.append(mask)
            stat.full_mask = torch.cat(expanded_masks, dim=0)

        if (
            stat.full_mask.device != layer.weight.device
            or stat.full_mask.dtype != layer.weight.dtype
        ):
            stat.full_mask = stat.full_mask.to(
                device=layer.weight.device,
                dtype=layer.weight.dtype,
            )

        layer.weight.mul_(stat.full_mask)
        if include_grad and layer.weight.grad is not None:
            layer.weight.grad.mul_(stat.full_mask)


def split_by_cluster(full_mask, tm):
    masks = []
    for start in range(0, full_mask.shape[0], tm):
        end = min(start + tm, full_mask.shape[0])
        masks.append(full_mask[start:end].detach().cpu())
    return masks


def contiguous_filter_groups(out_channels, tm):
    return [list(range(start, min(start + tm, out_channels))) for start in range(0, out_channels, tm)]


def filter_group_jaccard(keep_mask, groups):
    """Mean pairwise Jaccard similarity inside filter groups."""
    flat = keep_mask.reshape(keep_mask.shape[0], -1).bool().float()
    scores = []
    for group in groups:
        if len(group) < 2:
            continue
        group_mask = flat[group]
        intersection = group_mask @ group_mask.t()
        counts = group_mask.sum(dim=1)
        union = counts[:, None] + counts[None, :] - intersection
        similarity = torch.where(union > 0, intersection / union, torch.ones_like(union))
        upper = torch.triu_indices(len(group), len(group), offset=1, device=flat.device)
        scores.append(similarity[upper[0], upper[1]])
    if not scores:
        return 1.0
    return float(torch.cat(scores).mean().item())


def similarity_filter_groups(keep_mask, tm):
    """Greedily group filters with similar global-pruning keep masks.

    Filters that want to retain the same K positions are natural candidates for
    sharing one cross-filter mask. The grouping is deterministic and does not
    permute model channels; it only records which filters share a mask.
    """
    out_channels = keep_mask.shape[0]
    if out_channels <= tm:
        return [list(range(out_channels))]

    flat = keep_mask.reshape(out_channels, -1).bool().float()
    intersection = flat @ flat.t()
    counts = flat.sum(dim=1)
    union = counts[:, None] + counts[None, :] - intersection
    similarity = torch.where(union > 0, intersection / union, torch.ones_like(union))
    similarity.fill_diagonal_(-1.0)

    # Handle difficult-to-match filters first so their nearest neighbors are
    # not consumed by easier groups. Ties retain the original channel order.
    difficulty = similarity.max(dim=1).values.detach().cpu().tolist()
    seed_order = sorted(range(out_channels), key=lambda index: (difficulty[index], index))
    remaining = set(range(out_channels))
    groups = []
    for seed in seed_order:
        if seed not in remaining:
            continue
        group = [seed]
        remaining.remove(seed)
        while remaining and len(group) < tm:
            candidates = sorted(remaining)
            candidate_tensor = torch.tensor(candidates, device=similarity.device)
            member_tensor = torch.tensor(group, device=similarity.device)
            scores = similarity[member_tensor][:, candidate_tensor].mean(dim=0)
            best_offset = int(torch.argmax(scores).item())
            selected = candidates[best_offset]
            group.append(selected)
            remaining.remove(selected)
        groups.append(sorted(group))
    return groups


def make_filter_groups(full_mask, tm, grouping):
    if grouping == "contiguous":
        return contiguous_filter_groups(full_mask.shape[0], tm)
    if grouping == "similarity_jaccard":
        return similarity_filter_groups(full_mask, tm)
    raise ValueError(f"Unknown filter grouping: {grouping}")


def macs_from_masks(masks, out_channels, out_hw, tm):
    out_h, out_w = out_hw
    macs = 0
    for cluster_id, mask in enumerate(masks):
        start = cluster_id * tm
        end = min(start + tm, out_channels)
        filters = end - start
        if mask.dim() == 3:
            macs += int(mask.sum().item()) * filters * out_h * out_w
        else:
            macs += int(mask.sum().item()) * out_h * out_w
    return macs


def workload_balance(masks):
    useful = 0.0
    capacity = 0.0
    works = []
    for mask in masks:
        if mask.dim() == 3:
            work = torch.full((1,), float(mask.sum().item()))
        else:
            work = mask.reshape(mask.shape[0], -1).sum(dim=1).float()
        if work.numel() == 0:
            continue
        works.append(work)
        useful += float(work.sum().item())
        capacity += float(work.max().item()) * work.numel()

    if not works or capacity == 0:
        return 1.0, 0.0, 0.0

    all_work = torch.cat(works)
    return useful / capacity, float(all_work.mean().item()), float(all_work.std(unbiased=False).item())


def paper_cycle_estimate(masks, out_channels, positions, out_hw, tm, macs_per_pe):
    """Estimate paper-style PE-cluster cycles for weight-stationary execution.

    Each cluster maps up to ``tm`` filters to PEs. For cross-filter masks, all
    PEs share the same kept positions and skip synchronously. For unstructured
    masks, the cluster must process the union of nonzero positions, so PEs with
    zeros at a broadcast position sit idle.
    """
    out_pixels = out_hw[0] * out_hw[1]
    pixel_chunks = (out_pixels + macs_per_pe - 1) // macs_per_pe
    original_cycles = ((out_channels + tm - 1) // tm) * positions * pixel_chunks

    pruned_cycles = 0
    useful_macs = 0.0
    capacity_macs = 0.0
    for cluster_id, mask in enumerate(masks):
        start = cluster_id * tm
        end = min(start + tm, out_channels)
        filters = end - start
        if filters <= 0:
            continue

        if mask.dim() == 3:
            active_positions = int(mask.sum().item())
            useful_positions = active_positions * filters
        else:
            cluster_mask = mask.reshape(filters, -1).bool()
            active_positions = int(cluster_mask.any(dim=0).sum().item())
            useful_positions = int(cluster_mask.sum().item())

        pruned_cycles += active_positions * pixel_chunks
        useful_macs += useful_positions * out_pixels
        capacity_macs += active_positions * pixel_chunks * macs_per_pe * filters

    cycle_util = useful_macs / capacity_macs if capacity_macs else 1.0
    return original_cycles, pruned_cycles, cycle_util


def build_stats(
    name,
    method,
    spec,
    masks,
    original_weights,
    nonzero_weights,
    original_macs,
    pruned_macs,
    tm,
    macs_per_pe,
):
    util, avg_work, std_work = workload_balance(masks)
    kh, kw = spec.module.kernel_size
    out_ch = spec.module.weight.shape[0]
    positions = spec.module.weight.shape[1] * kh * kw
    out_hw = output_hw(spec.input_hw, spec.module)
    original_cycles, pruned_cycles, cycle_util = paper_cycle_estimate(
        masks, out_ch, positions, out_hw, tm, macs_per_pe
    )
    return LayerStats(
        name=name,
        method=method,
        kernel=f"{kh}x{kw}",
        original_weights=original_weights,
        nonzero_weights=nonzero_weights,
        original_macs=original_macs,
        pruned_macs=pruned_macs,
        original_cycles=original_cycles,
        pruned_cycles=pruned_cycles,
        pe_utilization=util,
        cycle_pe_utilization=cycle_util,
        avg_work_per_pe=avg_work,
        std_work_per_pe=std_work,
        masks=masks,
    )


@torch.no_grad()
def global_unstructured_reference(
    specs,
    tm,
    target_sparsity,
    include_1x1=True,
    filter_grouping="contiguous",
):
    """
    Build the paper-style sparsity reference.

    The paper first applies global unstructured pruning, then uses the resulting
    sparsity of similar layers/clusters as the reference for cross-filter
    structured pruning.
    """
    prunable_specs = [
        spec for spec in specs if include_1x1 or spec.module.kernel_size != (1, 1)
    ]
    all_weights = torch.cat([spec.module.weight.abs().reshape(-1) for spec in prunable_specs])
    num_prune = min(max(round(target_sparsity * all_weights.numel()), 0), all_weights.numel())
    flat_mask = torch.ones_like(all_weights)
    if num_prune > 0:
        flat_mask[torch.argsort(all_weights)[:num_prune]] = 0

    reference = {}
    offset = 0
    for spec in prunable_specs:
        weight = spec.module.weight
        original_weights = weight.numel()
        next_offset = offset + original_weights
        full_mask = flat_mask[offset:next_offset].reshape_as(weight)
        offset = next_offset

        filter_groups = make_filter_groups(full_mask, tm, filter_grouping)
        cluster_ratios = []
        for group in filter_groups:
            cluster_mask = full_mask[group]
            cluster_ratios.append(1 - float(cluster_mask.sum().item()) / cluster_mask.numel())

        contiguous_groups = contiguous_filter_groups(weight.shape[0], tm)

        reference[spec.name] = {
            "full_mask": full_mask.detach().cpu(),
            "layer_ratio": 1 - float(full_mask.sum().item()) / full_mask.numel(),
            "cluster_ratios": cluster_ratios,
            "filter_groups": filter_groups,
            "group_similarity": filter_group_jaccard(full_mask, filter_groups),
            "contiguous_group_similarity": filter_group_jaccard(full_mask, contiguous_groups),
        }
    return reference


def cluster_prune_ratio(spec_name, cluster_id, default_ratio, ratio_source, sparsity_reference):
    if ratio_source == "uniform":
        return default_ratio
    if sparsity_reference is None or spec_name not in sparsity_reference:
        return default_ratio
    if ratio_source == "global_layer":
        return sparsity_reference[spec_name]["layer_ratio"]
    if ratio_source == "global_cluster":
        return sparsity_reference[spec_name]["cluster_ratios"][cluster_id]
    raise ValueError(f"Unknown cross-filter ratio source: {ratio_source}")


def apply_ratio_limits(local_ratio, max_layer_prune_ratio=None, min_layer_keep_ratio=None, protected_prune_ratio=None):
    ratio = float(local_ratio)
    caps = []
    if max_layer_prune_ratio is not None:
        caps.append(float(max_layer_prune_ratio))
    if min_layer_keep_ratio is not None:
        caps.append(1.0 - float(min_layer_keep_ratio))
    if protected_prune_ratio is not None:
        caps.append(float(protected_prune_ratio))
    if caps:
        ratio = min(ratio, min(caps))
    return min(max(ratio, 0.0), 1.0)


def load_sensitivity_layer_caps(
    sensitivity_csv,
    model_name,
    criterion,
    protect_top_k,
    protected_prune_ratio,
    skip_top_k=0,
    min_mean_drop_pp=0.0,
):
    if not sensitivity_csv or protect_top_k <= 0:
        return {}, ""

    csv_path = Path(sensitivity_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Sensitivity CSV not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    candidates = [row for row in rows if row.get("model") == model_name]
    if not candidates:
        return {}, ""

    criteria = {row.get("criterion", "") for row in candidates}
    selected_criterion = criterion if criterion in criteria else "filter_l1"
    filtered = [
        row
        for row in candidates
        if row.get("criterion") == selected_criterion
        and float(row.get("mean_drop_pp", 0.0) or 0.0) >= min_mean_drop_pp
    ]
    filtered.sort(key=lambda row: float(row.get("mean_drop_pp", 0.0) or 0.0), reverse=True)

    caps = {}
    for index, row in enumerate(filtered[:protect_top_k]):
        layer_name = row["layer_name"]
        caps[layer_name] = 0.0 if index < skip_top_k else protected_prune_ratio

    if caps:
        preview = ", ".join(f"{name}@{cap:g}" for name, cap in list(caps.items())[:8])
        if len(caps) > 8:
            preview += ", ..."
        print(
            f"Sensitivity-aware layer caps for {model_name}/{selected_criterion}: "
            f"{preview}"
        )
    return caps, selected_criterion


@torch.no_grad()
def cross_filter_prune_(
    model,
    specs,
    tm,
    prune_ratio,
    criterion,
    include_1x1=True,
    ratio_source="global_cluster",
    sparsity_reference=None,
    macs_per_pe=8,
    max_layer_prune_ratio=None,
    min_layer_keep_ratio=None,
    protect_early_layers=0,
    protected_prune_ratio=0.2,
    layer_prune_caps=None,
    filter_grouping="contiguous",
):
    stats = []
    prunable_layer_index = 0
    for spec in specs:
        conv = spec.module
        kh, kw = conv.kernel_size
        if (kh, kw) == (1, 1) and not include_1x1:
            continue

        weight = conv.weight
        out_ch, in_ch, kh, kw = weight.shape
        positions = in_ch * kh * kw
        masks = []
        full_mask = torch.ones_like(weight)

        if (
            sparsity_reference is not None
            and spec.name in sparsity_reference
            and "filter_groups" in sparsity_reference[spec.name]
        ):
            filter_groups = sparsity_reference[spec.name]["filter_groups"]
        else:
            filter_groups = make_filter_groups(weight, tm, filter_grouping)

        protected_cap = protected_prune_ratio if prunable_layer_index < protect_early_layers else None
        sensitivity_cap = None if layer_prune_caps is None else layer_prune_caps.get(spec.name)
        if protected_cap is not None and sensitivity_cap is not None:
            protected_cap = min(protected_cap, sensitivity_cap)
        elif sensitivity_cap is not None:
            protected_cap = sensitivity_cap

        for cluster_id, group in enumerate(filter_groups):
            group_index = torch.tensor(group, device=weight.device)
            cluster = weight.index_select(0, group_index)
            local_ratio = cluster_prune_ratio(
                spec.name,
                cluster_id,
                prune_ratio,
                ratio_source,
                sparsity_reference,
            )
            local_ratio = apply_ratio_limits(
                local_ratio,
                max_layer_prune_ratio=max_layer_prune_ratio,
                min_layer_keep_ratio=min_layer_keep_ratio,
                protected_prune_ratio=protected_cap,
            )
            num_prune = min(max(round(local_ratio * positions), 0), positions)
            importance = compute_importance(cluster, criterion).reshape(-1)
            prune_idx = torch.argsort(importance)[:num_prune]
            flat_mask = torch.ones(positions, device=weight.device, dtype=weight.dtype)
            flat_mask[prune_idx] = 0
            mask = flat_mask.reshape(in_ch, kh, kw)
            weight[group_index] = cluster * mask
            full_mask[group_index] = mask.unsqueeze(0).expand(len(group), -1, -1, -1)
            masks.append(mask.detach().cpu())

        out_hw = output_hw(spec.input_hw, conv)
        original_weights = out_ch * in_ch * kh * kw
        nonzero_weights = int(torch.count_nonzero(weight).item())
        original_macs = original_weights * out_hw[0] * out_hw[1]
        pruned_macs = macs_from_masks(masks, out_ch, out_hw, tm)
        stat = build_stats(
                spec.name,
                f"cross_{criterion}_{ratio_source}_{filter_grouping}",
                spec,
                masks,
                original_weights,
                nonzero_weights,
                original_macs,
                pruned_macs,
                tm,
                macs_per_pe,
            )
        stat.full_mask = full_mask.detach().cpu()
        stats.append(stat)
        prunable_layer_index += 1
    return stats


@torch.no_grad()
def global_unstructured_prune_(
    model,
    specs,
    tm,
    target_sparsity,
    include_1x1=True,
    sparsity_reference=None,
    macs_per_pe=8,
):
    prunable_specs = [
        spec for spec in specs if include_1x1 or spec.module.kernel_size != (1, 1)
    ]
    if sparsity_reference is None:
        sparsity_reference = global_unstructured_reference(
            specs,
            tm=tm,
            target_sparsity=target_sparsity,
            include_1x1=include_1x1,
        )

    stats = []
    for spec in prunable_specs:
        conv = spec.module
        weight = conv.weight
        original_weights = weight.numel()
        mask = sparsity_reference[spec.name]["full_mask"].to(
            device=weight.device,
            dtype=weight.dtype,
        )
        weight.mul_(mask)
        masks = split_by_cluster(mask, tm)

        out_hw = output_hw(spec.input_hw, conv)
        nonzero_weights = int(torch.count_nonzero(weight).item())
        original_macs = original_weights * out_hw[0] * out_hw[1]
        pruned_macs = nonzero_weights * out_hw[0] * out_hw[1]
        stats.append(
            build_stats(
                spec.name,
                "global_unstructured",
                spec,
                masks,
                original_weights,
                nonzero_weights,
                original_macs,
                pruned_macs,
                tm,
                macs_per_pe,
            )
        )
    return stats


def summarize(stats):
    original_weights = sum(s.original_weights for s in stats)
    nonzero_weights = sum(s.nonzero_weights for s in stats)
    original_macs = sum(s.original_macs for s in stats)
    pruned_macs = sum(s.pruned_macs for s in stats)
    original_cycles = sum(s.original_cycles for s in stats)
    pruned_cycles = sum(s.pruned_cycles for s in stats)
    cycle_capacity = 0.0
    for stat in stats:
        if stat.cycle_pe_utilization > 0:
            cycle_capacity += stat.pruned_macs / stat.cycle_pe_utilization
    cycle_pe_util = pruned_macs / cycle_capacity if cycle_capacity else 1.0
    capacity = 0.0
    for stat in stats:
        if stat.pe_utilization > 0:
            capacity += stat.pruned_macs / stat.pe_utilization
    pe_util = pruned_macs / capacity if capacity else 1.0
    one_by_one = [s for s in stats if s.kernel == "1x1"]
    spatial = [s for s in stats if s.kernel != "1x1"]
    return {
        "original_weights": original_weights,
        "nonzero_weights": nonzero_weights,
        "pruning_ratio": 1 - nonzero_weights / original_weights if original_weights else 0,
        "original_macs": original_macs,
        "pruned_macs": pruned_macs,
        "mac_reduction": 1 - pruned_macs / original_macs if original_macs else 0,
        "original_cycles": original_cycles,
        "pruned_cycles": pruned_cycles,
        "cycle_reduction": 1 - pruned_cycles / original_cycles if original_cycles else 0,
        "estimated_speedup": original_cycles / pruned_cycles if pruned_cycles else 1.0,
        "pe_util": pe_util,
        "cycle_pe_util": cycle_pe_util,
        "num_1x1": len(one_by_one),
        "num_spatial": len(spatial),
    }


def print_report(title, stats):
    summary = summarize(stats)
    print()
    print(f"=== {title} ===")
    print(f"Pruned conv layers: {len(stats)} ({summary['num_1x1']} are 1x1)")
    print(f"Original conv weights: {summary['original_weights']}")
    print(f"Nonzero conv weights: {summary['nonzero_weights']}")
    print(f"Pruning ratio: {summary['pruning_ratio']:.2%}")
    print(f"Original MACs/image: {summary['original_macs']}")
    print(f"Pruned effective MACs/image: {summary['pruned_macs']}")
    print(f"MAC reduction: {summary['mac_reduction']:.2%}")
    print(f"Paper-style dense cycles/image: {summary['original_cycles']}")
    print(f"Paper-style pruned cycles/image: {summary['pruned_cycles']}")
    print(f"Estimated cycle reduction: {summary['cycle_reduction']:.2%}")
    print(f"Estimated speedup: {summary['estimated_speedup']:.2f}x")
    print(f"Estimated PE utilization: {summary['pe_util']:.2%}")
    print(f"Cycle-based PE utilization: {summary['cycle_pe_util']:.2%}")


def fine_tune(model, stats, train_loader, val_loader, args, device, label, use_amp):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.finetune_lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scaler = make_scaler(device, use_amp)
    for epoch in range(1, args.finetune_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            f"{label} ft {epoch}",
            prune_stats=stats,
            tm=args.tm,
            scaler=scaler,
            use_amp=use_amp,
        )
        val_loss, val_acc = evaluate(model, val_loader, device, use_amp=use_amp)
        print(
            f"{label} fine-tune epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.2%}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.2%}"
        )
    return evaluate(model, val_loader, device, use_amp=use_amp)


def train_dense_extra(model, train_loader, val_loader, args, device, use_amp):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.finetune_lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scaler = make_scaler(device, use_amp)
    for epoch in range(1, args.finetune_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            f"dense extra {epoch}",
            scaler=scaler,
            use_amp=use_amp,
        )
        val_loss, val_acc = evaluate(model, val_loader, device, use_amp=use_amp)
        print(
            f"Dense extra epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.2%}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.2%}"
        )
    return evaluate(model, val_loader, device, use_amp=use_amp)


def resolve_checkpoint_path(value, model_name, seed):
    if value is None:
        return None
    return Path(value.format(model=model_name, seed=seed))


def make_checkpoint_id(args, model_name, split_metadata):
    return (
        f"{args.dataset}-{model_name}-seed{args.seed}-"
        f"split{split_metadata['split_id']}-e{args.epochs}"
    )


def save_dense_checkpoint(
    path,
    model,
    args,
    model_name,
    num_classes,
    split_metadata,
    dense_val_acc,
    dense_test_acc,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_id = make_checkpoint_id(args, model_name, split_metadata)
    payload = {
        "format_version": 1,
        "checkpoint_id": checkpoint_id,
        "dataset": args.dataset,
        "model": model_name,
        "num_classes": num_classes,
        "image_size": args.image_size,
        "seed": args.seed,
        "epochs": args.epochs,
        "training_config": {
            "batch_size": args.batch_size,
            "lr": args.lr,
            "lr_scheduler": args.lr_scheduler,
            "min_lr": args.min_lr,
            "weight_decay": args.weight_decay,
            "deterministic": args.deterministic,
        },
        "split_metadata": dict(split_metadata),
        "dense_val_acc": float(dense_val_acc),
        "dense_test_acc": float(dense_test_acc),
        "model_state_dict": {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        },
    }
    torch.save(payload, path)
    print(f"Saved Dense checkpoint to: {path}")
    print(f"Checkpoint ID: {checkpoint_id}")
    return checkpoint_id


def load_dense_checkpoint(
    path,
    model,
    args,
    model_name,
    num_classes,
    split_metadata,
    device,
):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dense checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(
            f"Unsupported checkpoint format in {path}; expected a saved Dense checkpoint."
        )

    expected = {
        "dataset": args.dataset,
        "model": model_name,
        "num_classes": num_classes,
        "image_size": args.image_size,
        "seed": args.seed,
    }
    mismatches = [
        f"{key}: checkpoint={payload.get(key)!r}, current={value!r}"
        for key, value in expected.items()
        if payload.get(key) != value
    ]
    checkpoint_split = payload.get("split_metadata", {})
    for key in (
        "split_id",
        "split_seed",
        "val_size",
        "train_size",
        "validation_size",
    ):
        if checkpoint_split.get(key) != split_metadata.get(key):
            mismatches.append(
                f"{key}: checkpoint={checkpoint_split.get(key)!r}, "
                f"current={split_metadata.get(key)!r}"
            )
    if mismatches:
        raise ValueError("Dense checkpoint metadata mismatch: " + "; ".join(mismatches))

    model.load_state_dict(payload["model_state_dict"], strict=True)
    checkpoint_id = payload.get("checkpoint_id", path.stem)
    print(f"Loaded Dense checkpoint from: {path}")
    print(f"Checkpoint ID: {checkpoint_id}")
    return checkpoint_id, payload


def parse_criteria(value):
    choices = ["l1_sum", "l2_sum", "max_abs", "top2_mean_abs", "top4_mean_abs", "min_abs", "random"]
    if value == "all":
        return choices
    criteria = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in criteria if item not in choices]
    if unknown:
        raise ValueError(f"Unknown criteria {unknown}. Choices: {choices}")
    return criteria


def parse_models(value):
    if value == "all":
        return SUPPORTED_MODELS
    models = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in models if item not in SUPPORTED_MODELS]
    if unknown:
        raise ValueError(f"Unknown models {unknown}. Choices: {SUPPORTED_MODELS}")
    return models


def parse_args():
    parser = argparse.ArgumentParser(description="CNN pruning experiments.")
    parser.add_argument("--dataset", choices=["cifar10", "animals10", "imagenette", "imagefolder"], default="cifar10")
    parser.add_argument("--models", default="resnet34", help="Comma-separated model list or all.")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--animals-dir", default=None)
    parser.add_argument("--imagenette-dir", default=None)
    parser.add_argument("--imagefolder-dir", default=None)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--finetune-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "cosine"],
        default="none",
        help="Dense-training learning-rate schedule.",
    )
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--finetune-lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--tm", type=int, default=8)
    parser.add_argument(
        "--macs-per-pe",
        type=int,
        default=8,
        help="MAC units per PE for the paper-style cycle estimate.",
    )
    parser.add_argument("--prune-ratio", type=float, default=0.6)
    parser.add_argument("--cross-criteria", default="l1_sum,max_abs")
    parser.add_argument(
        "--filter-grouping",
        choices=["contiguous", "similarity_jaccard"],
        default="contiguous",
        help=(
            "How output filters are assigned to Tm clusters. similarity_jaccard "
            "groups filters whose global-pruning keep masks overlap."
        ),
    )
    parser.add_argument(
        "--cross-only",
        action="store_true",
        help="Skip dense-extra and global fine-tuning when only a new cross-filter variant is needed.",
    )
    parser.add_argument(
        "--protection-criteria",
        default=None,
        help=(
            "Comma-separated criteria that receive layer caps and early-layer protection. "
            "Defaults to every cross-filter criterion."
        ),
    )
    parser.add_argument(
        "--max-layer-prune-ratio",
        type=float,
        default=None,
        help="Optional cap for every cross-filter layer/cluster prune ratio.",
    )
    parser.add_argument(
        "--min-layer-keep-ratio",
        type=float,
        default=None,
        help="Optional minimum kept ratio for every cross-filter layer/cluster.",
    )
    parser.add_argument(
        "--protect-early-layers",
        type=int,
        default=0,
        help="Number of first prunable conv layers to protect with --protected-prune-ratio.",
    )
    parser.add_argument(
        "--protected-prune-ratio",
        type=float,
        default=0.2,
        help="Maximum prune ratio for protected early layers.",
    )
    parser.add_argument(
        "--cross-ratio-source",
        choices=["global_cluster", "global_layer", "uniform"],
        default="global_cluster",
        help=(
            "How cross-filter pruning chooses per-cluster prune ratios. "
            "global_cluster follows the paper most closely: use the sparsity of "
            "the corresponding cluster from global unstructured pruning."
        ),
    )
    parser.add_argument(
        "--sensitivity-csv",
        default=None,
        help="Optional layer_sensitivity_ranked.csv used to cap/skip sensitive layers.",
    )
    parser.add_argument(
        "--sensitivity-criteria",
        default=None,
        help=(
            "Comma-separated cross-filter criteria that receive sensitivity protection. "
            "Defaults to every criterion when --sensitivity-csv is used."
        ),
    )
    parser.add_argument(
        "--sensitivity-protect-top-k",
        type=int,
        default=0,
        help="Protect the top-K sensitive layers for each model/criterion.",
    )
    parser.add_argument(
        "--sensitivity-skip-top-k",
        type=int,
        default=0,
        help="Set prune ratio to 0 for the first N layers among the protected top-K.",
    )
    parser.add_argument(
        "--sensitivity-protected-prune-ratio",
        type=float,
        default=0.2,
        help="Maximum prune ratio for sensitivity-protected layers that are not skipped.",
    )
    parser.add_argument(
        "--sensitivity-min-mean-drop-pp",
        type=float,
        default=0.0,
        help="Only protect layers whose mean sensitivity drop is at least this many percentage points.",
    )
    parser.add_argument("--no-prune-1x1", action="store_true")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--train-dense-only",
        action="store_true",
        help="Train/load the Dense model, save its checkpoint, and stop before pruning.",
    )
    parser.add_argument(
        "--dense-checkpoint-in",
        default=None,
        help="Dense checkpoint to reuse. Supports {model} and {seed} placeholders.",
    )
    parser.add_argument(
        "--dense-checkpoint-out",
        default=None,
        help="Where to save a trained Dense checkpoint. Supports {model} and {seed} placeholders.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic CUDA algorithms when available.",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision.")
    parser.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--no-cudnn-benchmark", action="store_true")
    parser.add_argument("--save-dir", default="runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-save-csv", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def choose_device(requested):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_device_info(device):
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version from PyTorch: {torch.version.cuda}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU memory: {total_mem:.1f} GB")


def cross_ratio_config_label(args, sensitivity_enabled=True, protection_enabled=True):
    label = args.cross_ratio_source
    if args.filter_grouping != "contiguous":
        label += f"+group_{args.filter_grouping}"
    if protection_enabled and args.protect_early_layers > 0:
        label += f"+protect{args.protect_early_layers}@{args.protected_prune_ratio:g}"
    if protection_enabled and args.max_layer_prune_ratio is not None:
        label += f"+cap{args.max_layer_prune_ratio:g}"
    if protection_enabled and args.min_layer_keep_ratio is not None:
        label += f"+keep{args.min_layer_keep_ratio:g}"
    if sensitivity_enabled and args.sensitivity_csv and args.sensitivity_protect_top_k > 0:
        label += f"+sens{args.sensitivity_protect_top_k}@{args.sensitivity_protected_prune_ratio:g}"
        if args.sensitivity_skip_top_k > 0:
            label += f"+skip{args.sensitivity_skip_top_k}"
    return label


def result_rows(dataset, model_name, dense_acc, dense_extra_acc, results):
    def make_row(method, criterion, ratio_source, direct_acc, after_acc, summary):
        return {
            "dataset": dataset,
            "model": model_name,
            "method": method,
            "criterion": criterion,
            "ratio_source": ratio_source,
            "direct_acc": direct_acc,
            "after_extra_or_ft_acc": after_acc,
            "pruning_ratio": summary["pruning_ratio"],
            "mac_reduction": summary["mac_reduction"],
            "pe_utilization": summary["pe_util"],
            "cycle_pe_utilization": summary["cycle_pe_util"],
            "cycle_reduction": summary["cycle_reduction"],
            "estimated_speedup": summary["estimated_speedup"],
        }

    dense_summary = {
        "pruning_ratio": 0.0,
        "mac_reduction": 0.0,
        "pe_util": 1.0,
        "cycle_pe_util": 1.0,
        "cycle_reduction": 0.0,
        "estimated_speedup": 1.0,
    }
    rows = [make_row("Dense baseline", "", "", dense_acc, "", dense_summary)]
    if dense_extra_acc is not None:
        rows.append(
            make_row("Dense extra training", "", "", dense_acc, dense_extra_acc, dense_summary)
        )
    for result in results:
        summary = summarize(result.stats)
        rows.append(
            make_row(
                result.name,
                result.criterion,
                result.ratio_source,
                result.direct_acc,
                result.finetuned_acc,
                summary,
            )
        )
    return rows


def save_csv(args, rows, run_name=None):
    run_name = run_name or args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{run_name}_results.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print()
    print(f"Saved CSV results to: {path}")
    return path


def run_one_model(
    args,
    model_name,
    train_loader,
    val_loader,
    test_loader,
    num_classes,
    device,
    use_amp,
    include_1x1,
    criteria,
    split_metadata,
):
    print()
    print("=" * 80)
    print(f"=== Experiment: {model_name} on {args.dataset} ===")
    print("=" * 80)
    print_device_info(device)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {model_name}, classes={num_classes}, image_size={args.image_size}")
    print(f"Tm / PE cluster size: {args.tm}")
    print(f"Prune ratio: {args.prune_ratio:.2%}")
    print(f"Filter grouping: {args.filter_grouping}")
    print("Cross-filter criteria: " + ", ".join(criteria))
    print("Split ID: {}".format(split_metadata.get("split_id")))
    print(
        "Samples: train={}, validation={}, test={}".format(
            split_metadata.get("train_size"),
            split_metadata.get("validation_size"),
            split_metadata.get("test_size"),
        )
    )

    model = make_model(model_name, num_classes, args.image_size).to(device)
    checkpoint_in = resolve_checkpoint_path(args.dense_checkpoint_in, model_name, args.seed)
    checkpoint_out = resolve_checkpoint_path(args.dense_checkpoint_out, model_name, args.seed)
    checkpoint_id = None

    if checkpoint_in is not None:
        checkpoint_id, _ = load_dense_checkpoint(
            checkpoint_in,
            model,
            args,
            model_name,
            num_classes,
            split_metadata,
            device,
        )
    else:
        reset_training_rng(args.seed, train_loader)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
        scaler = make_scaler(device, use_amp)
        scheduler = None
        if args.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(args.epochs, 1),
                eta_min=args.min_lr,
            )
        print()
        print("=== Dense training ===")
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                epoch,
                scaler=scaler,
                use_amp=use_amp,
            )
            val_loss, val_acc = evaluate(model, val_loader, device, use_amp=use_amp)
            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.2%}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.2%}, "
                f"lr={optimizer.param_groups[0]['lr']:.6f}"
            )
            if scheduler is not None:
                scheduler.step()

    dense_val_loss, dense_val_acc = evaluate(model, val_loader, device, use_amp=use_amp)
    dense_loss, dense_acc = evaluate(model, test_loader, device, use_amp=use_amp)
    specs = collect_conv_specs(model, args.image_size)
    print()
    print("=== Dense baseline ===")
    print(f"Validation loss: {dense_val_loss:.4f}")
    print(f"Validation accuracy: {dense_val_acc:.2%}")
    print(f"Test loss: {dense_loss:.4f}")
    print(f"Test accuracy: {dense_acc:.2%}")
    print(f"Detected conv layers: {len(specs)}")

    if checkpoint_out is None and args.train_dense_only and checkpoint_in is None:
        checkpoint_out = (
            Path(args.save_dir)
            / "checkpoints"
            / f"{args.dataset}_{model_name}_seed{args.seed}_dense.pt"
        )
    if checkpoint_out is not None:
        checkpoint_id = save_dense_checkpoint(
            checkpoint_out,
            model,
            args,
            model_name,
            num_classes,
            split_metadata,
            dense_val_acc,
            dense_acc,
        )
    elif checkpoint_id is None:
        checkpoint_id = make_checkpoint_id(args, model_name, split_metadata)

    if args.train_dense_only:
        print("Dense-only mode complete; pruning stages were skipped.")
        rows = result_rows(args.dataset, model_name, dense_acc, "", [])[:1]
        rows[0]["seed"] = args.seed
        rows[0]["checkpoint_id"] = checkpoint_id
        rows[0]["validation_acc"] = dense_val_acc
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return rows

    dense_extra_model = None
    dense_extra_acc = None
    if not args.cross_only:
        print()
        print("=== Fair dense extra-training baseline ===")
        dense_extra_model = copy.deepcopy(model)
        reset_training_rng(args.seed + 100000, train_loader)
        _, dense_extra_val_acc = train_dense_extra(
            dense_extra_model,
            train_loader,
            val_loader,
            args,
            device,
            use_amp,
        )
        _, dense_extra_acc = evaluate(dense_extra_model, test_loader, device, use_amp=use_amp)
        print(f"Dense extra final validation accuracy: {dense_extra_val_acc:.2%}")
        print(f"Dense extra final test accuracy: {dense_extra_acc:.2%}")

    results = []
    sparsity_reference = global_unstructured_reference(
        specs,
        tm=args.tm,
        target_sparsity=args.prune_ratio,
        include_1x1=include_1x1,
        filter_grouping=args.filter_grouping,
    )

    if args.filter_grouping != "contiguous":
        grouped_similarity = np.mean(
            [item["group_similarity"] for item in sparsity_reference.values()]
        )
        contiguous_similarity = np.mean(
            [item["contiguous_group_similarity"] for item in sparsity_reference.values()]
        )
        print(
            "Mean within-cluster mask Jaccard: "
            f"contiguous={contiguous_similarity:.4f}, "
            f"{args.filter_grouping}={grouped_similarity:.4f}"
        )

    global_model = None
    if not args.cross_only:
        global_model = copy.deepcopy(model)
        global_specs = collect_conv_specs(global_model, args.image_size)
        global_stats = global_unstructured_prune_(
            global_model,
            global_specs,
            tm=args.tm,
            target_sparsity=args.prune_ratio,
            include_1x1=include_1x1,
            sparsity_reference=sparsity_reference,
            macs_per_pe=args.macs_per_pe,
        )
        print_report("Global unstructured pruning report", global_stats)
        _, global_direct_acc = evaluate(global_model, test_loader, device, use_amp=use_amp)
        reset_training_rng(args.seed + 100000, train_loader)
        fine_tune(
            global_model,
            global_stats,
            train_loader,
            val_loader,
            args,
            device,
            f"{model_name} global",
            use_amp,
        )
        _, global_ft_acc = evaluate(global_model, test_loader, device, use_amp=use_amp)
        results.append(
            Result(
                "Global unstructured",
                global_direct_acc,
                global_ft_acc,
                global_stats,
                ratio_source="global",
            )
        )

    sensitivity_criteria = (
        set(parse_criteria(args.sensitivity_criteria))
        if args.sensitivity_criteria
        else set(criteria)
    )
    protection_criteria = (
        set(parse_criteria(args.protection_criteria))
        if args.protection_criteria
        else set(criteria)
    )
    for criterion in criteria:
        sensitivity_enabled = criterion in sensitivity_criteria
        protection_enabled = criterion in protection_criteria
        if sensitivity_enabled:
            layer_prune_caps, _ = load_sensitivity_layer_caps(
                args.sensitivity_csv,
                model_name,
                criterion,
                args.sensitivity_protect_top_k,
                args.sensitivity_protected_prune_ratio,
                skip_top_k=args.sensitivity_skip_top_k,
                min_mean_drop_pp=args.sensitivity_min_mean_drop_pp,
            )
        else:
            layer_prune_caps = {}
        ratio_config_label = cross_ratio_config_label(
            args, sensitivity_enabled, protection_enabled
        )
        cross_model = copy.deepcopy(model)
        cross_specs = collect_conv_specs(cross_model, args.image_size)
        cross_stats = cross_filter_prune_(
            cross_model,
            cross_specs,
            tm=args.tm,
            prune_ratio=args.prune_ratio,
            criterion=criterion,
            include_1x1=include_1x1,
            ratio_source=args.cross_ratio_source,
            sparsity_reference=sparsity_reference,
            macs_per_pe=args.macs_per_pe,
            max_layer_prune_ratio=(
                args.max_layer_prune_ratio if protection_enabled else None
            ),
            min_layer_keep_ratio=(
                args.min_layer_keep_ratio if protection_enabled else None
            ),
            protect_early_layers=(args.protect_early_layers if protection_enabled else 0),
            protected_prune_ratio=args.protected_prune_ratio,
            layer_prune_caps=layer_prune_caps,
            filter_grouping=args.filter_grouping,
        )
        cross_name = f"Cross-filter {criterion} ({ratio_config_label})"
        print_report(f"{cross_name} pruning report", cross_stats)
        _, direct_acc = evaluate(cross_model, test_loader, device, use_amp=use_amp)
        reset_training_rng(args.seed + 100000, train_loader)
        fine_tune(
            cross_model,
            cross_stats,
            train_loader,
            val_loader,
            args,
            device,
            f"{model_name} cross {criterion}",
            use_amp,
        )
        _, ft_acc = evaluate(cross_model, test_loader, device, use_amp=use_amp)
        results.append(
            Result(
                cross_name,
                direct_acc,
                ft_acc,
                cross_stats,
                criterion=criterion,
                ratio_source=ratio_config_label,
            )
        )

    print()
    print("=== Final comparison ===")
    print(f"Dense baseline test accuracy: {dense_acc:.2%}")
    if dense_extra_acc is not None:
        print(f"Dense extra test accuracy: {dense_extra_acc:.2%}")
    for result in results:
        summary = summarize(result.stats)
        print(
            "{}: direct={:.2%}, after_ft={:.2%}, prune={:.2%}".format(
                result.name,
                result.direct_acc,
                result.finetuned_acc,
                summary.get("pruning_ratio", 0.0),
            )
        )

    rows = result_rows(args.dataset, model_name, dense_acc, dense_extra_acc, results)
    for row in rows:
        row["seed"] = args.seed
        row["checkpoint_id"] = checkpoint_id
        row["tm"] = args.tm
        row["target_prune_ratio"] = args.prune_ratio
        row["dense_test_acc"] = dense_acc
        row["dense_extra_test_acc"] = dense_extra_acc if dense_extra_acc is not None else ""
        after_acc = row.get("after_extra_or_ft_acc")
        row["accuracy_delta_vs_dense_extra"] = (
            after_acc - dense_extra_acc
            if isinstance(after_acc, (int, float)) and dense_extra_acc is not None
            else ""
        )

    del model
    if dense_extra_model is not None:
        del dense_extra_model
    if global_model is not None:
        del global_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def main():
    args = parse_args()
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    reset_training_rng(args.seed)
    device = choose_device(args.device)
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    elif device.type == "cuda" and not args.no_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    use_amp = args.amp_dtype if args.amp and device.type == "cuda" else False
    include_1x1 = not args.no_prune_1x1
    criteria = parse_criteria(args.cross_criteria)
    models = parse_models(args.models)
    pin_memory = args.pin_memory

    if args.dataset == "cifar10":
        (train_loader, val_loader, test_loader), num_classes, split_metadata = build_cifar10(
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
    elif args.dataset == "animals10":
        if args.animals_dir is None:
            raise ValueError("--animals-dir is required for --dataset animals10")
        (train_loader, val_loader, test_loader), num_classes, split_metadata = build_animals10(
            args.animals_dir,
            args.batch_size,
            args.image_size,
            args.train_limit,
            args.test_limit,
            args.num_workers,
            pin_memory,
        )
    elif args.dataset == "imagenette":
        if args.imagenette_dir is None:
            raise ValueError("--imagenette-dir is required for --dataset imagenette")
        (train_loader, val_loader, test_loader), num_classes, split_metadata = build_imagefolder_dataset(
            args.imagenette_dir,
            args.batch_size,
            args.image_size,
            args.train_limit,
            args.test_limit,
            args.num_workers,
            pin_memory,
        )
    else:
        if args.imagefolder_dir is None:
            raise ValueError("--imagefolder-dir is required for --dataset imagefolder")
        (train_loader, val_loader, test_loader), num_classes, split_metadata = build_imagefolder_dataset(
            args.imagefolder_dir,
            args.batch_size,
            args.image_size,
            args.train_limit,
            args.test_limit,
            args.num_workers,
            pin_memory,
        )

    all_rows = []
    base_run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    for model_name in models:
        rows = run_one_model(
            args,
            model_name,
            train_loader,
            val_loader,
            test_loader,
            num_classes,
            device,
            use_amp,
            include_1x1,
            criteria,
            split_metadata,
        )
        all_rows.extend(rows)
        if not args.no_save_csv:
            if len(models) == 1:
                save_csv(args, rows, base_run_name)
            else:
                save_csv(args, rows, f"{base_run_name}_{model_name}")

    if len(models) > 1 and not args.no_save_csv:
        save_csv(args, all_rows, f"{base_run_name}_all_models")


if __name__ == "__main__":
    main()
