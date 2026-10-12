import argparse
import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class Im2ColConv2d(nn.Module):
    """
    2D convolution implemented with im2col/unfold + GEMM.

    Input:
        x: [N, C, H, W]
    Weight:
        weight: [M, C, R, S]
    Internal GEMM:
        weight_col: [M, C * R * S]
        x_col:      [N, C * R * S, OH * OW]
        y_col:      [N, M, OH * OW]
    Output:
        y: [N, M, OH, OW]
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        kh, kw = kernel_size
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kh, kw))
        self.bias = nn.Parameter(torch.empty(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        bound = fan_in**-0.5
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        n, _, h, w = x.shape
        kh, kw = self.kernel_size
        out_h = (h + 2 * self.padding - kh) // self.stride + 1
        out_w = (w + 2 * self.padding - kw) // self.stride + 1

        x_col = F.unfold(
            x,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
        )
        weight_col = self.weight.reshape(self.out_channels, -1)

        # Batched GEMM: [M, K] x [N, K, L] -> [N, M, L]
        y_col = torch.matmul(weight_col, x_col)
        y_col = y_col + self.bias.reshape(1, -1, 1)
        return y_col.reshape(n, self.out_channels, out_h, out_w)


class SmallIm2ColCNN(nn.Module):
    """
    A compact CNN for MNIST.

    Both convolution layers use im2col + GEMM internally. MNIST is grayscale,
    so conv1 has C=1 instead of the earlier random-input demo's C=3.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = Im2ColConv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = Im2ColConv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = Im2ColConv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)


@dataclass
class PruneLayerStats:
    name: str
    method: str
    prune_ratio: float
    original_weights: int
    nonzero_weights: int
    original_macs_per_image: int
    pruned_macs_per_image: int
    masks: list[torch.Tensor]
    workload_utilization: float
    max_work_per_cluster: float
    avg_work_per_pe: float
    std_work_per_pe: float


@dataclass
class ExperimentResult:
    name: str
    direct_loss: float
    direct_acc: float
    finetuned_loss: float | None
    finetuned_acc: float | None
    stats: list[PruneLayerStats]


def make_subset(dataset, limit):
    if limit is None or limit >= len(dataset):
        return dataset
    return Subset(dataset, range(limit))


def build_loaders(data_dir, batch_size, train_limit=None, test_limit=None):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_dataset = make_subset(train_dataset, train_limit)
    test_dataset = make_subset(test_dataset, test_limit)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, test_loader


@torch.no_grad()
def apply_prune_masks_(model, stats, tm, include_grad=False):
    """
    Re-apply pruning masks after optimizer updates.

    During fine-tuning, gradients can make pruned weights nonzero again. This
    function keeps those positions at zero so the structured pruning pattern is
    preserved.
    """
    for item in stats:
        layer = getattr(model, item.name)
        for cluster_id, mask in enumerate(item.masks):
            cluster_start = cluster_id * tm
            cluster_end = min(cluster_start + tm, layer.weight.shape[0])
            mask = mask.to(device=layer.weight.device, dtype=layer.weight.dtype)

            # Cross-filter masks are [C, R, S] and broadcast to all filters.
            # Unstructured masks are [filters_in_cluster, C, R, S].
            layer.weight[cluster_start:cluster_end].mul_(mask)

            if include_grad and layer.weight.grad is not None:
                layer.weight.grad[cluster_start:cluster_end].mul_(mask)


def train_one_epoch(model, loader, optimizer, device, epoch, prune_stats=None, tm=4):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (x, target) in enumerate(loader, start=1):
        x = x.to(device)
        target = target.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        if prune_stats is not None:
            apply_prune_masks_(model, prune_stats, tm=tm, include_grad=True)
        optimizer.step()
        if prune_stats is not None:
            apply_prune_masks_(model, prune_stats, tm=tm)

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == target).sum().item()
        total_samples += x.size(0)

        if batch_idx % 100 == 0:
            avg_loss = total_loss / total_samples
            acc = total_correct / total_samples
            print(
                f"Epoch {epoch} batch {batch_idx:4d}: "
                f"loss={avg_loss:.4f}, train_acc={acc:.2%}"
            )

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, target in loader:
        x = x.to(device)
        target = target.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, target)

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == target).sum().item()
        total_samples += x.size(0)

    return total_loss / total_samples, total_correct / total_samples


def compute_cross_filter_importance(cluster, criterion):
    abs_cluster = cluster.abs()

    if criterion == "l1_sum":
        return abs_cluster.sum(dim=0)
    if criterion == "l2_sum":
        return torch.sqrt((cluster * cluster).sum(dim=0))
    if criterion == "max_abs":
        return abs_cluster.max(dim=0).values
    if criterion == "min_abs":
        return abs_cluster.min(dim=0).values
    if criterion == "random":
        return torch.rand(
            cluster.shape[1:],
            dtype=cluster.dtype,
            device=cluster.device,
        )

    raise ValueError(f"Unknown cross-filter importance criterion: {criterion}")


@torch.no_grad()
def cross_filter_prune_weight_(weight, tm, prune_ratio, criterion):
    """
    In-place cross-filter structured pruning for one conv weight tensor.

    For each cluster of Tm output filters, compute one shared [C, R, S]
    importance map, then prune the least important positions.

    The same mask is applied to every filter in that cluster.
    """
    out_channels, in_channels, kh, kw = weight.shape
    positions_per_filter = in_channels * kh * kw
    num_prune = round(prune_ratio * positions_per_filter)
    num_prune = min(max(num_prune, 0), positions_per_filter)
    masks = []

    for cluster_start in range(0, out_channels, tm):
        cluster_end = min(cluster_start + tm, out_channels)
        cluster = weight[cluster_start:cluster_end]

        importance = compute_cross_filter_importance(cluster, criterion)
        flat_importance = importance.reshape(-1)
        prune_indices = torch.argsort(flat_importance)[:num_prune]

        flat_mask = torch.ones(
            positions_per_filter,
            dtype=weight.dtype,
            device=weight.device,
        )
        flat_mask[prune_indices] = 0
        mask = flat_mask.reshape(in_channels, kh, kw)
        masks.append(mask.detach().cpu())

        weight[cluster_start:cluster_end].mul_(mask)

    return masks


@torch.no_grad()
def global_unstructured_prune_model_(model, target_sparsity, tm):
    """
    Global unstructured pruning across all convolution weights.

    This is the baseline from the paper's discussion: prune the smallest
    individual weights without forcing filters in a PE cluster to share masks.
    The result usually preserves accuracy well, but can create load imbalance.
    """
    conv_layers = get_conv_layer_specs(model)
    all_weights = torch.cat(
        [layer.weight.detach().abs().reshape(-1) for _, layer, _ in conv_layers]
    )
    num_prune = round(target_sparsity * all_weights.numel())
    num_prune = min(max(num_prune, 0), all_weights.numel())

    global_flat_mask = torch.ones_like(all_weights)
    if num_prune > 0:
        prune_indices = torch.argsort(all_weights)[:num_prune]
        global_flat_mask[prune_indices] = 0

    stats = []
    offset = 0
    for name, layer, input_hw in conv_layers:
        weight = layer.weight
        original_weights = weight.numel()
        out_channels, in_channels, kh, kw = weight.shape
        out_hw = output_hw(
            input_hw,
            kernel_size=kh,
            stride=layer.stride,
            padding=layer.padding,
        )
        original_macs = out_channels * in_channels * kh * kw * out_hw[0] * out_hw[1]

        next_offset = offset + original_weights
        full_mask = global_flat_mask[offset:next_offset].reshape_as(weight)
        offset = next_offset

        weight.mul_(full_mask)
        masks = split_full_mask_by_cluster(full_mask, tm)
        nonzero_weights = int(torch.count_nonzero(weight).item())
        pruned_macs = nonzero_weights * out_hw[0] * out_hw[1]
        stats.append(
            build_layer_stats(
                name=name,
                method="global_unstructured",
                prune_ratio=1 - nonzero_weights / original_weights,
                original_weights=original_weights,
                nonzero_weights=nonzero_weights,
                original_macs=original_macs,
                pruned_macs=pruned_macs,
                masks=masks,
                tm=tm,
            )
        )

    return stats


def split_full_mask_by_cluster(full_mask, tm):
    masks = []
    out_channels = full_mask.shape[0]
    for cluster_start in range(0, out_channels, tm):
        cluster_end = min(cluster_start + tm, out_channels)
        masks.append(full_mask[cluster_start:cluster_end].detach().cpu())
    return masks


def output_hw(input_hw, kernel_size=3, stride=1, padding=1):
    h, w = input_hw
    out_h = (h + 2 * padding - kernel_size) // stride + 1
    out_w = (w + 2 * padding - kernel_size) // stride + 1
    return out_h, out_w


def macs_per_image_from_masks(masks, out_channels, out_hw, tm):
    out_h, out_w = out_hw
    macs = 0
    for cluster_id, mask in enumerate(masks):
        cluster_start = cluster_id * tm
        cluster_end = min(cluster_start + tm, out_channels)
        filters_in_cluster = cluster_end - cluster_start
        if mask.dim() == 3:
            kept_positions = int(mask.sum().item())
            macs += kept_positions * filters_in_cluster * out_h * out_w
        else:
            macs += int(mask.sum().item()) * out_h * out_w
    return macs


def get_conv_layer_specs(model):
    return [
        ("conv1", model.conv1, (28, 28)),
        ("conv2", model.conv2, (14, 14)),
        ("conv3", model.conv3, (7, 7)),
    ]


def workload_balance_from_masks(masks, tm):
    """
    Simulate PE load balance inside each filter cluster.

    Work per PE is the number of nonzero weight positions assigned to that PE.
    A synchronized PE cluster takes max(work_per_pe) steps, so utilization is:
        useful_work / (cluster_steps * number_of_PEs)
    """
    all_work = []
    total_useful = 0
    total_capacity = 0
    max_work_per_cluster = []

    for mask in masks:
        if mask.dim() == 3:
            work = torch.full((tm,), float(mask.sum().item()))
            work = work[:1] if tm == 1 else work
        else:
            work = mask.reshape(mask.shape[0], -1).sum(dim=1).float()

        if work.numel() == 0:
            continue

        all_work.append(work)
        cluster_max = float(work.max().item())
        max_work_per_cluster.append(cluster_max)
        total_useful += float(work.sum().item())
        total_capacity += cluster_max * work.numel()

    if not all_work:
        return 1.0, 0.0, 0.0, 0.0

    work_values = torch.cat(all_work)
    utilization = total_useful / total_capacity if total_capacity else 1.0
    return (
        utilization,
        sum(max_work_per_cluster) / len(max_work_per_cluster),
        float(work_values.mean().item()),
        float(work_values.std(unbiased=False).item()),
    )


def build_layer_stats(
    name,
    method,
    prune_ratio,
    original_weights,
    nonzero_weights,
    original_macs,
    pruned_macs,
    masks,
    tm,
):
    utilization, avg_cluster_max, avg_work, std_work = workload_balance_from_masks(
        masks,
        tm=tm,
    )
    return PruneLayerStats(
        name=name,
        method=method,
        prune_ratio=prune_ratio,
        original_weights=original_weights,
        nonzero_weights=nonzero_weights,
        original_macs_per_image=original_macs,
        pruned_macs_per_image=pruned_macs,
        masks=masks,
        workload_utilization=utilization,
        max_work_per_cluster=avg_cluster_max,
        avg_work_per_pe=avg_work,
        std_work_per_pe=std_work,
    )


def prune_model_(model, tm, prune_ratios, criterion):
    """
    Apply cross-filter structured pruning to every convolution layer.
    """
    stats = []

    for name, layer, input_hw in get_conv_layer_specs(model):
        prune_ratio = prune_ratios[name]
        weight = layer.weight
        original_weights = weight.numel()
        out_channels, in_channels, kh, kw = weight.shape
        out_hw = output_hw(
            input_hw,
            kernel_size=kh,
            stride=layer.stride,
            padding=layer.padding,
        )
        original_macs = out_channels * in_channels * kh * kw * out_hw[0] * out_hw[1]

        masks = cross_filter_prune_weight_(
            weight,
            tm=tm,
            prune_ratio=prune_ratio,
            criterion=criterion,
        )
        nonzero_weights = int(torch.count_nonzero(weight).item())
        pruned_macs = macs_per_image_from_masks(
            masks,
            out_channels=out_channels,
            out_hw=out_hw,
            tm=tm,
        )

        stats.append(
            build_layer_stats(
                name=name,
                method=f"cross_filter_{criterion}",
                prune_ratio=prune_ratio,
                original_weights=original_weights,
                nonzero_weights=nonzero_weights,
                original_macs=original_macs,
                pruned_macs=pruned_macs,
                masks=masks,
                tm=tm,
            )
        )

    return stats


def summarize_stats(stats):
    original_weights = sum(item.original_weights for item in stats)
    nonzero_weights = sum(item.nonzero_weights for item in stats)
    original_macs = sum(item.original_macs_per_image for item in stats)
    pruned_macs = sum(item.pruned_macs_per_image for item in stats)
    useful = sum(item.pruned_macs_per_image for item in stats)
    capacity = 0.0
    for item in stats:
        if item.workload_utilization > 0:
            capacity += item.pruned_macs_per_image / item.workload_utilization
    workload_utilization = useful / capacity if capacity else 1.0
    return {
        "original_weights": original_weights,
        "nonzero_weights": nonzero_weights,
        "pruning_ratio": 1 - nonzero_weights / original_weights,
        "original_macs": original_macs,
        "pruned_macs": pruned_macs,
        "mac_reduction": 1 - pruned_macs / original_macs,
        "workload_utilization": workload_utilization,
    }


def print_pruning_report(stats, title="Pruning report", print_masks=False):
    summary = summarize_stats(stats)

    print()
    print(f"=== {title} ===")
    print(f"Original conv weight count: {summary['original_weights']}")
    print(f"Pruned nonzero conv weight count: {summary['nonzero_weights']}")
    print(f"Actual conv pruning ratio: {summary['pruning_ratio']:.2%}")
    print(f"Original conv MACs per image: {summary['original_macs']}")
    print(f"Pruned effective conv MACs per image: {summary['pruned_macs']}")
    print(f"Conv MAC reduction: {summary['mac_reduction']:.2%}")
    print(f"Estimated PE workload utilization: {summary['workload_utilization']:.2%}")

    for item in stats:
        print()
        print(f"Layer {item.name}")
        print(f"  method: {item.method}")
        print(f"  target prune ratio: {item.prune_ratio:.2%}")
        print(f"  original weights: {item.original_weights}")
        print(f"  nonzero weights: {item.nonzero_weights}")
        print(
            "  pruning ratio: "
            f"{1 - item.nonzero_weights / item.original_weights:.2%}"
        )
        print(f"  original MACs/image: {item.original_macs_per_image}")
        print(f"  pruned MACs/image: {item.pruned_macs_per_image}")
        print(f"  PE workload utilization: {item.workload_utilization:.2%}")
        print(f"  avg work/PE: {item.avg_work_per_pe:.2f}")
        print(f"  std work/PE: {item.std_work_per_pe:.2f}")
        print(f"  avg cluster max work: {item.max_work_per_cluster:.2f}")

        if print_masks:
            print("  masks:")
            for cluster_id, mask in enumerate(item.masks):
                shape_note = "[C, R, S]" if mask.dim() == 3 else "[filters, C, R, S]"
                print(f"    cluster {cluster_id}, shape = {shape_note}")
                print(mask.to(torch.int32).numpy())


def weighted_target_sparsity(model, prune_ratios):
    total_weights = 0
    target_pruned = 0
    for name, layer, _ in get_conv_layer_specs(model):
        weights = layer.weight.numel()
        total_weights += weights
        target_pruned += round(prune_ratios[name] * weights)
    return target_pruned / total_weights


def fine_tune_pruned_model(model, stats, train_loader, test_loader, args, device, label):
    if args.finetune_epochs <= 0:
        return None, None

    optimizer = torch.optim.Adam(model.parameters(), lr=args.finetune_lr)

    print()
    print(f"=== Fine-tuning {label} model ===")
    print("Pruned weights are kept at zero by re-applying masks after each update.")
    for epoch in range(1, args.finetune_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            f"{label} fine-tune {epoch}",
            prune_stats=stats,
            tm=args.tm,
        )
        test_loss, test_acc = evaluate(model, test_loader, device)
        print(
            f"{label} fine-tune epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.2%}, "
            f"test_loss={test_loss:.4f}, test_acc={test_acc:.2%}"
        )

    return evaluate(model, test_loader, device)


def run_pruned_experiment(
    name,
    model,
    prune_fn,
    train_loader,
    test_loader,
    args,
    device,
    print_masks=False,
):
    stats = prune_fn(model)
    print_pruning_report(stats, title=f"{name} pruning report", print_masks=print_masks)

    direct_loss, direct_acc = evaluate(model, test_loader, device)
    print()
    print(f"=== {name}: after pruning, before fine-tune ===")
    print(f"Test loss: {direct_loss:.4f}")
    print(f"Test accuracy: {direct_acc:.2%}")

    finetuned_loss, finetuned_acc = fine_tune_pruned_model(
        model,
        stats,
        train_loader,
        test_loader,
        args,
        device,
        label=name,
    )

    if finetuned_acc is not None:
        print()
        print(f"=== {name}: after fine-tuning ===")
        print(f"Test loss: {finetuned_loss:.4f}")
        print(f"Test accuracy: {finetuned_acc:.2%}")

    return ExperimentResult(
        name=name,
        direct_loss=direct_loss,
        direct_acc=direct_acc,
        finetuned_loss=finetuned_loss,
        finetuned_acc=finetuned_acc,
        stats=stats,
    )


def print_final_comparison(dense_loss, dense_acc, results):
    print()
    print("=== Final comparison ===")
    header = (
        f"{'Method':<28}"
        f"{'Direct acc':>12}"
        f"{'Fine-tuned acc':>16}"
        f"{'Prune':>10}"
        f"{'MAC red.':>10}"
        f"{'PE util.':>10}"
    )
    print(header)
    print("-" * len(header))
    print(
        f"{'Dense baseline':<28}"
        f"{dense_acc:>12.2%}"
        f"{'-':>16}"
        f"{'0.00%':>10}"
        f"{'0.00%':>10}"
        f"{'100.00%':>10}"
    )

    for result in results:
        summary = summarize_stats(result.stats)
        finetuned = (
            f"{result.finetuned_acc:.2%}" if result.finetuned_acc is not None else "-"
        )
        print(
            f"{result.name:<28}"
            f"{result.direct_acc:>12.2%}"
            f"{finetuned:>16}"
            f"{summary['pruning_ratio']:>10.2%}"
            f"{summary['mac_reduction']:>10.2%}"
            f"{summary['workload_utilization']:>10.2%}"
        )

    print()
    print(f"Dense baseline loss: {dense_loss:.4f}, accuracy: {dense_acc:.2%}")


def parse_cross_criteria(value):
    available = ["l1_sum", "l2_sum", "max_abs", "min_abs", "random"]
    if value == "all":
        return available

    criteria = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in criteria if item not in available]
    if unknown:
        raise ValueError(
            f"Unknown criteria {unknown}. Available: {available}, or use 'all'."
        )
    return criteria


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MNIST CNN with im2col+GEMM conv, then apply cross-filter structured pruning.",
    )
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=5e-4)
    parser.add_argument("--tm", type=int, default=8)
    parser.add_argument(
        "--prune-ratio",
        type=float,
        default=None,
        help="Optional global prune ratio. If set, it overrides conv1/conv2/conv3 ratios.",
    )
    parser.add_argument("--conv1-prune-ratio", type=float, default=0.25)
    parser.add_argument("--conv2-prune-ratio", type=float, default=0.5)
    parser.add_argument("--conv3-prune-ratio", type=float, default=0.5)
    parser.add_argument(
        "--cross-criteria",
        default="l1_sum",
        help=(
            "Comma-separated cross-filter importance criteria to test. "
            "Choices: l1_sum,l2_sum,max_abs,min_abs,random, or all."
        ),
    )
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--print-masks", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prune_ratios = {
        "conv1": args.conv1_prune_ratio,
        "conv2": args.conv2_prune_ratio,
        "conv3": args.conv3_prune_ratio,
    }
    if args.prune_ratio is not None:
        prune_ratios = {name: args.prune_ratio for name in prune_ratios}
    cross_criteria = parse_cross_criteria(args.cross_criteria)

    print(f"Device: {device}")
    print("Dataset: MNIST")
    print("Model: Small CNN with im2col/unfold + GEMM convolution")
    print(f"Tm: {args.tm}")
    print(
        "Prune ratios: "
        f"conv1={prune_ratios['conv1']:.2%}, "
        f"conv2={prune_ratios['conv2']:.2%}, "
        f"conv3={prune_ratios['conv3']:.2%}, "
        "fc=not pruned"
    )
    print(f"Fine-tune epochs: {args.finetune_epochs}")
    print(f"Cross-filter criteria: {', '.join(cross_criteria)}")

    train_loader, test_loader = build_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
    )

    model = SmallIm2ColCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print()
    print("=== Training ===")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
        )
        test_loss, test_acc = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.2%}, "
            f"test_loss={test_loss:.4f}, test_acc={test_acc:.2%}"
        )

    dense_test_loss, dense_test_acc = evaluate(model, test_loader, device)

    print()
    print("=== Dense baseline ===")
    print(f"Test loss: {dense_test_loss:.4f}")
    print(f"Test accuracy: {dense_test_acc:.2%}")

    global_target_sparsity = weighted_target_sparsity(model, prune_ratios)

    print()
    print("=== Experiment setup ===")
    print(
        "Global unstructured pruning target sparsity: "
        f"{global_target_sparsity:.2%}"
    )
    print(
        "Cross-filter structured pruning ratios: "
        f"conv1={prune_ratios['conv1']:.2%}, "
        f"conv2={prune_ratios['conv2']:.2%}, "
        f"conv3={prune_ratios['conv3']:.2%}"
    )

    results = []
    results.append(
        run_pruned_experiment(
            name="Global unstructured",
            model=copy.deepcopy(model),
            prune_fn=lambda m: global_unstructured_prune_model_(
                m,
                target_sparsity=global_target_sparsity,
                tm=args.tm,
            ),
            train_loader=train_loader,
            test_loader=test_loader,
            args=args,
            device=device,
            print_masks=args.print_masks,
        )
    )

    for criterion in cross_criteria:
        results.append(
            run_pruned_experiment(
                name=f"Cross-filter {criterion}",
                model=copy.deepcopy(model),
                prune_fn=lambda m, selected=criterion: prune_model_(
                    m,
                    tm=args.tm,
                    prune_ratios=prune_ratios,
                    criterion=selected,
                ),
                train_loader=train_loader,
                test_loader=test_loader,
                args=args,
                device=device,
                print_masks=args.print_masks,
            )
        )

    print_final_comparison(dense_test_loss, dense_test_acc, results)


if __name__ == "__main__":
    main()
