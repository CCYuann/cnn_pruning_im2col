#!/usr/bin/env python3
"""Run the frozen CIFAR-10 pruning recipes on other datasets.

Nothing is tuned here: hyperparameters, arms and ratios come from the CIFAR-10 study. Validation is the
last epoch, and test is read once when a run ends, so a finished run is never re-evaluated.
"""
import argparse, json, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
import retrain_recovery_worker as rw

core, common = rw.core, rw.common
PROJECT = rw.PROJECT
IMAGENET = ((.485, .456, .406), (.229, .224, .225))
DATA = {
    "cifar100": dict(classes=100, size=32, val=5000, norm=((.5071, .4867, .4408), (.2675, .2565, .2761))),
    "imagenette": dict(classes=10, size=160, val=1000, norm=IMAGENET, folder="imagenette2-160"),
    "imagewoof": dict(classes=10, size=160, val=1000, norm=IMAGENET, folder="imagewoof2-160"),
}


def build(name, seed, batch_size, workers, train_limit):
    spec = DATA[name]
    normalize = transforms.Normalize(*spec["norm"])
    if name == "cifar100":
        train_tf = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        eval_tf = transforms.Compose([transforms.ToTensor(), normalize])
        train_full = datasets.CIFAR100(PROJECT/"data", train=True, transform=train_tf)
        val_full = datasets.CIFAR100(PROJECT/"data", train=True, transform=eval_tf)
        test = datasets.CIFAR100(PROJECT/"data", train=False, transform=eval_tf)
    else:
        size = spec["size"]
        train_tf = transforms.Compose([transforms.RandomResizedCrop(size, scale=(.35, 1)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        eval_tf = transforms.Compose([transforms.Resize(round(size*1.15)), transforms.CenterCrop(size), transforms.ToTensor(), normalize])
        root = PROJECT/"data"/spec["folder"]
        train_full = datasets.ImageFolder(root/"train", train_tf)
        val_full = datasets.ImageFolder(root/"train", eval_tf)
        test = datasets.ImageFolder(root/"val", eval_tf)
    order = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(0)).tolist()
    train_ids = order[spec["val"]:]
    if train_limit:
        train_ids = train_ids[:train_limit]
    return core.make_loaders(Subset(train_full, train_ids), Subset(val_full, order[:spec["val"]]), test, batch_size, workers, True, seed)


def geometry(specs):
    out = {}
    for spec in specs:
        conv = spec.module
        h, w = core.output_hw(spec.input_hw, conv)
        out[spec.name] = dict(Hout=h, Wout=w, stride=conv.stride[0], clusters=conv.out_channels//8,
                              K=conv.in_channels*conv.kernel_size[0]*conv.kernel_size[1], Cin=conv.in_channels, kernel=conv.kernel_size[0])
    return out


def random_masks(masks, seed):
    """Sanity control: keep each cluster's position count, choose the positions at random."""
    generator, out = torch.Generator().manual_seed(seed), {}
    for name, mask in masks.items():
        order = torch.rand(mask.shape, generator=generator).argsort(1).argsort(1)
        out[name] = order < mask.sum(1, keepdim=True)
    return out


def checkpoint(args):
    return args.root/"checkpoints"/f"{args.dataset}_{args.model}_seed{args.seed}.pt"


def run(args, device):
    meta = DATA[args.dataset]
    name = f"{args.dataset}_{args.model}_seed{args.seed}_{args.arm}"+(f"_r{round(args.ratio*100)}" if args.arm != "dense" else "")+("_floor" if args.q_floor else "")+("_stem" if args.stem_keep else "")
    name += (f"_mac{args.mac_beta:g}" if args.allocator == "mac" else "")+(f"_n{args.q_floor_n}" if args.q_floor_n > 1 else "")
    name += ("_kd" if args.kd else "")+(f"_a{args.kd_alpha:g}" if args.kd_alpha != .9 else "")+(f"_wd{args.weight_decay:g}" if args.weight_decay != 5e-4 else "")+(f"_e{args.epochs}" if args.epochs else "")
    directory = args.root/name
    if (directory/"result.json").exists():
        return
    directory.mkdir(parents=True, exist_ok=True)
    model = core.make_model(args.model, meta["classes"], meta["size"]).to(device)
    epochs = 100 if args.arm == "dense" else 60
    epochs = args.epochs or epochs
    if args.arm != "dense":
        model.load_state_dict(torch.load(checkpoint(args), map_location=device, weights_only=False))
    teacher = None
    if args.kd:
        teacher = core.make_model(args.model, meta["classes"], meta["size"]).to(device).eval()
        teacher.load_state_dict(torch.load(checkpoint(args), map_location=device, weights_only=False))
        teacher.requires_grad_(False)
    specs = core.collect_conv_specs(model, meta["size"])
    if any(s.module.groups != 1 for s in specs):
        raise ValueError("Standard convolutions only")
    (args.root/f"geometry_{args.dataset}_{args.model}.json").write_text(json.dumps(geometry(specs), indent=1))
    full = None
    masks_at = lambda ratio: rw.build_masks(specs, ratio, args.allocator, args.mac_beta, args.q_floor, args.stem_keep, args.q_floor_n)
    if args.arm == "unstructured":
        full = rw.unstructured_masks(specs, args.ratio, device)
    elif args.arm == "rewind":
        full = rw.expand(specs, masks_at(args.ratio), device)
    elif args.arm == "random":
        full = rw.expand(specs, random_masks(rw.build_masks(specs, args.ratio, "cluster", .5), args.seed), device)
    if full:
        rw.apply(specs, full)
    train_loader, val_loader, test_loader = build(args.dataset, args.seed, args.batch_size, args.workers, args.train_limit)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=args.weight_decay)
    scaler = core.make_scaler(device, "float16")
    core.reset_training_rng(args.seed+100000, train_loader)
    end, history, started = round(epochs*2/3), [], time.monotonic()
    for epoch in range(epochs):
        # Budgets other than 60 epochs stretch the same LR curve; lr_at itself only covers budgets up to 100.
        lr = rw.lr_at("rewind" if args.arm == "random" else args.arm, epoch*60/epochs, 60)
        for group in optimizer.param_groups:
            group["lr"] = lr
        if args.arm == "gradual" and epoch < end and epoch % 2 == 0:
            previous = full
            full = rw.expand(specs, masks_at(rw.gradual_target(epoch, end, args.ratio)), device)
            if previous is not None and any((full[n] & ~previous[n]).any() for n in full):
                raise ValueError("Gradual mask regrew a pruned position")
            rw.apply(specs, full)
        model.train()
        loss_sum, correct, seen = 0., 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with core.autocast_context(device, "float16"):
                logits = model(x)
                if teacher is None:
                    loss = F.cross_entropy(logits, y)
                else:
                    with torch.no_grad():
                        reference = teacher(x)
                    loss = rw.kd_loss(logits, reference, y, alpha=args.kd_alpha)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if full:
                rw.apply(specs, full, grads=True)
            scaler.step(optimizer)
            scaler.update()
            if full:
                rw.apply(specs, full)
            loss_sum += loss.item()*len(x)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(x)
        val_loss, val_acc = core.evaluate(model, val_loader, device, use_amp="float16")
        if loss_sum != loss_sum or val_loss != val_loss:
            raise FloatingPointError("Non-finite loss")
        row = dict(epoch=epoch+1, lr=lr, train_loss=loss_sum/seen, train_accuracy=correct/seen, validation_accuracy=val_acc, seconds=time.monotonic()-started)
        history.append(row)
        print(json.dumps(row), flush=True)
    _, test_acc = core.evaluate(model, test_loader, device, use_amp="float16")
    common.write_csv(directory/"training_history.csv", history)
    result = dict(dataset=args.dataset, model=args.model, seed=args.seed, arm=args.arm, ratio=args.ratio if args.arm != "dense" else 0, q_floor=args.q_floor, stem_keep=args.stem_keep, allocator=args.allocator, mac_beta=args.mac_beta if args.allocator == "mac" else None, q_floor_n=args.q_floor_n, kd=args.kd, kd_alpha=args.kd_alpha, weight_decay=args.weight_decay,
                  epochs=epochs, final_validation=val_acc, final_test=test_acc, final_train=correct/seen, seconds=time.monotonic()-started)
    if full:
        for spec in specs:
            if torch.count_nonzero(spec.module.weight[~full[spec.name].bool()]):
                raise ValueError("Forbidden weights are nonzero")
        result.update(rw.cost(specs, full))
        if args.arm != "unstructured":
            result["q_zero"] = sum(int((full[s.name].flatten(1)[::8].sum(1) == 0).sum()) for s in specs)
            result["zero_layers"] = [s.name for s in specs if not full[s.name].any()]
            torch.save({s.name: full[s.name][::8].flatten(1).cpu() for s in specs}, directory/"masks.pt")
    state = {n: v.detach().cpu() for n, v in model.state_dict().items()}
    if not args.discard_model:
        torch.save(state, directory/"final_model.pt")
    if args.arm == "dense":
        checkpoint(args).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, checkpoint(args))
    common.write_json(directory/"result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(DATA), required=True)
    parser.add_argument("--model", choices=["resnet34", "resnet50"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", choices=["dense", "rewind", "gradual", "unstructured", "random"], required=True)
    parser.add_argument("--ratio", type=float, default=.98)
    parser.add_argument("--q-floor", action="store_true")
    parser.add_argument("--stem-keep", action="store_true")
    parser.add_argument("--q-floor-n", type=int, default=1)
    parser.add_argument("--kd", action="store_true", help="Dense teacher with T=4, alpha=0.9")
    parser.add_argument("--kd-alpha", type=float, default=.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--allocator", choices=["cluster", "mac"], default="cluster")
    parser.add_argument("--mac-beta", type=float, default=.5)
    parser.add_argument("--discard-model", action="store_true", help="Development runs keep masks and history only, to save disk")
    parser.add_argument("--root", type=Path, default=Path("runs/other_datasets"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=0, help="Zero uses the frozen budget; other values stretch the LR curve and tag the run name")
    parser.add_argument("--train-limit", type=int, default=None)
    args = parser.parse_args()
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    run(args, torch.device("cuda"))


if __name__ == "__main__":
    main()
