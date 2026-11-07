"""Fixed-cardinality shared-mask repair and deterministic convolution patch sampling."""

import torch
from torch.nn import functional as F


def sample_patches(value, conv, positions):
    """Return [Cin*Kh*Kw, samples] in the same K order as Conv2d weights."""
    kh, kw = conv.kernel_size
    ph, pw = conv.padding
    sh, sw = conv.stride
    dh, dw = conv.dilation
    out_w = (value.shape[3] + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    padded = F.pad(value, (pw, pw, ph, ph))
    channels = torch.arange(value.shape[1], device=value.device).repeat_interleave(kh * kw)
    dy = torch.arange(kh, device=value.device).repeat_interleave(kw).repeat(value.shape[1]) * dh
    dx = torch.arange(kw, device=value.device).repeat(kh).repeat(value.shape[1]) * dw
    image = torch.arange(value.shape[0], device=value.device)[:, None].expand_as(positions).reshape(-1)
    flat = positions.reshape(-1)
    yy, xx = (flat // out_w) * sh, (flat % out_w) * sw
    return padded[image[:, None], channels[None, :], yy[:, None] + dy, xx[:, None] + dx].t().contiguous()


def cluster_inputs(patches, conv, start, end):
    if conv.groups == 1:
        return patches.unsqueeze(0)
    k = conv.weight[0].numel()
    group_ids = torch.arange(start, end, device=patches.device) // (conv.out_channels // conv.groups)
    return patches.reshape(conv.groups, k, -1).index_select(0, group_ids)


def project(weight, inputs):
    if inputs.shape[0] == 1:
        return weight @ inputs[0]
    return torch.einsum("mk,mkp->mp", weight, inputs)


def errors(weight, inputs, mask, epsilon, valid=None):
    dense = project(weight, inputs)
    residual = project(weight * (~mask)[None, :], inputs)
    energy = dense.square().mean(1)
    denominator = energy + epsilon
    if valid is None:
        valid = energy >= epsilon
    relative = residual.square().mean(1) / denominator
    return relative, valid, residual, denominator


def metrics(weight, inputs, mask, epsilon):
    relative, valid, _, _ = errors(weight, inputs, mask, epsilon)
    return {"mean_error": relative[valid].mean().item() if valid.any() else 0.0,
            "max_error": relative[valid].max().item() if valid.any() else 0.0,
            "valid_filters": int(valid.sum()), "excluded_filters": int((~valid).sum()),
            "near_total_loss_filters": int(((relative >= .9) & valid).sum())}


@torch.no_grad()
def repair(weight, inputs, mask, epsilon, objective="worst", candidates=8, iterations=8):
    mask = mask.clone()
    relative, valid, residual, denominator = errors(weight, inputs, mask, epsilon)
    initial_mean = relative[valid].mean() if valid.any() else relative.new_tensor(0)
    swaps = []
    if not valid.any() or not mask.any() or mask.all():
        return mask, swaps
    n = inputs.shape[-1]
    energy_x = inputs.square().mean(-1)
    if inputs.shape[0] == 1:
        energy_x = energy_x.expand(weight.shape[0], -1)
    for _ in range(iterations):
        correlation = residual @ inputs[0].t() / n if inputs.shape[0] == 1 else torch.einsum("mp,mkp->mk", residual, inputs) / n
        linear = 2 * weight * correlation
        quadratic = weight.square() * energy_x
        remove_delta = (linear + quadratic) / denominator[:, None]
        add_delta = (-linear + quadratic) / denominator[:, None]
        kept, absent = mask.nonzero().flatten(), (~mask).nonzero().flatten()
        remove_score = remove_delta[valid].mean(0)[kept]
        worst = torch.where(valid, relative, -torch.inf).argmax()
        add_score = (add_delta[worst] if objective == "worst" else add_delta[valid].mean(0))[absent]
        aa = kept[torch.argsort(remove_score, stable=True)[:candidates]]
        bb = absent[torch.argsort(add_score, stable=True)[:candidates]]
        if inputs.shape[0] == 1:
            covariance = (inputs[0, aa] @ inputs[0, bb].t() / n).unsqueeze(0)
        else:
            covariance = torch.bmm(inputs[:, aa], inputs[:, bb].transpose(1, 2)) / n
        cross = 2 * weight[:, aa, None] * weight[:, None, bb] * covariance / denominator[:, None, None]
        trial = relative[:, None, None] + remove_delta[:, aa, None] + add_delta[:, None, bb] - cross
        mean = trial[valid].mean(0)
        maximum = trial[valid].amax(0)
        tolerance = 1e-6
        if objective == "worst":
            acceptable = (mean <= initial_mean + tolerance) & (maximum < relative[valid].max() - tolerance)
            score = maximum
        elif objective == "mean":
            acceptable = mean < relative[valid].mean() - tolerance
            score = mean
        else:
            raise ValueError(objective)
        if not acceptable.any():
            break
        flat = torch.where(acceptable, score, torch.inf).flatten().argmin().item()
        a, b = aa[flat // len(bb)].item(), bb[flat % len(bb)].item()
        changed = mask.clone()
        changed[a], changed[b] = False, True
        actual, _, next_residual, _ = errors(weight, inputs, changed, epsilon, valid)
        torch.testing.assert_close(actual, trial[:, flat // len(bb), flat % len(bb)], atol=2e-5, rtol=2e-4)
        if objective == "worst" and (actual[valid].mean() > initial_mean + tolerance or actual[valid].max() >= relative[valid].max() - tolerance):
            break
        if objective == "mean" and actual[valid].mean() >= relative[valid].mean() - tolerance:
            break
        swaps.append({"remove_k": a, "add_k": b, "mean_error": actual[valid].mean().item(), "max_error": actual[valid].max().item()})
        mask, relative, residual = changed, actual, next_residual
    return mask, swaps


def normalized_mask(weight, inputs, q):
    energy = weight.square() * inputs.square().mean(-1)
    fraction = energy / energy.sum(1, keepdim=True).clamp_min(1e-30)
    mask = torch.zeros(weight.shape[1], dtype=torch.bool, device=weight.device)
    mask[torch.argsort(fraction.sum(0), descending=True, stable=True)[:q]] = True
    return mask


def random_swaps(mask, count, seed):
    generator = torch.Generator(device=mask.device).manual_seed(seed)
    mask = mask.clone()
    for _ in range(count):
        aa, bb = mask.nonzero().flatten(), (~mask).nonzero().flatten()
        if not len(aa) or not len(bb):
            break
        a = aa[torch.randint(len(aa), (), generator=generator, device=mask.device)]
        b = bb[torch.randint(len(bb), (), generator=generator, device=mask.device)]
        mask[a], mask[b] = False, True
    return mask
