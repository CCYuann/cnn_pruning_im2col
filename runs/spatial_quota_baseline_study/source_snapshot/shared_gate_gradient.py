"""First-order shared-gate proposals; neither gradient pruning nor swaps are novel."""

import torch


def gate_derivative(weight, gradient, shadow, masks, tm=8):
    """dL/dg for shared gates with current kept weights and Dense dormant weights."""
    full = masks.repeat_interleave(tm, 0)[:weight.shape[0]].reshape_as(weight).to(weight.device)
    values = torch.where(full, weight, shadow)
    product = (values * gradient).flatten(1)
    rows = []
    for cluster, start in enumerate(range(0, len(weight), tm)):
        rows.append(product[start:start+tm].sum(0))
    return torch.stack(rows)


def propose(scores, masks):
    """At most one favorable remove/add pair per cluster, with exact kept counts."""
    result = masks.clone()
    log = []
    for cluster, (derivative, mask) in enumerate(zip(scores, masks)):
        kept = mask.nonzero().flatten().to(derivative.device)
        dropped = (~mask).nonzero().flatten().to(derivative.device)
        if not len(kept) or not len(dropped):
            continue
        remove = int(kept[derivative[kept].argmax()])
        add = int(dropped[derivative[dropped].argmin()])
        delta = float(derivative[add] - derivative[remove])
        if delta < -1e-8:
            result[cluster, remove] = False
            result[cluster, add] = True
            log.append(dict(cluster=cluster, remove=remove, add=add, predicted_ce_delta=delta))
    if not torch.equal(result.sum(1), masks.sum(1)):
        raise ValueError("Shared-gate swap changed a quota")
    return result, log


def matched_random(original, gated, seed):
    generator = torch.Generator().manual_seed(seed)
    result = original.clone()
    for cluster, mask in enumerate(original):
        removed = int((mask & ~gated[cluster]).sum())
        added = int((~mask & gated[cluster]).sum())
        if removed != added or removed > 1:
            raise ValueError("Expected zero or one exchange per cluster")
        if not removed:
            continue
        kept, dropped = mask.nonzero().flatten(), (~mask).nonzero().flatten()
        a = kept[torch.randint(len(kept), (), generator=generator)]
        b = dropped[torch.randint(len(dropped), (), generator=generator)]
        result[cluster, a] = False
        result[cluster, b] = True
    return result
