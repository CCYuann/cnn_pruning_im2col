"""Fixed-quota logical grouping and bounded, score-only local exchange search."""

import torch


VARIANTS = ["contiguous", "random0", "random1", "random2", "norm_sorted", "jaccard", "jaccard_swap"]


def validate_groups(groups, filters, tm=8):
    if filters % tm or groups.shape != (filters // tm, tm):
        raise ValueError("This experiment requires complete equal-size clusters")
    if not torch.equal(groups.flatten().sort().values.cpu(), torch.arange(filters)):
        raise ValueError("Groups must partition all filters exactly once")


def discarded(scores, q):
    if not 0 <= q <= scores.shape[-1]:
        raise ValueError("Invalid kept quota")
    if q == 0:
        return scores.sum(-1)
    if q == scores.shape[-1]:
        return scores.sum(-1) * 0
    removed = scores.shape[-1] - q
    if removed <= q:
        return scores.topk(removed, dim=-1, largest=False).values.sum(-1)
    return scores.sum(-1) - scores.topk(q, dim=-1).values.sum(-1)


def grouping_loss(weights, groups, q):
    validate_groups(groups, len(weights), groups.shape[1])
    return discarded(weights.double().abs()[groups].sum(1), q).sum()


def closure(baseline, candidate, lower):
    gap = baseline - lower
    if gap <= max(1e-10, abs(baseline) * 1e-12):
        return None
    return (baseline - candidate) / gap


def shared_masks(weights, groups, q):
    validate_groups(groups, len(weights), groups.shape[1])
    score = weights.double().abs()[groups].sum(1)
    masks = torch.zeros_like(score, dtype=torch.bool)
    # Stable ordering defines deterministic behavior when importance is tied.
    kept = torch.argsort(score, dim=1, descending=True, stable=True)[:, :q]
    masks.scatter_(1, kept, True)
    full = torch.zeros_like(weights, dtype=torch.bool)
    full[groups] = masks[:, None, :].expand(-1, groups.shape[1], -1)
    if not bool((full.sum(1) == q).all()):
        raise ValueError("Per-filter quota changed")
    return masks, full


@torch.no_grad()
def exchange_search(weights, initial, q, seed, rounds=4, batch_size=64):
    groups = initial.clone()
    validate_groups(groups, len(weights), groups.shape[1])
    weights = weights.double().abs()
    before = float(grouping_loss(weights, groups, q))
    history = []
    if len(groups) < 2 or q in (0, weights.shape[1]):
        return groups, dict(before=before, after=before, rounds=history, accepted=0)
    sums = weights[groups].sum(1)
    generator = torch.Generator().manual_seed(seed)
    proposals = min(4096, max(128, 2 * len(weights)))
    accepted_total = 0
    for iteration in range(rounds):
        a = torch.randint(len(groups), (proposals,), generator=generator)
        b = torch.randint(len(groups)-1, (proposals,), generator=generator)
        b += b >= a
        ia = torch.randint(groups.shape[1], (proposals,), generator=generator)
        ib = torch.randint(groups.shape[1], (proposals,), generator=generator)
        moves = torch.stack([a, b, ia, ib], 1).to(groups.device)
        deltas = []
        for chunk in moves.split(batch_size):
            ga, gb, pa, pb = chunk.unbind(1)
            wa, wb = weights[groups[ga, pa]], weights[groups[gb, pb]]
            old = discarded(sums[ga], q) + discarded(sums[gb], q)
            new = discarded(sums[ga]-wa+wb, q) + discarded(sums[gb]-wb+wa, q)
            deltas.append((new-old).cpu())
        delta = torch.cat(deltas)
        used, accepted = set(), 0
        for index in torch.argsort(delta, stable=True).tolist():
            if float(delta[index]) >= -1e-10:
                break
            ga, gb, pa, pb = moves[index].tolist()
            if ga in used or gb in used:
                continue
            fa, fb = int(groups[ga, pa]), int(groups[gb, pb])
            groups[ga, pa], groups[gb, pb] = fb, fa
            sums[ga] += weights[fb]-weights[fa]
            sums[gb] += weights[fa]-weights[fb]
            used.update([ga, gb])
            accepted += 1
        accepted_total += accepted
        history.append(dict(iteration=iteration+1, proposals=proposals, accepted=accepted,
                            cost=float(discarded(sums, q).sum())))
        if not accepted:
            break
    after = float(grouping_loss(weights, groups, q))
    if after > before + max(1e-9, abs(before)*1e-12):
        raise ValueError("Exchange search increased the objective")
    validate_groups(groups, len(weights), groups.shape[1])
    return groups, dict(before=before, after=after, rounds=history, accepted=accepted_total)
