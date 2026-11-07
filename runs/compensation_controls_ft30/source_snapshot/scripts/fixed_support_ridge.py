"""Prior-centered ridge compensation with an immutable shared support."""

import time

import torch


def positions_for_layer(images, area, limit, seed):
    if min(images, area, limit) <= 0:
        raise ValueError("Positive sampling dimensions required")
    count = min(area, max(1, limit // images))
    generator = torch.Generator().manual_seed(seed)
    return torch.stack([torch.randperm(area, generator=generator)[:count] for _ in range(images)])


def sampled_output(output, positions):
    selected = output.flatten(2).gather(2, positions[:, None, :].expand(-1, output.shape[1], -1))
    return selected.permute(1, 0, 2).reshape(output.shape[1], -1).contiguous()


@torch.no_grad()
def compensate(weight, mask, fit_x, fit_y, check_x, check_y):
    started = time.monotonic()
    if weight.ndim != 2 or mask.dtype != torch.bool or mask.shape != weight.shape[1:]:
        raise ValueError("Expected [filters,K] weights and a shared boolean [K] mask")
    for x, y in [(fit_x, fit_y), (check_x, check_y)]:
        if x.ndim != 2 or y.shape != (len(weight), x.shape[1]) or len(x) != weight.shape[1] or not x.shape[1]:
            raise ValueError("Misaligned reconstruction data")
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError("Non-finite reconstruction data")
    if not torch.isfinite(weight).all():
        raise ValueError("Non-finite weights")
    mask = mask.to(weight.device)
    q = int(mask.sum())
    original = weight[:, mask].double()
    x = fit_x[mask].double()
    z = check_x[mask].double()
    y, target = fit_y.double(), check_y.double()
    before_fit = float((original @ x-y).square().mean())
    before_check = float((original @ z-target).square().mean())
    regularizer, jitter_attempt = 0., 0
    value = original
    status = "empty" if q == 0 else "full" if q == weight.shape[1] else "accepted"
    raw_check = before_check
    if status == "accepted":
        gram = x @ x.T / x.shape[1]
        regularizer = max(1e-8, .001*float(gram.diag().mean()))
        rhs = x @ (y-original @ x).T / x.shape[1]
        for attempt in range(3):
            jitter_attempt = attempt
            matrix = gram.clone()
            matrix.diagonal().add_(regularizer * 10**attempt)
            factor, info = torch.linalg.cholesky_ex(matrix)
            if int(info) == 0:
                value = original + torch.cholesky_solve(rhs, factor).T
                break
        else:
            raise ArithmeticError("Cholesky failed after fixed numerical jitter attempts")
        if not torch.isfinite(value).all():
            raise ArithmeticError("Non-finite compensation solution")
        # Validate the actual stored FP32 coefficients, not just the FP64 solution.
        value = value.to(weight.dtype).double()
        raw_check = float((value @ z-target).square().mean())
        if raw_check > before_check:
            value, status = original, "fallback"
    result = torch.zeros_like(weight)
    result[:, mask] = value.to(weight.dtype)
    if torch.count_nonzero(result[:, ~mask]):
        raise AssertionError("Compensation changed forbidden positions")
    report = dict(q=q, status=status, lambda_base=regularizer, jitter_attempt=jitter_attempt,
                  fit_mse_before=before_fit, fit_mse_after=float((value @ x-y).square().mean()),
                  check_mse_before=before_check, candidate_check_mse=raw_check,
                  check_mse_after=float((value @ z-target).square().mean()),
                  weight_delta_l2=float((value-original).norm()), original_kept_l2=float(original.norm()),
                  original_abs_max=float(original.abs().max()) if q else 0.,
                  compensated_abs_max=float(value.abs().max()) if q else 0.,
                  fit_samples=x.shape[1], check_samples=z.shape[1],
                  seconds=time.monotonic()-started)
    return result, report
