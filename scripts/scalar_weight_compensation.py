"""A one-parameter-per-filter reconstruction control on an immutable support."""

import torch


@torch.no_grad()
def compensate_scalar(weight, mask, fit_x, fit_y, check_x, check_y):
    if weight.ndim!=2 or mask.dtype!=torch.bool or mask.shape!=weight.shape[1:]:
        raise ValueError("Expected weights[filters,K] and shared mask[K]")
    for x,y in [(fit_x,fit_y),(check_x,check_y)]:
        if x.ndim!=2 or x.shape[0]!=weight.shape[1] or not x.shape[1] or y.shape!=(len(weight),x.shape[1]):
            raise ValueError("Unaligned scalar reconstruction samples")
        if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
            raise ValueError("Non-finite calibration samples")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("Non-finite original weights")
    mask=mask.to(weight.device)
    original=weight[:,mask].double()
    x,z=fit_x[mask].double(),check_x[mask].double()
    y,target=fit_y.double(),check_y.double()
    prediction=original@x
    before_fit=float((prediction-y).square().mean())
    before_check=float((original@z-target).square().mean())
    q=int(mask.sum())
    alpha=torch.ones(len(weight),dtype=torch.float64,device=weight.device)
    status="empty" if q==0 else "full" if q==weight.shape[1] else "accepted"
    value=original
    raw_check=before_check
    if status=="accepted":
        den=prediction.square().sum(1)
        valid=den>1e-12
        alpha[valid]=(prediction*y).sum(1)[valid]/den[valid]
        value=(original*alpha[:,None]).to(weight.dtype).double()
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError("Non-finite scalar coefficients")
        raw_check=float((value@z-target).square().mean())
        if raw_check>before_check:
            value,status=original,"fallback"
    result=torch.zeros_like(weight)
    result[:,mask]=value.to(weight.dtype)
    return result,dict(q=q,status=status,alpha=alpha.cpu().tolist(),fit_mse_before=before_fit,
        fit_mse_after=float((value@x-y).square().mean()),check_mse_before=before_check,
        candidate_check_mse=raw_check,check_mse_after=float((value@z-target).square().mean()))
