"""Patch-position activation RMS for fixed-quota standard-convolution pruning."""

import torch
from torch.nn import functional as F


def patch_moments(inputs, conv):
    if conv.groups != 1 or conv.padding_mode != "zeros":
        raise ValueError("Only standard zero-padded convolutions are supported")
    patches = F.unfold(inputs.detach(), kernel_size=conv.kernel_size, dilation=conv.dilation,
                       padding=conv.padding, stride=conv.stride)
    if patches.shape[1] != conv.weight[0].numel():
        raise ValueError("Unfold K mismatch")
    squared_sum = patches.double().square().sum(dim=(0,2))
    count = patches.shape[0]*patches.shape[2]
    if count <= 0 or not bool(torch.isfinite(squared_sum).all()):
        raise ValueError("Invalid activation moments")
    return squared_sum, count


def rms_from_moments(squared_sum, count):
    if count <= 0 or bool((squared_sum < 0).any()) or not bool(torch.isfinite(squared_sum).all()):
        raise ValueError("Invalid activation moments")
    return (squared_sum.double()/count).sqrt()


def activation_scores(weights, rms):
    flat = weights.flatten(1).double().abs()
    if rms.ndim != 1 or len(rms) != flat.shape[1] or bool((rms<0).any()) or not bool(torch.isfinite(rms).all()):
        raise ValueError("Invalid activation RMS")
    return flat*rms.to(flat.device).unsqueeze(0)


def explicit_sample_moments(inputs, conv):
    """Independent slicing reference for small diagnostic batches."""
    ph,pw=conv.padding
    sh,sw=conv.stride
    dh,dw=conv.dilation
    kh,kw=conv.kernel_size
    x=F.pad(inputs.detach().double(),(pw,pw,ph,ph))
    oh=(inputs.shape[2]+2*ph-dh*(kh-1)-1)//sh+1
    ow=(inputs.shape[3]+2*pw-dw*(kw-1)-1)//sw+1
    values=[]
    for channel in range(inputs.shape[1]):
        for row in range(kh):
            for col in range(kw):
                patch=x[:,channel,row*dh:row*dh+oh*sh:sh,col*dw:col*dw+ow*sw:sw]
                values.append(patch.square().sum())
    return torch.stack(values),inputs.shape[0]*oh*ow
