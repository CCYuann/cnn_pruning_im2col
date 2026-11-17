"""Partial-sum worst case for the provable rewrite (hw_provable_analysis.py), for bias preload (F).

With the bias preloaded, the accumulator holds bias + the sum of a subset of the kept terms at every step.
The bound here is the extreme of that over ANY subset, so it holds for any q order and any input image.
Runs the same analysis and prints, per model, the final-sum bound and the partial-sum bound in signed bits.

    python3 scripts/hw_provable_partial_bound.py <output directory for the rerun csv files>
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import hw_provable_analysis as hp

hp.OUT = Path(sys.argv[1])
store = {}
original = hp.conv_interval

def wrapped(op, lo, hi):
    out = original(op, lo, hi)
    l, h = lo, hi
    if op.padding[0] > 0:
        l, h = l.clamp(max=0), h.clamp(min=0)
    w = op.weight.double()
    wp, wn = w.clamp(min=0), w.clamp(max=0)
    l, h = l.double()[None, :, None, None], h.double()[None, :, None, None]
    t_hi, t_lo = wp*h+wn*l, wp*l+wn*h          # per-term extremes
    bias = op.bias.flatten().double()
    p_hi = bias+t_hi.clamp(min=0).sum((1, 2, 3))  # largest value any partial sum can take
    p_lo = bias+t_lo.clamp(max=0).sum((1, 2, 3))
    no_bias_hi, no_bias_lo = t_hi.clamp(min=0).sum((1, 2, 3)), t_lo.clamp(max=0).sum((1, 2, 3))
    store[op.name] = dict(partial=hp.bits(int(p_lo.min().floor()), int(p_hi.max().ceil())), final=hp.bits(int(out[2].min()), int(out[3].max())),
                          acc_only=hp.bits(int(no_bias_lo.min().floor()), int(no_bias_hi.max().ceil())), pad=op.padding[0])
    return out

hp.conv_interval = wrapped
rows = {}
real_analyse = hp.analyse
def analyse(*a, **k):
    store.clear()
    r = real_analyse(*a, **k)
    rows[len(rows)] = dict(store)
    return r
hp.analyse = analyse
hp.main()
labels = [m[0] for m in hp.MODELS]
for i, label in enumerate(labels):
    s = rows[i]
    worse = {n: (v["final"], v["partial"]) for n, v in s.items() if v["partial"] > v["final"]}
    print(label, "final max", max(v["final"] for v in s.values()), "partial max", max(v["partial"] for v in s.values()),
          "acc-only (no bias) partial max", max(v["acc_only"] for v in s.values()), "layers where partial > final:", worse)
