"""Per-channel follow-up: are big-bias / big-shift channels constant? Is clamping bias to 28-bit and right_shift to a cap bit-exact?"""
import sys, torch, collections
sys.path.insert(0, "scripts")
import hw_range_report as H
from int8_hw_spec_eval import IntConv, FRACTION, quantize_multiplier
import resnet34_pruning_experiments as core
device = torch.device("cuda")
orig = IntConv.__call__
for label, dataset, path, note in H.MODELS:
    model = core.make_model("resnet34", 10 if dataset == "cifar10" else 100, 32).to(device).eval()
    model.load_state_dict(torch.load(path+"/final_model.pt", map_location=device, weights_only=False))
    cal, val = H.data(dataset, 5000)
    with torch.inference_mode():
        H.equalize(model, cal, device, 8.); scale, logit = H.calibrate(model, cal, device, .5)
        integer = H.IntegerResNet(model, scale, logit, device); integer.correct_bias(model, next(iter(cal))[0][:256])
    H.name_layers(model, integer)
    convs = [integer.stem]+[c for ops, last, short, _, _ in integer.items for c in [*ops, last]+([short] if short is not None else [])]
    # per-channel output range and |acc| max
    outr = {}
    def rec(self, x):
        acc = torch.nn.functional.conv2d(x.double(), self.weight, stride=self.stride, padding=self.padding).round().long()
        y = orig(self, x)
        lo, hi, am = y.amin((0,2,3)).flatten(), y.amax((0,2,3)).flatten(), acc.abs().amax((0,2,3)).flatten()
        if self.name in outr:
            a,b,c = outr[self.name]; lo,hi,am = torch.minimum(a,lo), torch.maximum(b,hi), torch.maximum(c,am)
        outr[self.name] = (lo,hi,am); return y
    IntConv.__call__ = rec
    ref = []
    with torch.inference_mode():
        for x, y in val: ref.append(integer(x).cpu())
    IntConv.__call__ = orig
    ref = torch.cat(ref)
    big_bias = big_shift = const_bb = const_bs = 0; shifts = collections.Counter()
    for c in convs:
        b = c.bias.flatten(); rs = c.right_shift.flatten(); lo, hi, am = outr[c.name]
        raw = torch.tensor([FRACTION-quantize_multiplier(float(u)/c.s_out, FRACTION).shift for u in c.unit.flatten().tolist()], device=device)
        for r in raw.tolist(): shifts[min(r, 99)] += 1
        const = (lo == hi)
        m = b.abs() >= 2**27; big_bias += int(m.sum()); const_bb += int((m & const).sum())
        m2 = raw > 32; big_shift += int(m2.sum()); const_bs += int((m2 & const).sum())
    # Bit-exactness tests: (a) bias clamped to signed 28-bit minus acc headroom, (b) right_shift capped at various values
    def run_with(bias_cap=None, shift_cap=None):
        saved = [(c.bias.clone(), c.right_shift.clone()) for c in convs]
        for c in convs:
            if bias_cap: c.bias = c.bias.clamp(-bias_cap, bias_cap)
            if shift_cap: c.right_shift = c.right_shift.clamp_max(shift_cap)
        out = []
        with torch.inference_mode():
            for x, y in val: out.append(integer(x).cpu())
        for c, (b, r) in zip(convs, saved): c.bias, c.right_shift = b, r
        out = torch.cat(out); return bool(torch.equal(out, ref)), int((out.argmax(1) != ref.argmax(1)).sum())
    accmax = max(int(outr[c.name][2].max()) for c in convs)
    cap = 2**27-1-accmax
    res = {"bias_clamp_28bit": run_with(bias_cap=cap)}
    for s in (40, 36, 32, 30):
        res[f"shift_cap_{s}"] = run_with(shift_cap=s)
    print(label, f"| ch |bias|>=2^27: {big_bias} (constant output {const_bb}) | ch raw shift>32: {big_shift} (constant {const_bs}) | max|acc| {accmax} (bits {accmax.bit_length()+1})")
    print("   raw right_shift histogram:", dict(sorted(shifts.items())))
    print("   bit-exact (logits identical, top-1 flips):", res, flush=True)
