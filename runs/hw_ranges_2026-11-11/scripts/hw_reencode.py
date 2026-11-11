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
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    norm = H.MEAN, H.STD
    if dataset == "cifar100": norm = H.DATA["cifar100"]["norm"]
    Dset = datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100
    test = DataLoader(Dset("data", train=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(*norm)])), batch_size=250, num_workers=2)
    def logits(loader):
        with torch.inference_mode(): return torch.cat([integer(x).cpu() for x, _ in loader])
    ref_val, ref_test = ref, logits(test)
    n = 0
    for c in convs:
        lo, hi, am = outr[c.name]; const = (lo == hi).flatten()
        if not const.any(): continue
        n += int(const.sum())
        w = c.weight.clone(); w[const] = 0; c.weight = w
        b = c.bias.clone().flatten(); b[const] = lo.flatten()[const].long(); c.bias = b.view_as(c.bias)
        m = c.multiplier.clone().expand_as(c.right_shift).clone().flatten(); m[const] = 1 << 14; c.multiplier = m.view(1, -1, 1, 1)
        r = c.right_shift.clone().flatten(); r[const] = 14; c.right_shift = r.view(1, -1, 1, 1)
    new_val, new_test = logits(val), logits(test)
    rs = torch.cat([c.right_shift.flatten() for c in convs]); bb = torch.cat([c.bias.flatten() for c in convs]).abs().max()
    # widest acc+bias after re-encoding, on validation
    peak = [0]
    def rec2(self, x):
        a = torch.nn.functional.conv2d(x.double(), self.weight, stride=self.stride, padding=self.padding).round().long()+self.bias
        peak[0] = max(peak[0], int(a.abs().max())); return orig(self, x)
    IntConv.__call__ = rec2; logits(val); IntConv.__call__ = orig
    print(label, f"re-encoded {n} channels | val logits identical {torch.equal(new_val, ref_val)} | test logits identical {torch.equal(new_test, ref_test)} (top-1 changes {int((new_test.argmax(1)!=ref_test.argmax(1)).sum())}/10000) | right_shift now {int(rs.min())}..{int(rs.max())} | max|bias| bits {int(bb).bit_length()+1} | max|acc+bias| bits {peak[0].bit_length()+1}", flush=True)
