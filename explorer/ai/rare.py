"""What do the behaviors that exist only once (one program in 1.1 trillion) compute? Full-input analysis."""
import collections
import random
import sys

sys.argv = ["x"]
import torch
import pbe

A = torch.arange(256, device="cuda").repeat_interleave(256)
B = torch.arange(256, device="cuda").repeat(256)


def full(p):
    prog = torch.tensor([[(p >> (8 * (4 - i))) & 255 for i in range(5)]], device="cuda").expand(65536, 5)
    return pbe.run(prog, A, B)


for p in (0x66342B51E2, 0x4663534AD0, 0x5C94C386C3):
    o = full(p).tolist()
    c = collections.Counter(o)
    main, n = c.most_common(1)[0]
    exc = [(i // 256, i % 256, v) for i, v in enumerate(o) if v != main]
    bs, as_ = sorted({b for a, b, v in exc}), sorted({a for a, b, v in exc})
    print(f"{p:010X}: answer {main} for {100 * n / 65536:.2f}% of all inputs, {len(c)} different answers")
    print(f"   exceptions: {len(exc):,} inputs, involving {len(bs)} values of b (e.g. {bs[:10]}) and {len(as_)} values of a")
    print(f"   examples (a, b, answer): {random.Random(1).sample(exc, min(6, len(exc)))}")

ones = [int(l.split()[2], 16) for l in open("data/catalog.txt") if not l.startswith("#") and l.split()[1] == "1"]
random.seed(3)
share, nans = [], []
for p in random.sample(ones, 2000):
    v, cnt = torch.unique(full(p), return_counts=True)
    share.append(cnt.max().item() / 65536)
    nans.append(len(v))
share.sort()
nans.sort()
print(f"\n2,000 unique-behavior programs on all 65,536 inputs: their most common answer covers "
      f"{100 * share[1000]:.1f}% of inputs (median); for 90% of them it covers more than {100 * share[200]:.1f}%. "
      f"Different answers per program: median {nans[1000]}, max {nans[-1]}")


# ---- a typical one: many different answers
random.seed(11)
for p in random.sample(ones, 400):
    o = full(p)
    v, cnt = torch.unique(o, return_counts=True)
    if len(v) >= 250 and cnt.max().item() / 65536 < 0.02:
        break
OPS = "NOP LDI ADDI SUBI LD ST ADD JMP JZ INC DEC ROL XOR OUT NOT HLT".split()
bs = [(p >> (8 * (4 - i))) & 255 for i in range(5)]
print(f"\ntypical: {p:010X} = " + " . ".join(f"{OPS[x >> 4]} {x & 15:X}" for x in bs))
o = o.view(256, 256)
print("      b=" + "".join(f"{b:5d}" for b in range(8)))
for a in range(8):
    print(f"a={a:3d} " + "".join(f"{int(o[a, b]):5d}" for b in range(8)))
