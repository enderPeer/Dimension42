"""Check nano_catalog (CUDA kernel) against the verified PyTorch interpreter (pbe.run) on one range."""
import collections
import subprocess
import sys

import torch

ARGS = sys.argv[1:]
sys.argv = ["x"]
import pbe

PA = torch.tensor([3, 200, 17, 0, 255, 91, 128, 44, 7, 160, 66, 1, 230, 99, 12, 181], device="cuda")
PB = torch.tensor([5, 13, 250, 0, 1, 77, 128, 210, 100, 33, 66, 254, 9, 140, 31, 2], device="cuda")


def fnv(outs):
    h = 1469598103934665603
    for o in outs:
        h = ((h ^ o) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h or 1


start, count = int(ARGS[0], 0) if ARGS else 0x4500000000, 1 << 22
kernel = collections.Counter()
out = subprocess.run(["../nano_catalog", "serve", "0"], input=f"{start} {count}\n", capture_output=True, text=True).stdout
lines = out.splitlines()
print(lines[0])
for l in lines[1:]:
    h, c, e = l.split()
    kernel[int(h, 16)] = int(c)

ref = collections.Counter()
for s in range(start, start + count, 1 << 18):
    idx = torch.arange(s, s + (1 << 18), device="cuda")
    prog = torch.stack([(idx >> (8 * (4 - i))) & 255 for i in range(5)], 1)
    outs = pbe.run_many(prog, PA.expand(len(prog), 16), PB.expand(len(prog), 16))
    ok = (outs >= 0).all(1) & (outs != outs[:, :1]).any(1)
    for row in outs[ok].tolist():
        ref[fnv(row)] += 1
print(f"kernel: {len(kernel)} behaviors, {sum(kernel.values()):,} programs | PyTorch: {len(ref)} behaviors, "
      f"{sum(ref.values()):,} programs | identical: {kernel == ref}")
