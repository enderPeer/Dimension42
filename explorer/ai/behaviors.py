"""How many different behaviors do random 5-byte NANO programs have? (a, b inputs, first OUT answer, 16 probe inputs)"""
import collections
import sys
import time

import torch

sys.argv = ["x"]
import pbe

cnt, ex, tot, t = collections.Counter(), {}, 0, time.time()
for it in range(400):
    prog = torch.randint(0, 256, (1 << 18, 5), device="cuda")
    outs = pbe.run_many(prog, pbe.PA.expand(len(prog), 16), pbe.PB.expand(len(prog), 16))
    ok = (outs >= 0).all(1) & (outs != outs[:, :1]).any(1)
    tot += len(prog)
    pg = prog[ok]
    for i, s in enumerate(pbe.signature(outs[ok]).tolist()):
        cnt[s] += 1
        ex.setdefault(s, bytes(pg[i].tolist()).hex().upper())
    if it in (9, 39, 99, 199, 399):
        print(f"{tot / 1e6:6.0f} M random programs: {sum(cnt.values()):,} useful, {len(cnt):,} different behaviors, "
              f"{sum(1 for v in cnt.values() if v == 1):,} seen only once ({time.time() - t:.0f} s)", flush=True)
names = {int(pbe.signature(f(pbe.PA, pbe.PB)[None])[0]): n for n, f in pbe.TESTS.items()}
names[int(pbe.signature(pbe.PA[None])[0])] = "a"
names[int(pbe.signature(pbe.PB[None])[0])] = "b"
print("most common behaviors:")
for s, c in cnt.most_common(15):
    print(f"  {c:>10,}  {names.get(s, '?'):16s} e.g. {ex[s]}")
