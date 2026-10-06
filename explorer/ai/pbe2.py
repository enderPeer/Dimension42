"""Programming by example, trained on the behavior catalog of ALL 1.1 trillion 5-byte programs.

Each training example: pick a behavior uniformly from the catalog (so rare behaviors count as much as
"print a"), pick one of its example programs, run it on 8 fresh random input pairs, and train
examples -> program. Behaviors equal to a test function are removed from the catalog first, so the
model never sees a program for any test function. Tests as in pbe.py: 8 examples of the function ->
the model writes programs -> each is proven on all 65,536 input pairs.

usage: python pbe2.py [minutes]       (adler40; log pbe2.log, model pbe2.pt, results pbe2_results.json)
"""
import json
import math
import pathlib
import random
import sys
import time

MINUTES = float(sys.argv[1]) if len(sys.argv) > 1 else 120
sys.argv = ["x"]
import torch
import torch.nn.functional as F

import pbe
from pbe import BOS, K, L, SEP, NONE, V, GPT, run_many, evaluate, TESTS

HERE = pathlib.Path(__file__).resolve().parent
dev = "cuda"
torch.manual_seed(1)
logf = open(HERE / "pbe2.log", "a")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    logf.write(s + "\n")
    logf.flush()


PROBE_A = [3, 200, 17, 0, 255, 91, 128, 44, 7, 160, 66, 1, 230, 99, 12, 181]   # same probes as nano_catalog.cu
PROBE_B = [5, 13, 250, 0, 1, 77, 128, 210, 100, 33, 66, 254, 9, 140, 31, 2]


def fnv(outs):
    h = 1469598103934665603
    for o in outs:
        h = ((h ^ o) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h or 1


# ---------------- catalog
rows = []
for line in (HERE / "data" / "catalog.txt").read_text().splitlines():
    if line and not line.startswith("#"):
        h, c, ex = line.split()
        rows.append((int(h, 16), int(c), [int(e, 16) for e in ex.split(",")]))
test_hash = {name: fnv([f(a, b) for a, b in zip(PROBE_A, PROBE_B)]) for name, f in TESTS.items()}
in_catalog = {name: next((c for h, c, _ in rows if h == th), 0) for name, th in test_hash.items()}
held = set(test_hash.values())
train = [r for r in rows if r[0] not in held]
log(f"catalog: {len(rows):,} behaviors; {len(rows) - len(train)} removed (test functions); training on {len(train):,}")
for name, c in in_catalog.items():
    log(f"  test '{name}': {c:,} programs in the whole 5-byte space have this behavior (random chance {c / 2**40:.2e})")

# all example programs as one tensor, plus per-behavior offsets, for fast uniform-behavior sampling
ex_list, off = [], [0]
for _, _, ex in train:
    ex_list += ex
    off.append(len(ex_list))
EX = torch.tensor([[(p >> (8 * (L - 1 - i))) & 255 for i in range(L)] for p in ex_list], device=dev)
OFF = torch.tensor(off, device=dev)
NB = len(train)


def batch(n):
    beh = torch.randint(0, NB, (n,), device=dev)
    cnt = OFF[beh + 1] - OFF[beh]
    pick = OFF[beh] + (torch.rand(n, device=dev) * cnt).long()
    prog = EX[pick]
    a = torch.randint(0, 256, (n, K), device=dev)
    b = torch.randint(0, 256, (n, K), device=dev)
    o = run_many(prog, a, b)
    o = torch.where(o < 0, torch.full_like(o, NONE), o)
    ex = torch.stack([a, b, o], 2).reshape(n, 3 * K)
    return torch.cat([torch.full((n, 1), BOS, device=dev), ex, torch.full((n, 1), SEP, device=dev), prog], 1)


model = GPT().to(dev)
log(f"model parameters: {sum(p.numel() for p in model.parameters()):,}; training {MINUTES:.0f} minutes")
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95))
BS, t0, step, next_eval, history = 1024, time.time(), 0, 0.0, []
while True:
    el = (time.time() - t0) / 60
    lr = 3e-4 * min(1.0, step / 2000) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, el / MINUTES))))
    for gp in opt.param_groups:
        gp["lr"] = lr
    x = batch(BS)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x[:, :-1])
    loss = F.cross_entropy(logits[:, -L:].float().reshape(-1, V), x[:, -L:].reshape(-1))
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    step += 1
    if step % 2000 == 0:
        log(f"step {step}  {el:5.1f} min  loss {loss.item():.3f}  ({step * BS:,} examples)")
    if el >= next_eval or el >= MINUTES:
        res = evaluate(model)
        history.append({"minutes": round(el, 1), "step": step, "tests": res})
        log(f"--- test at {el:.1f} min: " + "  ".join(f"{k}: {v['prompts_solved']}" for k, v in res.items()))
        (HERE / "pbe2_results.json").write_text(json.dumps(
            {"catalog_behaviors": len(rows), "trained_behaviors": NB, "test_programs_in_space": in_catalog,
             "history": history}, indent=1))
        torch.save(model.state_dict(), HERE / "pbe2.pt")
        next_eval += 15
        if el >= MINUTES:
            break
log("done")
