"""Train a small transformer to WRITE NANO programs that add, from the proven adders found by the search.

Task given to the model:  ADD  <cell of a>  <cell of b>  SEP
Model writes:             10 hex digits = one 5-byte NANO program
Every program it writes is then checked exactly: run on all 65,536 input pairs (nano_search verify).

Training layouts and test layouts are disjoint:
  test "new combination": both cells were seen in training, but never together
  test "unknown cell":    one cell (M10) never appears in training at all
Runs on adler40 (RTX 4090). usage: python train_adder_ai.py
"""
import json
import math
import pathlib
import random
import subprocess
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
TRAIN = [(5, 6), (5, 7), (5, 8), (5, 9), (6, 7), (6, 9), (7, 8), (8, 9)]
TESTS = {"new combination": [(6, 8), (7, 9)], "unknown cell M10": [(5, 10), (9, 10)]}
SAMPLES = 2000
torch.manual_seed(0)
random.seed(0)
dev = "cuda"

# tokens: 0..15 hex digit | 16..31 cell M0..MF | 32 ADD | 33 SEP
ADD, SEP, CELL = 32, 33, 16
V, T = 34, 14


def load(cells):
    rows = []
    for line in (DATA / f"adders_c{cells[0]}_{cells[1]}.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            rows.append(int(line.split()[0], 16))
    return rows


def seq(cells, prog):
    digits = [(prog >> (4 * (9 - i))) & 15 for i in range(10)]
    return [ADD, CELL + cells[0], CELL + cells[1], SEP] + digits


class GPT(nn.Module):
    def __init__(self, d=128, layers=4, heads=4):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(T, d)
        block = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(block, layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)
        self.register_buffer("mask", torch.triu(torch.full((T, T), float("-inf")), 1))

    def forward(self, x):
        n = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(n, device=x.device))
        h = self.body(h, mask=self.mask[:n, :n], is_causal=True)
        return self.head(self.norm(h))


# ---------------- data ----------------
known = {c: set(load(c)) for c in TRAIN + [c for t in TESTS.values() for c in t]}
train_rows = []
for c in TRAIN:
    for p in known[c]:
        train_rows.append(seq(c, p))
        train_rows.append(seq((c[1], c[0]), p))      # a+b = b+a: same programs with the cells swapped
X = torch.tensor(train_rows, dtype=torch.long)
print(f"training examples: {len(X):,} from {len(TRAIN)} layouts (both orders)")
train_programs = set().union(*(known[c] for c in TRAIN))

# ---------------- train ----------------
model = GPT().to(dev)
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
EPOCHS, BS = 4, 1024
steps = EPOCHS * math.ceil(len(X) / BS)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps)
t0 = time.time()
for ep in range(EPOCHS):
    perm = torch.randperm(len(X))
    tot = 0.0
    for i in range(0, len(X), BS):
        xb = X[perm[i:i + BS]].to(dev)
        logits = model(xb[:, :-1])
        loss = F.cross_entropy(logits[:, 3:].reshape(-1, V), xb[:, 4:].reshape(-1))   # only the program digits
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        tot += loss.item() * len(xb)
    print(f"epoch {ep + 1}: loss {tot / len(X):.4f}  ({time.time() - t0:.0f} s)")


# ---------------- generate + verify ----------------
@torch.no_grad()
def generate(cells, n, temperature=1.0):
    model.eval()
    x = torch.tensor([[ADD, CELL + cells[0], CELL + cells[1], SEP]] * n, device=dev)
    for _ in range(10):
        logits = model(x)[:, -1, :16] / temperature            # only hex digits are allowed here
        nxt = torch.multinomial(F.softmax(logits, -1), 1)
        x = torch.cat([x, nxt], 1)
    digits = x[:, 4:].tolist()
    return [sum(d << (4 * (9 - i)) for i, d in enumerate(ds)) for ds in digits]


def verify(progs, cells):
    """Exact check: each program run on all 65,536 input pairs (GPU, nano_search verify)."""
    data = "".join(f"5 {p:x} {cells[0]} {cells[1]}\n" for p in progs)
    out = subprocess.run([str(HERE.parent / "nano_search"), "verify", "1"], input=data, capture_output=True,
                         text=True, cwd=HERE.parent).stdout
    ok = {}
    for line in out.splitlines():
        f = line.split()
        ok[int(f[1], 16)] = f[4] == "fails=0"
    return [ok[p] for p in progs]


results = {"train_layouts": TRAIN, "samples_per_layout": SAMPLES, "tests": {}}
random_rate = {c: len(known[c]) / 2 ** 40 for c in known}
for name, layouts in [("trained layouts (sanity)", TRAIN[:2])] + list(TESTS.items()):
    for c in layouts:
        progs = generate(c, SAMPLES)
        good = verify(progs, c)
        valid = [p for p, g in zip(progs, good) if g]
        uniq = set(valid)
        res = {"valid_rate": len(valid) / SAMPLES, "unique_valid": len(uniq),
               "all_adders_for_layout": len(known[c]), "coverage": len(uniq & known[c]) / len(known[c]),
               "valid_not_in_training": len(uniq - train_programs),
               "valid_but_missing_from_exhaustive_list": len(uniq - known[c]),
               "random_guess_rate": random_rate[c],
               "examples": [f"{p:010X}" for p in list(uniq)[:8]]}
        results["tests"][f"{name}: a=M{c[0]} b=M{c[1]}"] = res
        print(f"{name:26s} a=M{c[0]:<2} b=M{c[1]:<2}: {100 * res['valid_rate']:6.2f}% of {SAMPLES} written programs are "
              f"proven adders ({res['unique_valid']} different, {100 * res['coverage']:.2f}% of all {len(known[c]):,}); "
              f"random guessing: {100 * random_rate[c]:.6f}%")
(HERE / "results.json").write_text(json.dumps(results, indent=1))
torch.save(model.state_dict(), HERE / "adder_ai.pt")
print("saved results.json, adder_ai.pt")
