"""Test the program-writing AI on a task it was never trained for: adding THREE numbers.

Experiment A (no practice):   model trained only on 2-number adders, then asked for a 3-number adder.
Experiment B (few examples):  model also trained on 3-number adders for 6 cell layouts, tested on 4 other layouts.

Every program the model writes climbs a staircase of checks:
  step 1  answers 2 + 2 + 2 = 6
  step 2  correct on 32 more triples
  step 3  correct on ALL 16,777,216 triples (proof on the GPU, nano_search3 verify)
Runs locally on one NVIDIA GPU by default. usage: python test_three.py --help
"""
import argparse
import collections
import json
import math
import pathlib
import random
import subprocess
import time

from experiment_support import parse_proof_output

HERE = pathlib.Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--gpu", type=int, default=0, help="CUDA device for training (default: 0)")
parser.add_argument("--verifier-gpu", type=int, help="CUDA device for verification (defaults to --gpu)")
parser.add_argument("--verifier", type=pathlib.Path, default=HERE.parent / "nano_search3")
parser.add_argument("--data", type=pathlib.Path, default=HERE / "data")
parser.add_argument("--output", type=pathlib.Path, default=HERE / "results_three_local.json")
parser.add_argument("--samples", type=int, default=2000)
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--check-only", action="store_true", help="Check input files and verifier path without starting training")
args = parser.parse_args()
if args.samples < 1 or args.epochs < 1 or args.gpu < 0 or (args.verifier_gpu is not None and args.verifier_gpu < 0):
    parser.error("samples/epochs must be positive and GPU indices nonnegative")
args.verifier = args.verifier.resolve()
args.data = args.data.resolve()
if not args.verifier.is_file():
    parser.error(f"Compile nano_search3.cu first; verifier not found: {args.verifier}")
required = ["adders_c5_6", "adders_c5_7", "adders_c5_8", "adders_c5_9", "adders_c6_7", "adders_c6_9", "adders_c7_8", "adders_c8_9",
            "add3_c5_6_7", "add3_c5_6_9", "add3_c5_7_8", "add3_c5_8_9", "add3_c6_7_9", "add3_c7_8_9",
            "add3_c5_6_8", "add3_c5_7_9", "add3_c6_7_8", "add3_c6_8_9"]
missing = [n for n in required if not (args.data / (n + ".txt")).is_file()]
if missing:
    parser.error("Run prepare_data.py first; missing datasets: " + ", ".join(missing))
if args.check_only:
    print("All 18 datasets and the verifier executable are present; CUDA execution has not been tested.")
    raise SystemExit(0)

import torch
import torch.nn as nn
import torch.nn.functional as F

if not torch.cuda.is_available() or args.gpu >= torch.cuda.device_count():
    parser.error("Requested training GPU is unavailable; install CUDA-enabled PyTorch and check --gpu")
verifier_gpu = args.gpu if args.verifier_gpu is None else args.verifier_gpu
if verifier_gpu >= torch.cuda.device_count():
    parser.error("Requested verifier GPU is unavailable")
torch.cuda.set_device(args.gpu)
DATA = args.data
TRAIN2 = [(5, 6), (5, 7), (5, 8), (5, 9), (6, 7), (6, 9), (7, 8), (8, 9)]
TRAIN3 = [(5, 6, 7), (5, 6, 9), (5, 7, 8), (5, 8, 9), (6, 7, 9), (7, 8, 9)]
TEST3 = [(5, 6, 8), (5, 7, 9), (6, 7, 8), (6, 8, 9)]
SAMPLES = args.samples
dev = f"cuda:{args.gpu}"
PAD, ADD, SEP, CELL = 32, 33, 34, 16                  # tokens 0..15 = hex digits, 16..31 = cells M0..MF
V, T = 35, 15
TRI = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 2, 3), (3, 2, 1), (2, 3, 1), (5, 0, 7), (255, 1, 0), (0, 255, 1),
       (1, 0, 255), (128, 64, 64), (100, 55, 1), (7, 250, 3), (9, 9, 9), (200, 0, 100), (13, 31, 77), (255, 255, 255),
       (64, 32, 16), (16, 32, 64), (6, 1, 0), (0, 6, 1), (77, 123, 45), (45, 77, 123), (2, 0, 0), (0, 2, 0), (0, 0, 2),
       (11, 2, 30), (30, 11, 2), (2, 30, 11), (90, 90, 90), (1, 1, 1)]


def load(name):
    return [int(l.split()[0], 16) for l in (DATA / name).read_text().splitlines() if l and not l.startswith("#")]


def prompt(cells):
    head = [ADD] + [CELL + c for c in cells] + [SEP]
    return [PAD] * (5 - len(head)) + head


def seq(cells, prog):
    return prompt(cells) + [(prog >> (4 * (9 - i))) & 15 for i in range(10)]


def first_out(p, cells, vals):
    """NANO, 5-byte program p, inputs vals in cells: first OUT value or None (same rules as the GPU code)."""
    M = [(p >> (8 * (4 - i))) & 255 for i in range(5)] + [0] * 11
    for c, v in zip(cells, vals):
        M[c] = v
    A = pc = 0
    for _ in range(64):
        ins = M[pc]
        pc = (pc + 1) % 5
        op, n = ins >> 4, ins & 15
        if op == 1: A = n
        elif op == 2: A = (A + n) & 255
        elif op == 3: A = (A - n) & 255
        elif op == 4: A = M[n]
        elif op == 5: M[n] = A
        elif op == 6: A = (A + M[n]) & 255
        elif op == 7: pc = n % 5
        elif op == 8:
            if A == 0: pc = n % 5
        elif op == 9: M[n] = (M[n] + 1) & 255
        elif op == 10: M[n] = (M[n] - 1) & 255
        elif op == 11:
            k = n & 7
            A = ((A << k) | (A >> (8 - k))) & 255
        elif op == 12: A ^= M[n]
        elif op == 13: return M[n]
        elif op == 14: M[n] ^= 255
        elif op == 15: return None
    return None


def proof(progs, cells):
    data = "".join(f"5 {p:x} {cells[0]} {cells[1]} {cells[2]}\n" for p in progs)
    out = subprocess.run([str(args.verifier), "verify", str(verifier_gpu)], input=data, capture_output=True,
                         text=True, cwd=args.verifier.parent, check=True).stdout
    return parse_proof_output(out, progs, cells)


class GPT(nn.Module):
    def __init__(self, d=128, layers=4, heads=4):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, d), nn.Embedding(T, d)
        block = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.body, self.norm, self.head = nn.TransformerEncoder(block, layers), nn.LayerNorm(d), nn.Linear(d, V)
        self.register_buffer("mask", torch.triu(torch.full((T, T), float("-inf")), 1))

    def forward(self, x):
        n = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(n, device=x.device))
        return self.head(self.norm(self.body(h, mask=self.mask[:n, :n], is_causal=True)))


def train(rows, seed):
    torch.manual_seed(seed)
    X = torch.tensor(rows, dtype=torch.long)
    model = GPT().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    epochs, bs = args.epochs, 1024
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * math.ceil(len(X) / bs))
    t0 = time.time()
    for ep in range(epochs):
        perm, tot = torch.randperm(len(X)), 0.0
        for i in range(0, len(X), bs):
            xb = X[perm[i:i + bs]].to(dev)
            loss = F.cross_entropy(model(xb[:, :-1])[:, 4:].reshape(-1, V), xb[:, 5:].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(xb)
        print(f"    epoch {ep + 1}: loss {tot / len(X):.4f} ({time.time() - t0:.0f} s)", flush=True)
    return model


@torch.no_grad()
def generate(model, cells, n):
    model.eval()
    x = torch.tensor([prompt(cells)] * n, device=dev)
    for _ in range(10):
        nxt = torch.multinomial(F.softmax(model(x)[:, -1, :16], -1), 1)   # only hex digits may be written
        x = torch.cat([x, nxt], 1)
    return [sum(d << (4 * (9 - i)) for i, d in enumerate(r)) for r in x[:, 5:].tolist()]


def staircase(model, cells, truth, seen):
    progs = generate(model, cells, SAMPLES)
    s1 = [p for p in progs if first_out(p, cells, (2, 2, 2)) == 6]
    s2 = [p for p in s1 if all(first_out(p, cells, t) == (sum(t) & 255) for t in TRI)]
    ok = proof(sorted(set(s2)), cells) if s2 else {}
    s3 = [p for p in s2 if ok.get(p)]
    u3 = set(s3)
    answers = collections.Counter(first_out(p, cells, (2, 2, 2)) for p in progs)
    return {"written": len(progs), "step1_2+2+2=6": len(s1), "step2_32_triples": len(s2), "step3_all_16.7M_triples": len(s3),
            "different_proven": len(u3), "proven_not_in_training": len(u3 - seen),
            "proven_but_not_in_exhaustive_list": len(u3 - truth), "all_adders_for_layout": len(truth),
            "random_guessing_rate": len(truth) / 2 ** 40,
            "most_common_answers_for_2+2+2": {str(k): v for k, v in answers.most_common(5)},
            "examples": [f"{p:010X}" for p in list(u3)[:6]]}


truth3 = {c: set(load(f"add3_c{c[0]}_{c[1]}_{c[2]}.txt")) for c in TRAIN3 + TEST3}
rows2 = [seq(o, p) for c in TRAIN2 for p in load(f"adders_c{c[0]}_{c[1]}.txt") for o in (c, c[::-1])]
seen2 = {p for c in TRAIN2 for p in load(f"adders_c{c[0]}_{c[1]}.txt")}
print({f"M{c}": len(v) for c, v in truth3.items()})
results = {}

print("Experiment A: trained on 2-number adders only", flush=True)
mA = train(rows2, 1)
results["A_no_practice"] = {f"a=M{c[0]} b=M{c[1]} c=M{c[2]}": staircase(mA, c, truth3[c], seen2) for c in TEST3}

print("Experiment B: + 3-number adders for 6 layouts (all 6 orders of the cells)", flush=True)
import itertools
rows3 = [seq(o, p) for c in TRAIN3 for p in truth3[c] for o in itertools.permutations(c)]
reps = max(1, len(rows2) // (4 * len(rows3)))         # 3-number examples make up about 20 % of training
seen3 = seen2 | {p for c in TRAIN3 for p in truth3[c]}
print(f"    {len(rows2):,} two-number + {len(rows3):,} three-number examples (x{reps})", flush=True)
mB = train(rows2 + rows3 * reps, 2)
results["B_few_examples"] = {f"a=M{c[0]} b=M{c[1]} c=M{c[2]}": staircase(mB, c, truth3[c], seen3) for c in TEST3}

for exp, res in results.items():
    print(f"\n{exp}")
    for k, r in res.items():
        print(f"  {k}: of {r['written']}: step1 {r['step1_2+2+2=6']:5d}  step2 {r['step2_32_triples']:5d}  "
              f"step3 PROVEN {r['step3_all_16.7M_triples']:5d}  ({r['different_proven']} different, "
              f"{r['proven_not_in_training']} new)   answers for 2+2+2: {r['most_common_answers_for_2+2+2']}")
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(results, indent=1))
print(f"saved {args.output}")
