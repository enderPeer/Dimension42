"""Programming by example: an AI that writes a NANO program from input/output examples.

Training data = random 5-byte NANO programs from the whole 2^40 space, generated and run live on the GPU
(never stored). Each example: 8 input pairs (a in M5, b in M6) with the program's answer (first OUT value),
followed by the program itself. The model learns: examples -> program.

Held-out test: a set of target functions (a+b, a XOR b, ...). Every random program that behaves like one of
them on 16 probe inputs is thrown out of training, so the model never sees a program for any test function.
At test time it gets 8 examples of the function and writes programs; each is proven on all 65,536 input pairs.

usage: python pbe.py [minutes]        (runs on adler40, logs to pbe.log, model to pbe.pt)
"""
import json
import math
import pathlib
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
dev = "cuda"
L, K = 5, 8                                      # program bytes, examples per prompt
NONE, SEP, BOS = 256, 257, 258                   # tokens: 0..255 byte values
V, T = 259, 1 + 3 * K + 1 + L                    # vocab, sequence length (31)
MINUTES = float(sys.argv[1]) if len(sys.argv) > 1 else 120
torch.manual_seed(0)
log_f = open(HERE / "pbe.log", "a")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log_f.write(s + "\n")
    log_f.flush()


# ---------------------------------------------------------------- NANO on the GPU, many machines at once
def run(prog, a, b):
    """prog (N,5) long, a/b (N,) long -> first OUT value (N,), -1 if none. Same rules as nano_search.cu."""
    N = prog.shape[0]
    M = torch.zeros(N, 16, dtype=torch.long, device=prog.device)
    M[:, :L] = prog
    M[:, 5], M[:, 6] = a, b
    A = torch.zeros(N, dtype=torch.long, device=prog.device)
    pc = torch.zeros_like(A)
    out = torch.full_like(A, -1)
    done = torch.zeros(N, dtype=torch.bool, device=prog.device)
    for _ in range(64):
        ins = M.gather(1, pc[:, None])[:, 0]
        op, n = ins >> 4, ins & 15
        val = M.gather(1, n[:, None])[:, 0]
        act = ~done
        is_out = act & (op == 13)
        out = torch.where(is_out, val, out)
        done = done | is_out | (act & (op == 15))
        live = act & ~is_out & (op != 15)
        k = n & 7
        rol = ((A << k) | (A >> (8 - k))) & 255
        nA = A
        nA = torch.where(op == 1, n, nA)
        nA = torch.where(op == 2, (A + n) & 255, nA)
        nA = torch.where(op == 3, (A - n) & 255, nA)
        nA = torch.where(op == 4, val, nA)
        nA = torch.where(op == 6, (A + val) & 255, nA)
        nA = torch.where(op == 11, rol, nA)
        nA = torch.where(op == 12, A ^ val, nA)
        wr = torch.where(op == 5, A, torch.where(op == 9, val + 1, torch.where(op == 10, val - 1, 255 - val))) & 255
        dow = live & ((op == 5) | (op == 9) | (op == 10) | (op == 14))
        M.scatter_(1, n[:, None], torch.where(dow, wr, val)[:, None])
        jump = live & ((op == 7) | ((op == 8) & (A == 0)))
        pc = torch.where(jump, n % L, (pc + 1) % L)
        A = torch.where(live, nA, A)
        if bool(done.all()):
            break
    return out


def run_many(prog, a, b):
    """prog (P,5), a/b (P,Q) -> outputs (P,Q)"""
    P, Q = a.shape
    return run(prog.repeat_interleave(Q, 0), a.reshape(-1), b.reshape(-1)).reshape(P, Q)


# ---------------------------------------------------------------- test functions (held out from training)
TESTS = {
    "a + b": lambda a, b: (a + b) & 255,
    "a XOR b": lambda a, b: a ^ b,
    "2a + b": lambda a, b: (2 * a + b) & 255,
    "a + b + 1": lambda a, b: (a + b + 1) & 255,
    "3b": lambda a, b: (3 * b) & 255,
    "a + 5": lambda a, b: (a + 5) & 255,
    "NOT b": lambda a, b: 255 - b,
    "a - 1": lambda a, b: (a - 1) & 255,
    "rotate a left 3": lambda a, b: ((a << 3) | (a >> 5)) & 255,
    "a XOR b XOR 7": lambda a, b: a ^ b ^ 7,
}
g = torch.Generator(device=dev).manual_seed(123)
PA = torch.randint(0, 256, (16,), device=dev, generator=g)          # 16 probe inputs define a "behavior"
PB = torch.randint(0, 256, (16,), device=dev, generator=g)
W = torch.randint(1, 2 ** 61, (16,), device=dev, generator=g)
held_sigs = {int(((f(PA, PB) + 1) * W).sum()) for f in TESTS.values()}


def signature(outs):
    return ((outs + 1) * W).sum(1)


# ---------------------------------------------------------------- live data generator
seen = {}                                          # behavior -> how often it was used (cap per behavior)
CAP = 4
stats = {"random": 0, "kept": 0, "held_out_hits": 0}


def fresh(n):
    """n training sequences from random programs: no junk, no held-out behavior, each behavior <= CAP times."""
    rows = []
    while sum(len(r) for r in rows) < n:
        prog = torch.randint(0, 256, (1 << 18, L), device=dev)
        outs = run_many(prog, PA.expand(len(prog), 16), PB.expand(len(prog), 16))
        stats["random"] += len(prog)
        ok = (outs >= 0).all(1) & (outs != outs[:, :1]).any(1)        # always answers, and depends on the input
        prog, outs = prog[ok], outs[ok]
        sig = signature(outs).tolist()
        keep = []
        for i, s in enumerate(sig):
            if s in held_sigs:
                stats["held_out_hits"] += 1
                continue
            c = seen.get(s, 0)
            if c < CAP:
                seen[s] = c + 1
                keep.append(i)
        if not keep:
            continue
        prog = prog[torch.tensor(keep, device=dev)]
        a = torch.randint(0, 256, (len(prog), K), device=dev)
        b = torch.randint(0, 256, (len(prog), K), device=dev)
        o = run_many(prog, a, b)
        o = torch.where(o < 0, torch.full_like(o, NONE), o)
        ex = torch.stack([a, b, o], 2).reshape(len(prog), 3 * K)
        rows.append(torch.cat([torch.full((len(prog), 1), BOS, device=dev), ex,
                               torch.full((len(prog), 1), SEP, device=dev), prog], 1))
        stats["kept"] += len(prog)
    return torch.cat(rows)[:n]


# ---------------------------------------------------------------- model
class GPT(nn.Module):
    def __init__(self, d=384, layers=8, heads=8):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, d), nn.Embedding(T, d)
        block = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.body, self.norm, self.head = nn.TransformerEncoder(block, layers), nn.LayerNorm(d), nn.Linear(d, V)
        self.register_buffer("mask", torch.triu(torch.full((T, T), float("-inf")), 1))

    def forward(self, x):
        n = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(n, device=x.device))
        return self.head(self.norm(self.body(h, mask=self.mask[:n, :n], is_causal=True)))


@torch.no_grad()
def write_programs(model, a, b, o, n):
    """n programs for one set of K examples."""
    model.eval()
    ex = torch.stack([a, b, o], 1).reshape(1, 3 * K)
    x = torch.cat([torch.tensor([[BOS]], device=dev), ex, torch.tensor([[SEP]], device=dev)], 1).repeat(n, 1)
    for _ in range(L):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)[:, -1, :256].float()
        x = torch.cat([x, torch.multinomial(F.softmax(logits, -1), 1)], 1)
    model.train()
    return x[:, -L:]


ALL_A = torch.arange(256, device=dev).repeat_interleave(256)
ALL_B = torch.arange(256, device=dev).repeat(256)


def proven(prog, f):
    """exact: program must answer f(a, b) for all 65,536 pairs"""
    want = f(ALL_A, ALL_B)
    return [bool((run(p.expand(65536, L), ALL_A, ALL_B) == want).all()) for p in prog]


def evaluate(model, prompts=5, per_prompt=40):
    res = {}
    for name, f in TESTS.items():
        good, uniq, solved_prompts = 0, set(), 0
        for t in range(prompts):
            a = torch.randint(0, 256, (K,), device=dev)
            b = torch.randint(0, 256, (K,), device=dev)
            progs = torch.unique(write_programs(model, a, b, f(a, b), per_prompt), dim=0)
            # cheap filter first (the 8 shown examples), then the full proof
            fit = run_many(progs, a.expand(len(progs), K), b.expand(len(progs), K)) == f(a, b)
            cand = progs[fit.all(1)]
            ok = proven(cand, f) if len(cand) else []
            hits = [bytes(p.tolist()).hex().upper() for p, k in zip(cand, ok) if k]
            good += len(hits)
            uniq |= set(hits)
            solved_prompts += bool(hits)
        res[name] = {"prompts_solved": f"{solved_prompts}/{prompts}", "distinct_proven_programs": len(uniq),
                     "examples": sorted(uniq)[:4]}
    return res


if __name__ == "__main__":
    # self-test: the GPU interpreter must agree with the plain Python reference
    sys.path.insert(0, str(HERE.parent))
    import nano_ref
    tp = torch.randint(0, 256, (20000, L), device=dev)
    ta, tb = torch.randint(0, 256, (20000,), device=dev), torch.randint(0, 256, (20000,), device=dev)
    got = run(tp, ta, tb).tolist()
    bad = 0
    for i in range(20000):
        p = int.from_bytes(bytes(tp[i].tolist()), "big")
        r = nano_ref.first_out(p, 5, int(ta[i]), int(tb[i]))
        bad += (r if r is not None else -1) != got[i]
    log(f"self-test: GPU interpreter vs Python reference on 20,000 random programs: {bad} mismatches")
    assert bad == 0

    model = GPT().to(dev)
    log(f"model parameters: {sum(p.numel() for p in model.parameters()):,}; training for {MINUTES:.0f} minutes")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95))
    BS, t0, step, next_eval = 1024, time.time(), 0, 0
    history = []
    while True:
        el = (time.time() - t0) / 60
        lr = 3e-4 * min(1.0, step / 2000) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, el / MINUTES))))
        for gp in opt.param_groups:
            gp["lr"] = lr
        x = fresh(BS)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x[:, :-1])
        loss = F.cross_entropy(logits[:, -L:].float().reshape(-1, V), x[:, -L:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 500 == 0:
            log(f"step {step}  {el:5.1f} min  loss {loss.item():.3f}  examples {stats['kept']:,} from "
                f"{stats['random']:,} random programs, {len(seen):,} different behaviors, "
                f"{stats['held_out_hits']:,} held-out programs thrown away")
        if el >= next_eval or el >= MINUTES:
            res = evaluate(model)
            history.append({"minutes": round(el, 1), "step": step, "examples": stats["kept"], "tests": res})
            log(f"--- test at {el:.1f} min: " + "  ".join(f"{k}: {v['prompts_solved']}" for k, v in res.items()))
            (HERE / "pbe_results.json").write_text(json.dumps({"stats": stats, "history": history}, indent=1))
            torch.save(model.state_dict(), HERE / "pbe.pt")
            next_eval += 15
            if el >= MINUTES:
                break
    log("done")
