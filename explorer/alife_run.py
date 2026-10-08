"""Artificial-life toolkit: classify all 2^40 five-byte programs (nano_alife.cu), then re-check every stored
program with an independent Python implementation. Writes results/alife/*.txt and toolkit.json."""
import collections
import json
import pathlib
import queue
import re
import subprocess
import threading
import time

GPUS = [("adler40", 0), ("adler40", 1), ("knecht24", 0), ("knecht24", 1), ("knecht24", 2)]
CH, NCH, L = 1 << 30, 1024, 5
NAMES = ["rep", "mutrep", "healer", "walker", "stable", "repair", "repair2"]
todo = queue.Queue()
for k in range(NCH):
    todo.put(k)
lock = threading.Lock()
totals, stored, done = collections.Counter(), {}, 0
t0 = time.perf_counter()


def worker(node, gpu):
    global done
    proc = subprocess.Popen(["ssh", "-o", "BatchMode=yes", node, f"cd ~/dimension42-explorer && exec ./nano_alife serve {gpu}"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    while True:
        try:
            k = todo.get_nowait()
        except queue.Empty:
            break
        proc.stdin.write(f"{k * CH} {CH}\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        m = re.search(r"done start=(\d+) " + " ".join(f"{n}=(\\d+)" for n in NAMES) + r" overflow=(\d+) .*cands=(.*)$", line)
        if not m or int(m.group(1)) != k * CH or m.group(9) != "0":
            print(f"!! {node} gpu{gpu} failed on chunk {k}: {line[:120]!r}", flush=True)
            todo.put(k)
            break
        with lock:
            for i, n in enumerate(NAMES):
                totals[n] += int(m.group(2 + i))
            for e in m.group(10).strip().split(","):
                if e:
                    p, f = e.split("-")
                    stored[int(p, 16)] = int(f, 16)
            done += 1
            if done % 128 == 0:
                print(f"  {done}/{NCH} chunks, {time.perf_counter() - t0:.0f} s", flush=True)
    proc.stdin.close()
    proc.wait()


threads = [threading.Thread(target=worker, args=g) for g in GPUS]
[t.start() for t in threads]
[t.join() for t in threads]
assert done == NCH
print(f"all 1,099,511,627,776 five-byte programs in {time.perf_counter() - t0:.0f} s")


# ---------------- independent re-check (plain Python, same definitions)
def run(M, p0, full):
    M = list(M)
    A = pc = 0
    wrote, edits, halted, work, rep, changed = 0, 0, False, False, False, False
    for _ in range(64):
        ins = M[pc]
        pc = (pc + 1) % L
        op, n = ins >> 4, ins & 15
        if op == 15:
            halted = True
            break
        v, oA = M[n], A
        if op == 1: A = n
        elif op == 2: A = (A + n) & 255
        elif op == 3: A = (A - n) & 255
        elif op == 4: A = v
        elif op == 6: A = (A + v) & 255
        elif op == 11: A = ((A << (n & 7)) | (A >> (8 - (n & 7)))) & 255
        elif op == 12: A ^= v
        if op in (5, 9, 10, 14):
            w = {5: oA, 9: (v + 1) & 255, 10: (v - 1) & 255, 14: 255 - v}[op]
            if n < L:
                wrote |= 1 << n
                changed |= w != v
                edits += (w >> 4) == (v >> 4) and w != v
            else:
                work = True
            M[n] = w
            if full and not rep and n >= L:
                rep = any(M[k:k + L] == p0 for k in range(L, 12))
        if op == 7 or (op == 8 and oA == 0):
            pc = n % L
        if A:
            work = True
    intact = M[:L] == p0
    if not full:
        return intact, 0
    f = 0
    if rep:
        f |= 1
    elif any(sum(a != b for a, b in zip(M[k:k + L], p0)) == 1 and
             next(M[k + i] for i in range(L) if M[k + i] != p0[i]) != 0 for k in range(L, 12)):
        f |= 2
    if changed and intact: f |= 4
    if edits >= 4: f |= 8
    if not halted and work and intact: f |= 16
    return f, wrote


def flags_of(p):
    p0 = [(p >> (8 * (L - 1 - i))) & 255 for i in range(L)]
    f, wrote = run(p0 + [0] * 11, p0, True)
    if sum(b != 0 for b in p0) < 2:
        f &= ~3
    rep = 0
    for j in range(L):
        if wrote >> j & 1:
            ok = True
            for d in (p0[j] ^ 1, p0[j] ^ 0x80, p0[j] ^ 0xFF):
                M = p0 + [0] * 11
                M[j] = d
                ok = ok and run(M, p0, False)[0]
            rep |= ok << j
    return f | rep << 8


bad = [p for p, f in stored.items() if flags_of(p) != f]
print(f"independent re-check of all {len(stored):,} stored programs: {len(stored) - len(bad):,} identical"
      + (f", MISMATCH {[hex(b) for b in bad[:5]]}" if bad else ""))

out = pathlib.Path(__file__).with_name("results") / "alife"
out.mkdir(parents=True, exist_ok=True)
OPS = "NOP LDI ADDI SUBI LD ST ADD JMP JZ INC DEC ROL XOR OUT NOT HLT".split()
dis = lambda p: " · ".join(OPS[b >> 4] + ("" if b >> 4 in (0, 15) else f" {b & 15:X}") for b in [(p >> (8 * (4 - i))) & 255 for i in range(5)])
cats = {"replicators": lambda f: f & 1, "mutating_replicators": lambda f: f & 2, "repairers_multi": lambda f: bin(f >> 8).count("1") >= 2,
        "healers_sample": lambda f: f & 4, "walkers_sample": lambda f: f & 8, "stable_sample": lambda f: f & 16,
        "repairers_sample": lambda f: f >> 8}
toolkit = {"counts_in_all_5_byte_programs": dict(totals), "lists": {}}
for name, test in cats.items():
    rows = sorted(p for p, f in stored.items() if test(f))
    toolkit["lists"][name] = len(rows)
    with open(out / f"{name}.txt", "w") as fh:
        fh.write("# program flags(hex: 1 rep 2 mutrep 4 healer 8 walker 10 stable, repair mask << 8)  instructions\n")
        for p in rows:
            fh.write(f"{p:010X} {stored[p]:04x}  {dis(p)}\n")
(out / "toolkit.json").write_text(json.dumps(toolkit, indent=1))
print("counts over all five-byte programs:", dict(totals))
print("stored lists:", toolkit["lists"])
