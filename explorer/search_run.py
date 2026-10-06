"""Find every NANO program of 1..5 bytes that answers 2 + 2 = 4, and prove which ones really add.

Pass 1: all 9 GPUs search all programs (dynamic chunk queue) -> counts per category + ADD candidates.
Pass 2: every ADD candidate is checked on all 65,536 input pairs on the NVIDIA GPUs.
Writes results/add2/summary.json and results/add2/adders_L{n}.txt.
"""
import json
import pathlib
import queue
import re
import subprocess
import threading
import time

GPUS = [
    ("adler40",  "RTX 4090",    "./nano_search serve 0"),
    ("adler40",  "RTX 4080",    "./nano_search serve 1"),
    ("knecht24", "RTX 3060 #0", "./nano_search serve 0"),
    ("knecht24", "RTX 3060 #1", "./nano_search serve 1"),
    ("knecht24", "RTX 3060 #2", "./nano_search serve 2"),
    ("specht32", "RX 9070 XT",  "./nano_search_vk serve 1"),
    ("specht32", "RX 9060 XT",  "./nano_search_vk serve 0"),
    ("falke64",  "R9700 #0",    "./nano_search_vk serve 0"),
    ("falke64",  "R9700 #1",    "./nano_search_vk serve 1"),
]
VERIFIERS = [("adler40", "RTX 4090", 0), ("adler40", "RTX 4080", 1),
             ("knecht24", "RTX 3060 #0", 0), ("knecht24", "RTX 3060 #1", 1), ("knecht24", "RTX 3060 #2", 2)]
import sys
CELLS = tuple(int(x) for x in sys.argv[sys.argv.index("--cells") + 1:][:2]) if "--cells" in sys.argv else (5, 6)
CH = 1 << 30
CATS = ["says4", "add", "const", "mul", "twoa", "twob", "other", "overflow"]

jobs = [(5, k * CH, CH) for k in range(1024)] + [(4, k * CH, CH) for k in range(4)] + \
       [(3, 0, 1 << 24), (2, 0, 1 << 16), (1, 0, 256)]
todo = queue.Queue()
for j in jobs:
    todo.put(j)
lock = threading.Lock()
done = []                                   # (L, start, count, counts dict, cands set, device, secs)
t0 = time.perf_counter()


def worker(node, label, cmd):
    proc = subprocess.Popen(["ssh", "-o", "BatchMode=yes", node, f"cd ~/dimension42-explorer && exec {cmd}"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    while True:
        try:
            L, s, n = todo.get_nowait()
        except queue.Empty:
            break
        proc.stdin.write(f"{L} {s} {n} {CELLS[0]} {CELLS[1]}\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        m = re.search(r"done L=(\d+) start=(\d+) count=(\d+) " + " ".join(f"{c}=(\\d+)" for c in CATS)
                      + r" secs=([\d.]+) device=.*? cands=(.*)$", line)
        if not m or int(m.group(2)) != s:
            print(f"!! {label} failed on L={L} start={s}: {line[:150]!r} - job put back, device retired")
            todo.put((L, s, n))
            break
        counts = {c: int(m.group(4 + i)) for i, c in enumerate(CATS)}
        cands = {int(x, 16) for x in m.group(13).strip().split(",") if x}
        with lock:
            done.append((L, s, n, counts, cands, label, float(m.group(12))))
            if len(done) % 128 == 0 or len(done) == len(jobs):
                el = time.perf_counter() - t0
                progs = sum(d[2] for d in done)
                print(f"  {len(done):4d}/{len(jobs)} jobs  {el:6.1f} s  {progs / el / 1e9:5.2f} G programs/s", flush=True)
    proc.stdin.close()
    proc.wait()


print(f"inputs a=M{CELLS[0]}, b=M{CELLS[1]} - pass 1: {sum(j[2] for j in jobs):,} programs (all of 1..5 bytes) on {len(GPUS)} GPUs")
threads = [threading.Thread(target=worker, args=g) for g in GPUS]
[t.start() for t in threads]
[t.join() for t in threads]
t_search = time.perf_counter() - t0
assert len(done) == len(jobs), f"only {len(done)} of {len(jobs)} jobs done"

perL = {L: {c: 0 for c in CATS} for L in range(1, 6)}
cands = {L: set() for L in range(1, 6)}
for L, s, n, c, cs, dev, secs in done:
    for k in CATS:
        perL[L][k] += c[k]
    cands[L] |= cs
assert all(perL[L]["overflow"] == 0 for L in perL), "candidate buffer overflowed"
assert all(perL[L]["add"] == len(cands[L]) for L in perL), "candidate count mismatch"
print(f"pass 1 done in {t_search:.1f} s")

# ---- pass 2: prove every candidate on all 65,536 input pairs ----
allc = [(L, p) for L in range(1, 6) for p in sorted(cands[L])]
print(f"pass 2: {len(allc):,} ADD candidates x 65,536 input pairs on {len(VERIFIERS)} NVIDIA GPUs")
t1 = time.perf_counter()
parts = [allc[i::len(VERIFIERS)] for i in range(len(VERIFIERS))]
verified = {}


def verify(v, part):
    node, label, dev = v
    if not part:
        return
    data = "".join(f"{L} {p:x} {CELLS[0]} {CELLS[1]}\n" for L, p in part)
    out = subprocess.run(["ssh", "-o", "BatchMode=yes", node, f"cd ~/dimension42-explorer && ./nano_search verify {dev}"],
                         input=data, capture_output=True, text=True).stdout
    for line in out.splitlines():
        m = re.match(r"(\d+) ([0-9a-f]+) \d+ \d+ fails=(\d+) first_fail=(\d+) outputs=(\d+) halts=(\d+) steps=(\d+)", line)
        if m:
            with lock:
                verified[(int(m.group(1)), int(m.group(2), 16))] = {
                    "fails": int(m.group(3)), "first_fail": int(m.group(4)), "outputs": int(m.group(5)),
                    "halts": int(m.group(6)), "steps": int(m.group(7))}


vt = [threading.Thread(target=verify, args=(v, part)) for v, part in zip(VERIFIERS, parts)]
[t.start() for t in vt]
[t.join() for t in vt]
t_verify = time.perf_counter() - t1
assert len(verified) == len(allc), f"verified {len(verified)} of {len(allc)}"

out = pathlib.Path(__file__).with_name("results") / ("add2" if CELLS == (5, 6) else f"add2_c{CELLS[0]}_{CELLS[1]}")
out.mkdir(parents=True, exist_ok=True)
summary = {"cells": CELLS, "definition": {"input": f"a in M{CELLS[0]}, b in M{CELLS[1]}", "answer": "first value output with OUT",
                          "search": "all programs of 1..5 bytes run with a=2, b=2; answer 4 -> tested on 32 more pairs",
                          "proof": "ADD candidates run on all 65,536 pairs a,b in 0..255, must answer (a+b) mod 256"},
           "seconds": {"search": t_search, "verify": t_verify}, "per_length": {}}
for L in range(1, 6):
    real = sorted(p for (l, p), r in verified.items() if l == L and r["fails"] == 0)
    clean = [p for p in real if verified[(L, p)]["outputs"] == 1 and verified[(L, p)]["halts"]]
    failed = sorted(p for (l, p), r in verified.items() if l == L and r["fails"] > 0)
    summary["per_length"][L] = {"programs": 256 ** L, **perL[L], "add_proven": len(real),
                                "add_failed_full_test": len(failed), "add_clean": len(clean),
                                "failed_examples": [f"{p:0{2 * L}x}" for p in failed[:20]]}
    with open(out / f"adders_L{L}.txt", "w") as f:
        f.write("# program  outputs_at_2+2  halts  steps   (all proven on 65,536 input pairs)\n")
        for p in real:
            r = verified[(L, p)]
            f.write(f"{p:0{2 * L}X} {r['outputs']} {r['halts']} {r['steps']}\n")
(out / "summary.json").write_text(json.dumps(summary, indent=1))

print(f"pass 2 done in {t_verify:.1f} s\n")
print(f"{'bytes':>5} {'programs':>18} {'say 4':>14} {'real adders':>12} {'clean':>8} {'always 4':>13} {'2a':>9} {'2b':>9} {'a*b':>6} {'coincidence':>12}")
for L in range(1, 6):
    s = summary["per_length"][L]
    print(f"{L:>5} {s['programs']:>18,} {s['says4']:>14,} {s['add_proven']:>12,} {s['add_clean']:>8,} {s['const']:>13,} "
          f"{s['twoa']:>9,} {s['twob']:>9,} {s['mul']:>6,} {s['other']:>12,}")
    if s["add_failed_full_test"]:
        print(f"      {s['add_failed_full_test']} passed 32 pairs but failed the full 65,536-pair test")
print(f"\nresults: {out}")
