"""Find every NANO program of 1..5 bytes that adds THREE numbers, and prove it on all 16,777,216 triples.

usage: python search3_run.py --cells A B C        (a in M[A], b in M[B], c in M[C])
Writes results/add3_cA_B_C/summary.json and adders_L{n}.txt.
"""
import json
import pathlib
import queue
import re
import subprocess
import sys
import threading
import time

CELLS = tuple(int(x) for x in sys.argv[sys.argv.index("--cells") + 1:][:3])
GPUS = [
    ("adler40",  "RTX 4090",    "./nano_search3 serve 0"),
    ("adler40",  "RTX 4080",    "./nano_search3 serve 1"),
    ("knecht24", "RTX 3060 #0", "./nano_search3 serve 0"),
    ("knecht24", "RTX 3060 #1", "./nano_search3 serve 1"),
    ("knecht24", "RTX 3060 #2", "./nano_search3 serve 2"),
    ("specht32", "RX 9070 XT",  "./nano_search3_vk serve 1"),
    ("specht32", "RX 9060 XT",  "./nano_search3_vk serve 0"),
    ("falke64",  "R9700 #0",    "./nano_search3_vk serve 0"),
    ("falke64",  "R9700 #1",    "./nano_search3_vk serve 1"),
]
VERIFIERS = [("adler40", 0), ("adler40", 1), ("knecht24", 0), ("knecht24", 1), ("knecht24", 2)]
CH = 1 << 30
CATS = ["says", "add", "const", "other", "overflow"]
jobs = [(5, k * CH, CH) for k in range(1024)] + [(4, k * CH, CH) for k in range(4)] + \
       [(3, 0, 1 << 24), (2, 0, 1 << 16), (1, 0, 256)]
todo = queue.Queue()
for j in jobs:
    todo.put(j)
lock = threading.Lock()
done = []
t0 = time.perf_counter()


def worker(node, label, cmd):
    proc = subprocess.Popen(["ssh", "-o", "BatchMode=yes", node, f"cd ~/dimension42-explorer && exec {cmd}"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    while True:
        try:
            L, s, n = todo.get_nowait()
        except queue.Empty:
            break
        proc.stdin.write(f"{L} {s} {n} {CELLS[0]} {CELLS[1]} {CELLS[2]}\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        m = re.search(r"done L=(\d+) start=(\d+) count=(\d+) says=(\d+) add=(\d+) const=(\d+) other=(\d+) overflow=(\d+) "
                      r"secs=([\d.]+) device=.*? cands=(.*)$", line)
        if not m or int(m.group(2)) != s:
            print(f"!! {label} failed on L={L} start={s}: {line[:150]!r} - job put back, device retired")
            todo.put((L, s, n))
            break
        with lock:
            done.append((L, {c: int(m.group(4 + i)) for i, c in enumerate(CATS)},
                         {int(x, 16) for x in m.group(10).strip().split(",") if x}))
    proc.stdin.close()
    proc.wait()


threads = [threading.Thread(target=worker, args=g) for g in GPUS]
[t.start() for t in threads]
[t.join() for t in threads]
t_search = time.perf_counter() - t0
assert len(done) == len(jobs), f"only {len(done)} of {len(jobs)} jobs done"
perL = {L: {c: 0 for c in CATS} for L in range(1, 6)}
cands = {L: set() for L in range(1, 6)}
for L, c, cs in done:
    for k in CATS:
        perL[L][k] += c[k]
    cands[L] |= cs
assert all(perL[L]["overflow"] == 0 and perL[L]["add"] == len(cands[L]) for L in perL)

allc = [(L, p) for L in range(1, 6) for p in sorted(cands[L])]
t1 = time.perf_counter()
verified = {}


def verify(v, part):
    node, dev = v
    if not part:
        return
    data = "".join(f"{L} {p:x} {CELLS[0]} {CELLS[1]} {CELLS[2]}\n" for L, p in part)
    out = subprocess.run(["ssh", "-o", "BatchMode=yes", node, f"cd ~/dimension42-explorer && ./nano_search3 verify {dev}"],
                         input=data, capture_output=True, text=True).stdout
    for line in out.splitlines():
        m = re.match(r"(\d+) ([0-9a-f]+) \d+ \d+ \d+ fails=(\d+) outputs=(\d+) halts=(\d+) steps=(\d+)", line)
        if m:
            with lock:
                verified[(int(m.group(1)), int(m.group(2), 16))] = [int(m.group(i)) for i in range(3, 7)]


vt = [threading.Thread(target=verify, args=(v, allc[i::len(VERIFIERS)])) for i, v in enumerate(VERIFIERS)]
[t.start() for t in vt]
[t.join() for t in vt]
t_verify = time.perf_counter() - t1
assert len(verified) == len(allc), f"verified {len(verified)} of {len(allc)}"

out = pathlib.Path(__file__).with_name("results") / f"add3_c{CELLS[0]}_{CELLS[1]}_{CELLS[2]}"
out.mkdir(parents=True, exist_ok=True)
summary = {"cells": CELLS, "seconds": {"search": t_search, "verify": t_verify}, "per_length": {}}
line = f"a=M{CELLS[0]} b=M{CELLS[1]} c=M{CELLS[2]}: search {t_search:.0f} s, proof {t_verify:.0f} s |"
for L in range(1, 6):
    real = sorted(p for (l, p), r in verified.items() if l == L and r[0] == 0)
    failed = [p for (l, p), r in verified.items() if l == L and r[0] > 0]
    summary["per_length"][L] = {**perL[L], "add_proven": len(real), "add_failed_full_test": len(failed)}
    with open(out / f"adders_L{L}.txt", "w") as f:
        f.write("# program outputs_at_2+2+2 halts steps   (proven on all 16,777,216 triples)\n")
        for p in real:
            r = verified[(L, p)]
            f.write(f"{p:0{2 * L}X} {r[1]} {r[2]} {r[3]}\n")
    if perL[L]["says"]:
        line += f" {L}B: {perL[L]['says']:,} say 6, {len(real):,} proven" + (f" ({len(failed)} failed proof)" if failed else "")
(out / "summary.json").write_text(json.dumps(summary, indent=1))
print(line, flush=True)
