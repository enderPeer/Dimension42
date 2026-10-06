"""Compute every 5-byte NANO program (2^40) on all 9 GPUs of the cluster.

The space is cut into 1024 chunks of 2^30 programs. One worker per GPU runs in "serve"
mode; whenever a GPU finishes a chunk it gets the next one (dynamic load balancing).
Each chunk's classes (1 byte per program) are zstd-compressed on the node by the CPUs
while the GPU computes the next chunk, and stored as ~/dimension42-explorer/L5/chunk_NNNN.zst.

usage: python cluster5.py            writes results/L5/manifest.json
"""
import json
import pathlib
import queue
import re
import subprocess
import threading
import time

import sys
L = 5
TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""   # e.g. run2: separate output dir
DIR = f"L{L}{TAG}"
CHUNK = 1 << 30
NCHUNKS = (256 ** L) // CHUNK          # 1024

# node, label, worker command (serve mode: L, gpu, zstd threads)
GPUS = [
    ("adler40",  "RTX 4090",    f"./nano_cuda serve {L} 0 10"),
    ("adler40",  "RTX 4080",    f"./nano_cuda serve {L} 1 8"),
    ("knecht24", "RTX 3060 #0", f"./nano_cuda serve {L} 0 6"),
    ("knecht24", "RTX 3060 #1", f"./nano_cuda serve {L} 1 6"),
    ("knecht24", "RTX 3060 #2", f"./nano_cuda serve {L} 2 6"),
    ("specht32", "RX 9070 XT",  f"./nano_vk serve {L} 1 6"),
    ("specht32", "RX 9060 XT",  f"./nano_vk serve {L} 0 5"),
    ("falke64",  "R9700 #0",    f"./nano_vk serve {L} 0 6"),
    ("falke64",  "R9700 #1",    f"./nano_vk serve {L} 1 6"),
]

todo = queue.Queue()
for k in range(NCHUNKS):
    todo.put(k)
results, lock = {}, threading.Lock()
t_start = time.perf_counter()


def worker(node, label, cmd):
    proc = subprocess.Popen(
        ["ssh", "-o", "BatchMode=yes", node, f"cd ~/dimension42-explorer && mkdir -p {DIR} && exec {cmd}"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    while True:
        try:
            k = todo.get_nowait()
        except queue.Empty:
            break
        proc.stdin.write(f"{k * CHUNK} {CHUNK} {DIR}/chunk_{k:04d}.zst\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        m = re.search(r"done start=(\d+) programs=(\d+) hashsum=(\d+) counts=([\d,]+) secs=([\d.]+)", line)
        if not m or int(m.group(1)) != k * CHUNK:
            print(f"!! {node} {label} failed on chunk {k}: {line.strip()!r} - chunk put back, device retired")
            todo.put(k)
            break
        with lock:
            results[k] = {"node": node, "device": label, "hashsum": int(m.group(3)),
                          "counts": [int(c) for c in m.group(4).split(",")], "secs": float(m.group(5)),
                          "file": f"chunk_{k:04d}.zst"}
            done = len(results)
            if done % 64 == 0 or done == NCHUNKS:
                el = time.perf_counter() - t_start
                print(f"  {done:4d}/{NCHUNKS} chunks  {el:6.1f} s  {done * CHUNK / el / 1e9:5.2f} G programs/s", flush=True)
    proc.stdin.close()
    proc.wait()


print(f"L={L}: {256 ** L:,} programs, {NCHUNKS} chunks of {CHUNK:,}, {len(GPUS)} GPUs")
threads = [threading.Thread(target=worker, args=g) for g in GPUS]
for t in threads:
    t.start()
for t in threads:
    t.join()
elapsed = time.perf_counter() - t_start

assert len(results) == NCHUNKS, f"only {len(results)} of {NCHUNKS} chunks done"
counts = [sum(r["counts"][i] for r in results.values()) for i in range(6)]
hashsum = sum(r["hashsum"] for r in results.values()) % 2 ** 64
names = ["IDLE", "HALT", "ACTIVE", "SELF-MOD", "OUTPUT", "DRAW"]
print(f"\nall programs computed : {sum(counts):,} of {256 ** L:,}")
print("classes               : " + ", ".join(f"{n} {c:,}" for n, c in zip(names, counts)))
print(f"hash of all states    : {hashsum}")
print(f"wall time             : {elapsed:.1f} s  ({256 ** L / elapsed / 1e9:.2f} G programs/s incl. compression)")
per = {}
for r in results.values():
    per.setdefault(r["device"], []).append(r["secs"])
for g in GPUS:
    s = per.get(g[1], [])
    print(f"  {g[1]:12s} {len(s):4d} chunks  {sum(s):6.1f} s busy")

out = pathlib.Path(__file__).with_name("results") / DIR
out.mkdir(parents=True, exist_ok=True)
manifest = {"L": L, "programs": 256 ** L, "chunk_programs": CHUNK, "chunks": NCHUNKS,
            "format": "chunk_NNNN.zst = zstd stream of 2^30 bytes; byte i = class of program NNNN*2^30+i",
            "classes": dict(zip(names, counts)), "class_codes": {n: i + 1 for i, n in enumerate(names)},
            "hashsum": hashsum, "wall_seconds": elapsed,
            "chunk": {str(k): results[k] for k in range(NCHUNKS)}}
(out / "manifest.json").write_text(json.dumps(manifest, indent=1))
print(f"manifest              : {out / 'manifest.json'}")
