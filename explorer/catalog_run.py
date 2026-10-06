"""Distill all 1.1 trillion 5-byte NANO programs into a catalog of distinct behaviors.

Behavior = the 16 answers on fixed probe inputs (a in M5, b in M6). Only useful behaviors are kept:
the program answers every probe and the answer depends on the input. For each behavior: how many
programs have it, and up to 16 example programs (from different parts of the space).
Writes results/catalog/catalog.txt  ("hash count example,example,...", most common first).
"""
import pathlib
import queue
import re
import subprocess
import threading
import time

GPUS = [("adler40", 0), ("adler40", 1), ("knecht24", 0), ("knecht24", 1), ("knecht24", 2)]
CH, NCH = 1 << 30, 1024
todo = queue.Queue()
for k in range(NCH):
    todo.put(k)
lock = threading.Lock()
cat = {}                                          # hash -> [count, [examples]]
stats = {"chunks": 0, "useful": 0, "overflow": 0}
t0 = time.perf_counter()


def worker(node, gpu):
    proc = subprocess.Popen(["ssh", "-o", "BatchMode=yes", node, f"cd ~/dimension42-explorer && exec ./nano_catalog serve {gpu}"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    while True:
        try:
            k = todo.get_nowait()
        except queue.Empty:
            break
        proc.stdin.write(f"{k * CH} {CH}\n")
        proc.stdin.flush()
        head = proc.stdout.readline()
        m = re.match(r"done start=(\d+) programs=\d+ useful=(\d+) entries=(\d+) overflow=(\d+)", head)
        if not m or int(m.group(1)) != k * CH:
            print(f"!! {node} gpu{gpu} failed on chunk {k}: {head[:120]!r}")
            todo.put(k)
            break
        rows = [proc.stdout.readline().split() for _ in range(int(m.group(3)))]
        with lock:
            for h, c, e in rows:
                entry = cat.setdefault(int(h, 16), [0, []])
                entry[0] += int(c)
                if len(entry[1]) < 16:
                    entry[1].append(e)
            stats["chunks"] += 1
            stats["useful"] += int(m.group(2))
            stats["overflow"] += int(m.group(4))
            if stats["chunks"] % 128 == 0:
                print(f"  {stats['chunks']:4d}/{NCH} chunks  {time.perf_counter() - t0:6.1f} s  "
                      f"{len(cat):,} different behaviors so far", flush=True)
    proc.stdin.close()
    proc.wait()


threads = [threading.Thread(target=worker, args=g) for g in GPUS]
[t.start() for t in threads]
[t.join() for t in threads]
assert stats["chunks"] == NCH and stats["overflow"] == 0, stats
out = pathlib.Path(__file__).with_name("results") / "catalog"
out.mkdir(parents=True, exist_ok=True)
with open(out / "catalog.txt", "w") as f:
    f.write("# behavior_hash program_count example_programs (5-byte NANO, a in M5, b in M6)\n")
    for h, (c, ex) in sorted(cat.items(), key=lambda kv: -kv[1][0]):
        f.write(f"{h:016x} {c} {','.join(e.zfill(10) for e in ex)}\n")
counts = sorted((v[0] for v in cat.values()), reverse=True)
print(f"\n{len(cat):,} different useful behaviors among {stats['useful']:,} useful programs "
      f"(of 1,099,511,627,776) in {time.perf_counter() - t0:.0f} s")
print(f"behaviors with 1 program: {sum(1 for c in counts if c == 1):,}, with <= 10: {sum(1 for c in counts if c <= 10):,}, "
      f"top 2 behaviors cover {100 * sum(counts[:2]) / stats['useful']:.1f}% of useful programs")
print(f"saved {out / 'catalog.txt'}")
