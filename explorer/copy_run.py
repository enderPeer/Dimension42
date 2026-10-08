"""Find every NANO program of 1..5 bytes that copies itself (see nano_copy.cu), then check each
copier independently with a plain Python interpreter. Writes results/copiers/copiers_L{n}.txt."""
import collections
import pathlib
import queue
import re
import subprocess
import threading
import time

GPUS = [("adler40", 0), ("adler40", 1), ("knecht24", 0), ("knecht24", 1), ("knecht24", 2)]
CH = 1 << 30
jobs = [(5, k * CH, CH) for k in range(1024)] + [(4, k * CH, CH) for k in range(4)] + [(3, 0, 1 << 24), (2, 0, 1 << 16), (1, 0, 256)]
todo = queue.Queue()
for j in jobs:
    todo.put(j)
lock = threading.Lock()
found, done = {}, 0
t0 = time.perf_counter()


def worker(node, gpu):
    global done
    proc = subprocess.Popen(["ssh", "-o", "BatchMode=yes", node, f"cd ~/dimension42-explorer && exec ./nano_copy serve {gpu}"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    while True:
        try:
            L, s, n = todo.get_nowait()
        except queue.Empty:
            break
        proc.stdin.write(f"{L} {s} {n}\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        m = re.search(r"done L=(\d+) start=(\d+) count=\d+ copiers=(\d+) overflow=(\d+) .*cands=(.*)$", line)
        if not m or int(m.group(2)) != s or m.group(4) != "0":
            print(f"!! {node} gpu{gpu} failed on L={L} start={s}: {line[:120]!r}")
            todo.put((L, s, n))
            break
        with lock:
            for e in m.group(5).strip().split(","):
                if e:
                    p, l, k, step, intact, persists = e.split("-")
                    found[(int(l), int(p, 16))] = (int(k), int(step), int(intact), int(persists))
            done += 1
    proc.stdin.close()
    proc.wait()


threads = [threading.Thread(target=worker, args=g) for g in GPUS]
[t.start() for t in threads]
[t.join() for t in threads]
assert done == len(jobs), f"{done} of {len(jobs)} jobs"
print(f"searched all 1..5-byte programs in {time.perf_counter() - t0:.0f} s: {len(found):,} copiers")


# ---- independent check with a plain interpreter
def run_check(p, L):
    p0 = [(p >> (8 * (L - 1 - i))) & 255 for i in range(L)]
    M = p0 + [0] * (16 - L)
    A = pc = 0
    for step in range(1, 65):
        ins = M[pc]
        pc = (pc + 1) % L
        op, n = ins >> 4, ins & 15
        if op == 15:
            break
        v, oldA = M[n], A
        if op == 1: A = n
        elif op == 2: A = (A + n) & 255
        elif op == 3: A = (A - n) & 255
        elif op == 4: A = v
        elif op == 6: A = (A + v) & 255
        elif op == 11: A = ((A << (n & 7)) | (A >> (8 - (n & 7)))) & 255
        elif op == 12: A ^= v
        if op in (5, 9, 10, 14):
            M[n] = {5: oldA, 9: (v + 1) & 255, 10: (v - 1) & 255, 14: 255 - v}[op]
            for k in range(L, 16 - L + 1):
                if M[k:k + L] == p0:
                    return k, step, int(M[:L] == p0)
        if op == 7 or (op == 8 and oldA == 0):
            pc = n % L
    return None


bad = [(L, p) for (L, p), (k, step, intact, _) in found.items() if run_check(p, L) != (k, step, intact)]
print(f"independent check: {len(found) - len(bad):,} of {len(found):,} confirmed" + (f", MISMATCH e.g. {bad[:3]}" if bad else ""))
out = pathlib.Path(__file__).with_name("results") / "copiers"
out.mkdir(parents=True, exist_ok=True)
by = collections.defaultdict(list)
for (L, p), v in found.items():
    by[L].append((p, *v))
for L in range(1, 6):
    rows = sorted(by[L], key=lambda r: (-r[3], r[2], r[0]))
    with open(out / f"copiers_L{L}.txt", "w") as f:
        f.write("# program copy_at_offset first_step original_intact copy_still_there_at_end\n")
        for p, k, step, intact, persists in rows:
            f.write(f"{p:0{2 * L}X} {k} {step} {intact} {persists}\n")
    real = sum(1 for r in rows if r[3])
    print(f"  {L} bytes: {len(rows):,} copiers, {real:,} with the original still intact (real reproduction)")
