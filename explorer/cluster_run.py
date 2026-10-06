"""Run every NANO program of L bytes on the whole cluster at once and collect the results here.

Each device gets a contiguous slice proportional to its measured sustained speed.
All devices on a node start together; afterwards the class bytes of every slice are
copied to this PC, assembled in program order and checked.

usage: python cluster_run.py [L] [--only DEVICE]
"""
import concurrent.futures as cf
import hashlib
import pathlib
import re
import subprocess
import sys
import time

# measured sustained rates (million programs / s), full 4-byte space, 2026-10-06
DEVICES = [
    # node,       label,          command (L start count appended / inserted),         rate
    ("adler40",  "RTX 4090",      "./nano_cuda {L} {s} {n} 0",                         4564),
    ("adler40",  "RTX 4080",      "./nano_cuda {L} {s} {n} 1",                         2820),
    ("adler40",  "CPU 22 thr",    "nice -n 19 ./nano_cpu {L} {s} {n} 22",                70),
    ("knecht24", "RTX 3060 #0",   "./nano_cuda {L} {s} {n} 0",                          724),
    ("knecht24", "RTX 3060 #1",   "./nano_cuda {L} {s} {n} 1",                          742),
    ("knecht24", "RTX 3060 #2",   "./nano_cuda {L} {s} {n} 2",                          746),
    ("knecht24", "CPU 21 thr",    "nice -n 19 ./nano_cpu {L} {s} {n} 21",                40),
    ("specht32", "RX 9070 XT",    "./nano_vk {L} {s} {n} 1",                           2018),
    ("specht32", "RX 9060 XT",    "./nano_vk {L} {s} {n} 0",                           1124),
    ("specht32", "CPU 10 thr",    "nice -n 19 ./nano_cpu {L} {s} {n} 10",                25),
    ("falke64",  "R9700 #0",      "./nano_vk {L} {s} {n} 0",                           2160),
    ("falke64",  "R9700 #1",      "./nano_vk {L} {s} {n} 1",                           2028),
    ("falke64",  "CPU 10 thr",    "nice -n 19 ./nano_cpu {L} {s} {n} 10",                38),
]

L = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 3
only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
NOOUT = "--nooutput" in sys.argv   # count + hash only, no class bytes copied back
devs = [d for d in DEVICES if only is None or d[1] == only]
TOTAL = 256 ** L

# slices proportional to speed, multiples of 1024, last one takes the rest
rate_sum = sum(d[3] for d in devs)
slices, s = [], 0
for i, d in enumerate(devs):
    n = TOTAL - s if i == len(devs) - 1 else (TOTAL * d[3] // rate_sum) // 1024 * 1024
    slices.append((d, s, n))
    s += n
assert s == TOTAL


def run_node(node):
    mine = [(i, d, s, n) for i, (d, s, n) in enumerate(slices) if d[0] == node]
    script = ["cd ~/dimension42-explorer", "t0=$(date +%s.%N)"]
    for i, d, s, n in mine:
        script.append(f"({d[2].format(L=L, s=s, n=n)}{' --nooutput' if NOOUT else ''} > part{i}.bin 2> part{i}.log) &")
    script += ["wait", "t1=$(date +%s.%N)", 'echo "node_secs=$(echo "$t1 - $t0" | bc)"']
    script += [f'echo "part{i} $(grep -hE "programs=|ERROR" part{i}.log)"' for i, *_ in mine]
    t = time.perf_counter()
    res = subprocess.run(["ssh", "-o", "BatchMode=yes", node, "\n".join(script)],
                         capture_output=True, text=True)
    t_done = time.perf_counter()
    # copy the class bytes of every slice back here
    data = subprocess.run(["ssh", "-o", "BatchMode=yes", node,
                           "cd ~/dimension42-explorer && cat " + " ".join(f"part{i}.bin" for i, *_ in mine)
                           + " && rm -f " + " ".join(f"part{i}.bin part{i}.log" for i, *_ in mine)],
                          capture_output=True).stdout
    t_copied = time.perf_counter()
    return node, mine, res.stdout, data, t_done - t, t_copied - t


print(f"L={L}: {TOTAL:,} programs on {len(devs)} devices")
t0 = time.perf_counter()
with cf.ThreadPoolExecutor() as ex:
    results = list(ex.map(run_node, sorted({d[0] for d in devs})))
t_all = time.perf_counter() - t0

classes = bytearray(0 if NOOUT else TOTAL)
hashsum, counts, compute = 0, [0] * 6, []
for node, mine, out, data, t_run, t_copy in results:
    node_secs = float(re.search(r"node_secs=([\d.]+)", out).group(1))
    print(f"\n{node}: on-node {node_secs:.3f} s, incl. ssh {t_run:.3f} s, incl. copy-back {t_copy:.3f} s")
    off = 0
    for i, d, s, n in mine:
        line = re.search(rf"^part{i} (.*)$", out, re.M).group(1)
        if "ERROR" in line or "programs=" not in line:
            sys.exit(f"  {d[1]}: FAILED: {line}")
        secs = float(re.search(r"secs=([\d.]+)", line).group(1))
        hashsum += int(re.search(r"hashsum=(\d+)", line).group(1))
        counts = [a + int(b) for a, b in zip(counts, re.search(r"counts=([\d,]+)", line).group(1).split(","))]
        compute.append(secs)
        if not NOOUT:
            classes[s:s + n] = data[off:off + n]
            off += n
        print(f"  {d[1]:12s} programs {s:>10,} .. {s + n - 1:>10,}  ({n:>10,})  compute {secs * 1000:8.2f} ms")
    assert off == len(data), f"{node}: got {len(data)} bytes, expected {off}"

hashsum %= 2 ** 64
names = ["IDLE", "HALT", "ACTIVE", "SELF-MOD", "OUTPUT", "DRAW"]
print(f"\nall programs computed : {sum(counts):,} of {TOTAL:,}")
print("classes               : " + ", ".join(f"{k} {v:,}" for k, v in zip(names, counts)))
print(f"hash of all states    : {hashsum}")
if not NOOUT:
    print(f"md5 of class file     : {hashlib.md5(classes).hexdigest()}")
print(f"compute (slowest dev) : {max(compute) * 1000:.2f} ms")
copied = "no results copied back" if NOOUT else f"copy {TOTAL / 2**20:.0f} MiB of results back"
print(f"end-to-end on this PC : {t_all:.3f} s (start over SSH, run, {copied})")
if not NOOUT:
    out = pathlib.Path(__file__).with_name("results") / f"classes_L{L}.bin"
    out.parent.mkdir(exist_ok=True)
    out.write_bytes(classes)
    print(f"saved                 : {out}")
