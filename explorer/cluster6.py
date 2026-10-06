"""Map every 6-byte NANO program (2^48 = 281 trillion) and store the class of each one on the cluster.

65,536 chunks of 2^32 programs; chunk k = programs k*2^32 .. (k+1)*2^32-1, stored zstd-compressed as
L6/xx/yy.zst (xx = first program byte, yy = second). GPUs: everything except the RTX 4090 (it trains the AI).
Storage: adler40's 4080 writes on adler40; all other GPUs write to falke64 (knecht24 and specht32 send over the
LAN with a restricted key that can only store files below ~/dimension42-explorer/L6).
Safety: a guard checks free disk space every minute and stops a storage node below its reserve.
Resumable: every finished chunk is appended to results/L6/chunks.jsonl; a restart skips them.

usage: python cluster6.py
"""
import json
import pathlib
import queue
import re
import subprocess
import threading
import time

L, CH, NCH = 6, 1 << 32, 1 << 16
OUT = pathlib.Path(__file__).with_name("results") / "L6"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "chunks.jsonl"
SINK = "NANO_SINK='ssh -i ~/.ssh/d42_transfer -o BatchMode=yes -o IdentitiesOnly=yes ender@192.168.178.188'"
# node, label, command, where the data lands
GPUS = [
    ("adler40",  "RTX 4080",    f"./nano_cuda serve {L} 1 10",           "adler40"),
    ("knecht24", "RTX 3060 #0", f"{SINK} ./nano_cuda serve {L} 0 6",     "falke64"),
    ("knecht24", "RTX 3060 #1", f"{SINK} ./nano_cuda serve {L} 1 6",     "falke64"),
    ("knecht24", "RTX 3060 #2", f"{SINK} ./nano_cuda serve {L} 2 6",     "falke64"),
    ("specht32", "RX 9070 XT",  f"{SINK} ./nano_vk serve {L} 1 6",       "falke64"),
    ("specht32", "RX 9060 XT",  f"{SINK} ./nano_vk serve {L} 0 5",       "falke64"),
    ("falke64",  "R9700 #0",    f"./nano_vk serve {L} 0 6",              "falke64"),
    ("falke64",  "R9700 #1",    f"./nano_vk serve {L} 1 6",              "falke64"),
]
RESERVE_GB = {"adler40": 230, "falke64": 450}       # keep at least 25 % of each storage disk free
full = {n: False for n in RESERVE_GB}
free_gb = {n: 0 for n in RESERVE_GB}
stop = threading.Event()
lock = threading.Lock()

done = set()
if LOG.exists():
    for line in LOG.read_text().splitlines():
        if line.strip():
            done.add(json.loads(line)["chunk"])
todo = queue.Queue()
for k in range(NCH):
    if k not in done:
        todo.put(k)
start_done, t0 = len(done), time.perf_counter()
logf = open(LOG, "a")


def ssh(node, cmd):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", node, cmd], capture_output=True, text=True).stdout


def guard():
    while not stop.is_set():
        for node, reserve in RESERVE_GB.items():
            try:
                gb = int(ssh(node, "df -BG --output=avail ~ | tail -1").strip().rstrip("G"))
            except ValueError:
                continue
            free_gb[node] = gb
            if gb < reserve and not full[node]:
                full[node] = True
                print(f"!! {node}: only {gb} GB free (reserve {reserve} GB) - no more chunks are stored there", flush=True)
        stop.wait(60)


def worker(node, label, cmd, dest):
    proc = subprocess.Popen(["ssh", "-o", "BatchMode=yes", node, f"cd ~/dimension42-explorer && exec env {cmd}"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    while not stop.is_set() and not full[dest]:
        try:
            k = todo.get_nowait()
        except queue.Empty:
            break
        path = f"L6/{k >> 8:02x}/{k & 255:02x}.zst"
        proc.stdin.write(f"{k * CH} {CH} {path}\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        m = re.search(r"done start=(\d+) programs=(\d+) hashsum=(\d+) counts=([\d,]+) secs=([\d.]+)", line)
        if not m or int(m.group(1)) != k * CH:
            print(f"!! {label} failed on chunk {k}: {line[:150]!r} - chunk put back, device retired", flush=True)
            todo.put(k)
            break
        rec = {"chunk": k, "file": path, "node": dest, "device": label, "hashsum": int(m.group(3)),
               "counts": [int(c) for c in m.group(4).split(",")], "secs": float(m.group(5))}
        with lock:
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            done.add(k)
            n = len(done)
            if n % 256 == 0:
                el = time.perf_counter() - t0
                rate = (n - start_done) * CH / el
                eta = (NCH - n) * CH / rate / 3600 if rate else 0
                print(f"  {n:6d}/{NCH} chunks ({100 * n / NCH:5.1f}%)  {rate / 1e9:5.2f} G programs/s  "
                      f"ETA {eta:4.1f} h  free: falke {free_gb['falke64']} GB, adler {free_gb['adler40']} GB", flush=True)
    proc.stdin.close()
    proc.wait()


ssh("adler40", "mkdir -p ~/dimension42-explorer/L6")
print(f"6-byte map: {NCH - len(done):,} of {NCH:,} chunks to do ({len(done):,} already done)", flush=True)
g = threading.Thread(target=guard, daemon=True)
g.start()
time.sleep(3)
threads = [threading.Thread(target=worker, args=d) for d in GPUS]
[t.start() for t in threads]
[t.join() for t in threads]
stop.set()
print(f"finished: {len(done):,} of {NCH:,} chunks done, {time.perf_counter() - t0:.0f} s this run", flush=True)
