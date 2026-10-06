"""Check the 5-byte results independently of the run that produced them.

1. On every node: decompress every chunk file it holds, count the class bytes with a
   separate program (nano_count) and compare with what the GPU reported: exactly 2^30
   bytes, no invalid byte, identical counts. Also records the sha256 of each file.
2. Recompute 16 random chunks on a different kind of GPU (NVIDIA <-> AMD) and compare
   the hash over all final states.
Adds sha256 + verification results to results/L5/manifest.json.
"""
import json
import pathlib
import random
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

man_path = pathlib.Path(__file__).with_name("results") / "L5" / "manifest.json"
man = json.loads(man_path.read_text())
chunks = man["chunk"]
CHUNK = man["chunk_programs"]


def ssh(node, cmd):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", node, cmd], capture_output=True, text=True).stdout


def check_node(node):
    script = ("cd ~/dimension42-explorer/L5 && ls chunk_*.zst | nice -n 19 xargs -P 6 -I{} sh -c "
              "'echo \"{} $(zstd -dc {} | ../nano_count) sha256=$(sha256sum < {} | cut -c1-64) size=$(stat -c %s {})\"'")
    return node, ssh(node, script)


nodes = sorted({c["node"] for c in chunks.values()})
bad = 0
seen = set()
with ThreadPoolExecutor() as ex:
    for node, out in ex.map(check_node, nodes):
        n_ok = 0
        for line in out.splitlines():
            m = re.match(r"chunk_(\d+)\.zst bytes=(\d+) counts=([\d,]+) bad=(\d+) sha256=(\w+) size=(\d+)", line)
            if not m:
                continue
            k = str(int(m.group(1)))
            c = chunks[k]
            ok = (c["node"] == node and int(m.group(2)) == CHUNK and int(m.group(4)) == 0
                  and [int(x) for x in m.group(3).split(",")] == c["counts"])
            c["sha256"], c["zst_bytes"], c["file_check"] = m.group(5), int(m.group(6)), ok
            seen.add(k)
            n_ok += ok
            bad += not ok
            if not ok:
                print(f"!! chunk {k} on {node}: {line}")
        print(f"{node}: {n_ok} chunk files decompressed and recounted, all match: {n_ok == sum(1 for c in chunks.values() if c['node'] == node)}")
missing = set(chunks) - seen
print(f"files checked: {len(seen)} of {len(chunks)}, mismatches: {bad}, missing: {len(missing)}")

# 2. recompute random chunks on the other GPU family
random.seed(42)
sample = random.sample(sorted(chunks, key=int), 16)
other = {"adler40": ("falke64", "./nano_vk", 0), "knecht24": ("specht32", "./nano_vk", 1),
         "specht32": ("adler40", "./nano_cuda", 0), "falke64": ("adler40", "./nano_cuda", 1)}
recheck = []
for k in sample:
    c = chunks[k]
    node, binary, dev = other[c["node"]]
    out = ssh(node, f"cd ~/dimension42-explorer && {binary} 5 {int(k) * CHUNK} {CHUNK} {dev} --nooutput 2>&1")
    h = int(re.search(r"hashsum=(\d+)", out).group(1))
    same = h == c["hashsum"]
    recheck.append({"chunk": int(k), "first": c["device"], "second": re.search(r"device=(.*)", out).group(1).strip(), "same": same})
    print(f"chunk {int(k):4d}: {c['device']:12s} vs {recheck[-1]['second']:38s} {'SAME' if same else 'DIFFERENT'}")
man["verification"] = {"files_recounted": len(seen), "file_mismatches": bad,
                       "recomputed_on_other_gpu": recheck,
                       "recompute_all_same": all(r["same"] for r in recheck)}
man_path.write_text(json.dumps(man, indent=1))
print(f"total compressed size: {sum(c.get('zst_bytes', 0) for c in chunks.values()) / 2**30:.2f} GiB")
