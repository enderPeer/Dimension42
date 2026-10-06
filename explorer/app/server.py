"""Dimension42 program explorer: local web server over the verified 5-byte results.

    python app/server.py          then open http://127.0.0.1:8742

Data: results/L5/chunk_NNNN.zst (1024 chunks, each 2^30 class bytes) + manifest.json.
Only listens on 127.0.0.1.
"""
import http.server
import json
import pathlib
import random
import re
import threading
from collections import OrderedDict

import numpy as np
import zstandard

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT.parent / "results" / "L5"
CACHE = DATA / "cache"
CACHE.mkdir(exist_ok=True)
MAN = json.loads((DATA / "manifest.json").read_text())
CHUNK = MAN["chunk_programs"]                       # 2^30
NCH = MAN["chunks"]                                 # 1024
COUNTS = np.array([MAN["chunk"][str(k)]["counts"] for k in range(NCH)], dtype=np.int64)   # (1024, 6)
NODE = [MAN["chunk"][str(k)]["device"] + " @ " + MAN["chunk"][str(k)]["node"] for k in range(NCH)]

# class colours (index = class 1..6): IDLE HALT ACTIVE SELF-MOD OUTPUT DRAW
COLORS = np.array([[96, 96, 104], [214, 72, 72], [64, 196, 222], [236, 210, 64],
                   [92, 206, 104], [206, 98, 222]], dtype=np.float32)

_chunks = OrderedDict()                             # small LRU of decompressed chunks
_lock = threading.Lock()


def load_chunk(k):
    """Decompress chunk k (1 GiB) into memory; keep the 2 most recent."""
    with _lock:
        if k in _chunks:
            _chunks.move_to_end(k)
            return _chunks[k]
    buf = bytearray(CHUNK)
    mv, n = memoryview(buf), 0
    with open(DATA / f"chunk_{k:04d}.zst", "rb") as f, \
            zstandard.ZstdDecompressor().stream_reader(f, read_across_frames=True) as r:
        while n < CHUNK:
            got = r.readinto(mv[n:])
            if not got:
                break
            n += got
    if n != CHUNK:
        raise IOError(f"chunk {k}: {n} bytes, expected {CHUNK}")
    arr = np.frombuffer(buf, dtype=np.uint8)
    with _lock:
        _chunks[k] = arr
        while len(_chunks) > 2:
            _chunks.popitem(last=False)
    return arr


def chunk_image(k):
    """1024x1024 RGB picture of chunk k: pixel = 1024 consecutive programs, colour = class mix."""
    path = CACHE / f"chunk_{k:04d}.rgb"
    if path.exists():
        return path.read_bytes()
    blocks = load_chunk(k).reshape(-1, 1024)
    counts = np.empty((blocks.shape[0], 6), dtype=np.float32)
    for c in range(1, 7):
        counts[:, c - 1] = (blocks == c).sum(axis=1, dtype=np.uint16)
    rgb = (counts @ COLORS / 1024.0).clip(0, 255).astype(np.uint8).tobytes()
    path.write_bytes(rgb)
    return rgb


def overview_rgb():
    mix = COUNTS / COUNTS.sum(axis=1, keepdims=True)
    return (mix @ COLORS).clip(0, 255).astype(np.uint8).tobytes()


def random_program(cls):
    """A random program of class cls (1..6), chunk chosen in proportion to how many it holds."""
    w = COUNTS[:, cls - 1].astype(np.float64)
    k = int(np.random.choice(NCH, p=w / w.sum()))
    arr = load_chunk(k)
    for _ in range(64):                             # look in random 1 Mi windows first (fast)
        off = random.randrange(0, CHUNK, 1 << 20)
        hits = np.flatnonzero(arr[off:off + (1 << 20)] == cls)
        if len(hits):
            return k * CHUNK + off + int(random.choice(hits))
    return k * CHUNK + int(np.flatnonzero(arr == cls)[0])


ADD2 = ROOT.parent / "results" / "add2"
_adders = {}


def adders(L):
    """Proven adders of length L: list of (program, outputs at 2+2, halts, steps), clean ones first."""
    if L not in _adders:
        rows = []
        path = ADD2 / f"adders_L{L}.txt"
        if path.exists():
            for line in path.read_text().splitlines():
                if line and not line.startswith("#"):
                    p, o, h, s = line.split()
                    rows.append((int(p, 16), int(o), int(h), int(s)))
        rows.sort(key=lambda r: (not (r[1] == 1 and r[2] == 1), r[3], r[0]))
        _adders[L] = rows
    return _adders[L]


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype="application/json", code=200):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path.split("?")[0]
        try:
            if p == "/":
                return self.send((ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
            if p == "/api/summary":
                return self.send({"programs": MAN["programs"], "chunks": NCH, "chunk_programs": CHUNK,
                                  "classes": MAN["classes"], "hashsum": MAN["hashsum"],
                                  "wall_seconds": MAN["wall_seconds"], "verification": MAN.get("verification", {}),
                                  "counts": COUNTS.tolist(), "node": NODE})
            if p == "/api/overview.rgb":
                return self.send(overview_rgb(), "application/octet-stream")
            m = re.fullmatch(r"/api/chunk/(\d+)\.rgb", p)
            if m and int(m.group(1)) < NCH:
                return self.send(chunk_image(int(m.group(1))), "application/octet-stream")
            m = re.fullmatch(r"/api/block/(\d+)/(\d+)", p)
            if m and int(m.group(1)) < NCH and int(m.group(2)) < CHUNK // 1024:
                k, b = int(m.group(1)), int(m.group(2))
                return self.send(load_chunk(k)[b * 1024:(b + 1) * 1024].tobytes(), "application/octet-stream")
            m = re.fullmatch(r"/api/program/(\d+)", p)
            if m and int(m.group(1)) < MAN["programs"]:
                q = int(m.group(1))
                k = q // CHUNK
                return self.send({"program": q, "class": int(load_chunk(k)[q % CHUNK]), "chunk": k, "computed_by": NODE[k]})
            if p == "/api/add2/summary":
                f = ADD2 / "summary.json"
                return self.send(json.loads(f.read_text()) if f.exists() else {"error": "no search results yet"})
            m = re.fullmatch(r"/api/add2/list/([1-5])/(\d+)", p)
            if m:
                rows = adders(int(m.group(1)))
                off = int(m.group(2))
                return self.send({"total": len(rows), "offset": off,
                                  "rows": [{"program": r[0], "outputs": r[1], "halts": r[2], "steps": r[3]}
                                           for r in rows[off:off + 100]]})
            m = re.fullmatch(r"/api/random/([1-6])", p)
            if m:
                return self.send({"program": random_program(int(m.group(1)))})
            self.send({"error": "not found"}, code=404)
        except Exception as e:                      # keep the server alive, report the problem
            self.send({"error": str(e)}, code=500)


if __name__ == "__main__":
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8742), Handler)
    print("Dimension42 program explorer on http://127.0.0.1:8742")
    srv.serve_forever()
