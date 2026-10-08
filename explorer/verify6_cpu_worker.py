"""Low-priority, bounded CPU rechecks of completed L6 chunks; never writes GPU map files.

Run one independent worker per node using a frozen JSON plan. Each full chunk is
recomputed in small segments with nano_cpu --nooutput. Only counts and checksums
are retained. These are consistency checks, not byte-for-byte or formal proofs.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time

CHUNK = 1 << 32
MASK64 = (1 << 64) - 1
REPORT = re.compile(r"programs=(\d+) hashsum=(\d+) counts=([\d,]+) secs=([\d.]+) device=CPU-(\d+)threads")


def parse_report(stderr, expected_count):
    matches = REPORT.findall(stderr)
    if len(matches) != 1:
        raise ValueError("CPU returned no unique result report")
    count, checksum, counts, seconds, threads = matches[0]
    counts = list(map(int, counts.split(",")))
    if int(count) != expected_count or len(counts) != 6 or sum(counts) != expected_count:
        raise ValueError("CPU report does not cover the requested program range")
    return {"programs": int(count), "hashsum": int(checksum), "counts": counts,
            "cpu_seconds": float(seconds), "threads": int(threads)}


def merge(total, part):
    return {"programs": total["programs"] + part["programs"],
            "hashsum": (total["hashsum"] + part["hashsum"]) & MASK64,
            "counts": [a + b for a, b in zip(total["counts"], part["counts"])]}


def matches_expected(total, expected):
    return (total["programs"] == CHUNK and sum(total["counts"]) == CHUNK
            and total["counts"] == expected["counts"] and total["hashsum"] == expected["hashsum"])


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def resource_pause():
    # Keep this worker subordinate to existing jobs and stop launching segments
    # when the host is under pressure. Scheduler niceness applies continuously.
    cpus = os.cpu_count() or 1
    if os.getloadavg()[0] > cpus * 0.85:
        return "high host load"
    mem = dict((line.split(":", 1)[0], line.split(":", 1)[1].strip())
               for line in Path("/proc/meminfo").read_text().splitlines())
    if int(mem["MemAvailable"].split()[0]) < 4 * 1024 * 1024:
        return "less than 4 GiB available RAM"
    for sensor in Path("/sys/class/hwmon").glob("hwmon*"):
        try:
            if (sensor / "name").read_text().strip() not in {"coretemp", "k10temp"}:
                continue
            if any(int(p.read_text()) >= 85000 for p in sensor.glob("temp*_input")):
                return "CPU temperature at least 85 C"
        except (OSError, ValueError):
            continue
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--segment", type=int, default=1 << 24)
    parser.add_argument("--max-hours", type=float, default=4)
    args = parser.parse_args()
    if not 1 <= args.threads <= 2 or not 1 <= args.segment <= 1 << 26 or not 0 < args.max_hours <= 12:
        parser.error("Use 1–2 threads, segments of at most 2^26, and a time budget of at most 12 hours")
    args.binary = args.binary.resolve()
    plan = json.loads(args.plan.read_text())
    chunks = plan["chunks"]
    if len({r["chunk"] for r in chunks}) != len(chunks):
        raise ValueError("Plan repeats a chunk")
    for row in chunks:
        if not 0 <= row["chunk"] < 65536 or len(row["counts"]) != 6 or sum(row["counts"]) != CHUNK:
            raise ValueError("Invalid expected chunk")
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (args.output.parent / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    current_nice = os.nice(0)
    if current_nice < 19:
        os.nice(19 - current_nice)
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    os.environ["OMP_DYNAMIC"] = "FALSE"
    status_path = args.output / "status.json"
    checkpoint_path = args.output / "checkpoint.json"
    results_path = args.output / "results.jsonl"
    stop_path = args.output / "STOP"
    completed = {}
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            r = json.loads(line)
            if r["matched"]:
                completed[r["chunk"]] = r
            else:
                raise RuntimeError("A prior mismatch requires review before resuming")
    started = time.monotonic()
    status = {"node": plan["node"], "run_id": plan["run_id"], "pid": os.getpid(),
              "threads": args.threads, "nice": os.nice(0), "started_utc": timestamp(),
              "planned_chunks": len(chunks), "completed_chunks": len(completed),
              "source_sha256": plan["source_sha256"], "state": "starting"}
    def update(**values):
        status.update(values)
        status["updated_utc"] = timestamp()
        write_json(status_path, status)
    try:
        for expected in chunks:
            k = expected["chunk"]
            if k in completed:
                continue
            total = {"programs": 0, "hashsum": 0, "counts": [0] * 6}
            if checkpoint_path.exists():
                previous = json.loads(checkpoint_path.read_text())
                if previous["chunk"] == k:
                    total = previous["total"]
            print(f"{timestamp()} checking chunk {k} from {expected['device']}", flush=True)
            while total["programs"] < CHUNK:
                if stop_path.exists() or time.monotonic() - started >= args.max_hours * 3600:
                    update(state="stopped", reason="stop file or time budget", chunk=k,
                           programs_in_chunk=total["programs"])
                    return
                reason = resource_pause()
                if reason:
                    update(state="waiting_for_headroom", reason=reason)
                    time.sleep(10)
                    continue
                update(state="running", reason=None, chunk=k, gpu_device=expected["device"],
                       programs_in_chunk=total["programs"])
                count = min(args.segment, CHUNK - total["programs"])
                command = [str(args.binary), "6", str(k * CHUNK + total["programs"]),
                           str(count), str(args.threads), "--nooutput"]
                result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                        text=True, check=True, timeout=180)
                part = parse_report(result.stderr, count)
                total = merge(total, part)
                write_json(checkpoint_path, {"chunk": k, "total": total})
                update(programs_in_chunk=total["programs"],
                       last_segment_programs_per_second=count / max(part["cpu_seconds"], 1e-9))
            record = {"chunk": k, "node": plan["node"], "gpu_device": expected["device"],
                      "matched": matches_expected(total, expected), "actual": total,
                      "expected": {"hashsum": expected["hashsum"], "counts": expected["counts"]},
                      "checked_utc": timestamp()}
            with results_path.open("a") as out:
                out.write(json.dumps(record) + "\n")
            print(f"{timestamp()} chunk {k}: {'MATCH' if record['matched'] else 'MISMATCH'}", flush=True)
            if not record["matched"]:
                update(state="mismatch", chunk=k)
                return
            completed[k] = record
            update(completed_chunks=len(completed))
        update(state="complete")
    except Exception as error:
        update(state="error", error=str(error))
        raise


if __name__ == "__main__":
    main()
