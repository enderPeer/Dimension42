"""Export a small, exhaustively checked NANO trace sample using only the CPU."""
from pathlib import Path
import hashlib
import json
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "explorer"))
from demo import run
from nano_ref import first_out


def main():
    source = ROOT / "explorer/results/add2/adders_L5.txt"
    programs = [int(line.split()[0], 16) for line in source.read_text().splitlines()
                if line and not line.startswith("#")][:4]
    tasks = [(p, "add_mod_256", lambda a, b: (a + b) & 255) for p in programs]
    tasks.append((0x17C5915071, "xor_with_7", lambda a, b: a ^ b ^ 7))
    rng = random.Random(42)
    pairs = [(0, 0), (255, 1), (128, 128), (255, 255)] + [(rng.randrange(256), rng.randrange(256)) for _ in range(28)]
    rows, proofs = [], []
    for program, task, target in tasks:
        outputs = bytearray()
        for a in range(256):
            for b in range(256):
                output = first_out(program, 5, a, b)
                if output != target(a, b):
                    raise RuntimeError(f"Verification failed for {program:010X}, inputs {a}, {b}")
                outputs.append(output)
        proofs.append({"program_hex": f"{program:010X}", "task": task, "input_pairs_checked": 65536,
                       "failures": 0, "outputs_sha256_a_major_b_minor": hashlib.sha256(outputs).hexdigest()})
        for a, b in pairs:
            output, trace = run(f"{program:010X}", {5: a, 6: b}, trace=True)
            if output != target(a, b):
                raise RuntimeError("Trace interpreter disagrees with the checked reference")
            rows.append({"program_hex": f"{program:010X}", "task": task, "input_a": a, "input_b": b,
                         "output": output, "trace": trace})
    destination = ROOT / "datasets/nano-adders-sample"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "traces.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")
    (destination / "verification.json").write_text(json.dumps({
        "source": "https://github.com/enderPeer/Dimension42",
        "reference": "explorer/nano_ref.py:first_out",
        "reference_sha256": hashlib.sha256((ROOT / "explorer/nano_ref.py").read_bytes()).hexdigest(),
        "semantics": "5-byte NANO; inputs M5/M6; A and other non-code cells zero; first output within 64 steps",
        "programs": proofs, "trace_rows": len(rows), "sample_seed": 42,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Verified {len(tasks)} programs on {len(tasks) * 65536:,} pairs; exported {len(rows)} traces")


if __name__ == "__main__":
    main()
