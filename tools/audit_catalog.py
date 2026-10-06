"""Recount catalog signatures and check singleton self-modification on the original probes."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "explorer"))
from demo import run

PA = [3, 200, 17, 0, 255, 91, 128, 44, 7, 160, 66, 1, 230, 99, 12, 181]
PB = [5, 13, 250, 0, 1, 77, 128, 210, 100, 33, 66, 254, 9, 140, 31, 2]


def changed_code(program, a, b):
    _, trace = run(program, {5: a, 6: b}, trace=True)
    return any(write["cell"] < 5 for step in trace for write in step["writes"])


if __name__ == "__main__":
    rows = [line.split() for line in (ROOT / "explorer/results/catalog/catalog.txt").read_text().splitlines()
            if line and not line.startswith("#")]
    singletons = [row[2].split(",")[0] for row in rows if int(row[1]) == 1]
    modifying = sum(any(changed_code(p, a, b) for a, b in zip(PA, PB)) for p in singletons)
    print(json.dumps({"probe_signatures": len(rows), "singleton_signatures": len(singletons),
                      "singleton_programs_modifying_code_on_any_probe": modifying,
                      "percent": 100 * modifying / len(singletons)}, indent=2))
