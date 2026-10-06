"""Checks for the portable quickstart and false-positive verification handling."""
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "explorer"), str(ROOT / "explorer/ai")]
from demo import run
from nano_ref import first_out
from experiment_support import parse_proof_output
from prepare_data import prepare


class PublicQuickstartTests(unittest.TestCase):
    def test_proof_requires_all_candidates_and_correct_layout(self):
        line = "5 456657d7f0 5 6 7 fails=0 outputs=1 halts=0 steps=4\n"
        p = 0x456657D7F0
        self.assertEqual(parse_proof_output(line, [p], (5, 6, 7)), {p: True})
        for output, programs, cells in [("", [p], (5, 6, 7)), (line * 2, [p], (5, 6, 7)),
                                        (line, [p, 1], (5, 6, 7)), (line, [p], (5, 7, 9)),
                                        ("ERROR unavailable GPU", [p], (5, 6, 7))]:
            with self.subTest(output=output, cells=cells), self.assertRaises(RuntimeError):
                parse_proof_output(output, programs, cells)
        self.assertFalse(parse_proof_output(line.replace("fails=0", "fails=1"), [p], (5, 6, 7))[p])

    def test_data_preparation_and_preflight_need_no_torch_or_cluster(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "data"
            files = prepare(ROOT / "explorer/results", dest)
            self.assertEqual(len(files), 23)
            self.assertIn("adders_c5_6.txt", files)
            self.assertIn("add3_c5_6_8.txt", files)
            prepare(ROOT / "explorer/results", dest)  # identical files are safe to reuse
            check = subprocess.run([sys.executable, str(ROOT / "explorer/ai/test_three.py"),
                                    "--check-only", "--data", str(dest), "--verifier", sys.executable],
                                   capture_output=True, text=True)
            self.assertEqual(check.returncode, 0, check.stderr)
            (dest / "adders_c5_6.txt").write_text("different local data")
            with self.assertRaises(FileExistsError):
                prepare(ROOT / "explorer/results", dest)

    def test_trace_interpreter_matches_existing_reference(self):
        rng = random.Random(123)
        for _ in range(2000):
            p, a, b = rng.randrange(1 << 40), rng.randrange(256), rng.randrange(256)
            output, _ = run(f"{p:010X}", {5: a, 6: b}, trace=True)
            self.assertEqual(output, first_out(p, 5, a, b))

    def test_self_modifying_highlights(self):
        output, trace = run("17C5915071", {5: 255, 6: 128}, trace=True)
        self.assertEqual(output, 255 ^ 128 ^ 7)
        self.assertEqual((len(trace), trace[-1]["byte"]), (46, "D0"))
        for inputs in ({5: 1, 6: 2, 7: 3}, {5: 255, 6: 255, 7: 255}, {5: 0, 6: 0, 7: 0}):
            output, trace = run("67A05CE2", inputs, trace=True)
            self.assertEqual(output, sum(inputs.values()) & 255)
            self.assertEqual((len(trace), trace[-1]["byte"]), (16, "DF"))

    def test_exported_traces_match_reference(self):
        path = ROOT / "datasets/nano-adders-sample/traces.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(rows), 160)
        for row in rows:
            self.assertEqual(row["output"], first_out(int(row["program_hex"], 16), 5, row["input_a"], row["input_b"]))


if __name__ == "__main__":
    unittest.main()
