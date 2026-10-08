import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "explorer"))
from verify6_cpu_worker import CHUNK, MASK64, matches_expected, merge, parse_report


class CPUVerificationTests(unittest.TestCase):
    def test_requires_exact_coverage(self):
        report = "programs=3 hashsum=7 counts=1,0,0,0,1,1 secs=0.001 device=CPU-2threads"
        self.assertEqual(parse_report(report, 3)["hashsum"], 7)
        for broken in ("", report + "\n" + report, report.replace("counts=1", "counts=2")):
            with self.assertRaises(ValueError):
                parse_report(broken, 3)
        with self.assertRaises(ValueError):
            parse_report(report, 4)

    def test_segment_sums_wrap_and_comparison_checks_every_count(self):
        a = {"programs": CHUNK - 1, "hashsum": MASK64, "counts": [CHUNK - 1, 0, 0, 0, 0, 0]}
        b = {"programs": 1, "hashsum": 2, "counts": [0, 1, 0, 0, 0, 0]}
        total = merge(a, b)
        expected = {"hashsum": 1, "counts": [CHUNK - 1, 1, 0, 0, 0, 0]}
        self.assertTrue(matches_expected(total, expected))
        self.assertFalse(matches_expected(a, expected))
        self.assertFalse(matches_expected(total, {**expected, "hashsum": 2}))
        self.assertFalse(matches_expected(total, {**expected, "counts": [CHUNK, 0, 0, 0, 0, 0]}))


if __name__ == "__main__":
    unittest.main()
