import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "explorer"))
from nano256_ref import execute, parse_initial


class Nano256Tests(unittest.TestCase):
    def test_five_bytes_reach_highest_memory_address(self):
        result = execute(bytes.fromhex("0F 11 5F DF F0"))
        self.assertEqual(result["memory"][255], 1)
        self.assertEqual(result["outputs_first8"], [1])
        self.assertEqual(result["steps"], 5)
        self.assertTrue(result["halted"])
        self.assertFalse(result["self_modified"])

    def test_512_counts_do_not_wrap_at_256(self):
        result = execute(bytes.fromhex("D0"))
        self.assertEqual(result["steps"], 512)
        self.assertEqual(result["output_count"], 512)
        self.assertEqual(result["outputs_first8"], [208] * 8)
        self.assertEqual(result["stop_reason"], "step_limit")
        self.assertNotEqual(result["state_hash"], execute(bytes.fromhex("D0"), max_steps=256)["state_hash"])

    def test_bank_is_used_for_memory_but_not_code_fetch(self):
        result = execute(bytes.fromhex("0F 9F 71"))
        self.assertEqual(result["steps"], 512)
        self.assertEqual(result["memory"][255], 0)  # 256 increments wrap to zero
        self.assertEqual(result["seen_memory_bits"][255], 255)
        self.assertEqual(result["class"], "ACTIVE")
        self.assertEqual(result["memory"][:3], [15, 159, 113])
        short = execute(bytes.fromhex("0F 9F 71"), max_steps=64)
        self.assertEqual(short["memory"][255], 32)

    def test_cross_bank_reads_and_writes(self):
        result = execute(bytes.fromhex("01 4F 02 5E DE F0"), initial_memory={31: 213})
        self.assertEqual(result["memory"][46], 213)
        self.assertEqual(result["memory"][31], 213)
        self.assertEqual(result["outputs_first8"], [213])

    def test_delayed_output_after_original_budget(self):
        # With initial M[31]=16, this counter emits zero at step 65.
        program = bytes.fromhex("01 AF 4F 85 71 DF F0")
        before = execute(program, max_steps=64, initial_memory={31: 16})
        after = execute(program, initial_memory={31: 16})
        self.assertEqual(before["output_count"], 0)
        self.assertEqual(after["outputs_first8"], [0])
        self.assertTrue(after["halted"])
        self.assertEqual(after["steps"], 66)

    def test_self_modification_and_first_output_mode(self):
        result = execute(bytes.fromhex("17 C5 91 50 71"), initial_memory={5: 255, 6: 128}, first_output=True, trace=True)
        self.assertEqual(result["outputs_first8"], [255 ^ 128 ^ 7])
        self.assertEqual(result["steps"], 46)
        self.assertTrue(result["self_modified"])
        self.assertEqual(result["stop_reason"], "first_output")
        self.assertEqual(len(result["trace"]), 46)

    def test_invalid_requests_rejected(self):
        for program, options in [(b"", {}), (bytes(17), {}), (b"\xf0", {"max_steps": 0}),
                                 (b"\xf0", {"max_steps": 65536}), (b"\xf0", {"max_steps": 1.5}),
                                 (b"\xf0", {"initial_memory": {0: 1}}),
                                 (b"\xf0", {"initial_memory": {256: 1}}),
                                 (b"\xf0", {"initial_memory": {20: 256}})]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                execute(program, **options)
        with self.assertRaises(ValueError):
            parse_initial(["31=1", "0x1f=2"])


if __name__ == "__main__":
    unittest.main()
