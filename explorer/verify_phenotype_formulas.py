"""Verify three six-byte formulas against demo.run on every byte input pair."""
import collections
import hashlib
import json
from pathlib import Path

from demo import run


def inverted_nibble_swap(a, b):
    value = a ^ 255
    return ((value << 4) | (value >> 4)) & 255


def xor_add_permutation(a, b):
    value = 0
    for _ in range(7):
        value = ((value ^ b) + 3) & 255
    value = ((value ^ (b ^ 255)) + 3) & 255
    return value ^ (b ^ 255)


def xor_increment_partition(a, b):
    value = 0
    for _ in range(4):
        value = ((value ^ b) + 1) & 255
    for _ in range(2):
        value = ((value ^ (b ^ 255)) + 1) & 255
    return value


SPECS = [
    ("A5A1E168E74E", inverted_nibble_swap,
     "ROL8(a XOR 255, 4)", "a", 57),
    ("77C7ED5CA223", xor_add_permutation,
     "x=0; repeat 7 times: x=((x XOR b)+3) mod 256; "
     "x=((x XOR (b XOR 255))+3) mod 256; return x XOR (b XOR 255)",
     "b", 54),
    ("C731EA2259A2", xor_increment_partition,
     "x=0; repeat 4 times: x=((x XOR b)+1) mod 256; "
     "repeat 2 times: x=((x XOR (b XOR 255))+1) mod 256; return x",
     "b", 52),
]


def main():
    results = []
    for code, formula, dependency, argument, steps in SPECS:
        answers = bytearray()
        for a in range(256):
            for b in range(256):
                output, _ = run(code, {6: a, 7: b})
                expected = formula(a, b)
                assert output == expected, (code, a, b, output, expected)
                answers.append(output)
        unary_answers = [formula(i, 0) if argument == "a" else formula(0, i)
                         for i in range(256)]
        output_counts = collections.Counter(unary_answers)
        observed, trace = run(code, {6: 3, 7: 5}, trace=True)
        assert len(trace) == steps
        result = {
            "program": code,
            "formula": dependency,
            "depends_on": argument,
            "verified_input_pairs": 65536,
            "formula_mismatches": 0,
            "truth_table_sha256": hashlib.sha256(answers).hexdigest(),
            "distinct_outputs": len(output_counts),
            "is_permutation_of_dependent_input": len(output_counts) == 256,
            "unary_output_frequencies": dict(sorted(output_counts.items())),
            "unary_input_0_through_15_outputs": unary_answers[:16],
            "example_a3_b5": {"output": observed, "steps": len(trace)},
            "example_trace_a3_b5": trace,
        }
        results.append(result)
        print(code, "all 65,536 pairs passed; distinct outputs", len(output_counts),
              "input 0..15:", unary_answers[:16])
        if len(output_counts) < 20:
            print("unary frequencies:", dict(sorted(output_counts.items())))
    target = Path(__file__).parent / "results/L6_phenotypes/formulas-last-three.json"
    target.write_text(json.dumps({
        "schema": "nano6-independent-formula-verification-v1",
        "interpreter": "explorer/demo.py:run",
        "semantics": "original NANO; 16-byte memory; first OUT within 64 steps; M6=a, M7=b",
        "rarity_scope": "No global rarity claim; candidates selected from a uniform sample",
        "results": results,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
