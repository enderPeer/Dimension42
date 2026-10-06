"""Strict verification-result parsing shared by the portable experiment runner."""
import re


def parse_proof_output(stdout, programs, cells):
    expected = set(programs)
    results = {}
    for line in stdout.splitlines():
        match = re.fullmatch(
            r"5 ([0-9a-fA-F]+) (\d+) (\d+) (\d+) fails=(\d+) outputs=\d+ halts=\d+ steps=\d+",
            line.strip(),
        )
        if not match:
            raise RuntimeError(f"Unexpected verifier output: {line[:160]}")
        program = int(match[1], 16)
        if tuple(int(match[i]) for i in (2, 3, 4)) != tuple(cells):
            raise RuntimeError("Verifier returned a different input layout")
        if program not in expected or program in results:
            raise RuntimeError("Verifier returned an unexpected or duplicate program")
        results[program] = int(match[5]) == 0
    if set(results) != expected:
        raise RuntimeError("Verifier did not return a result for every candidate")
    return results
