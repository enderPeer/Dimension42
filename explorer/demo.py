"""Trace two tiny self-modifying programs. Uses Python's standard library only."""
import argparse

OPS = "NOP LDI ADDI SUBI LD ST ADD JMP JZ INC DEC ROL XOR OUT NOT HLT".split()


def run(program, inputs, trace=False):
    code = bytes.fromhex(program)
    if not 1 <= len(code) <= 8:
        raise ValueError("Expected 1..8 bytes of code")
    memory = list(code) + [0] * (16 - len(code))
    for cell, value in inputs.items():
        if not len(code) <= cell < 16 or not 0 <= value <= 255:
            raise ValueError("Inputs must be bytes in non-code memory")
        memory[cell] = value
    accumulator = pc = 0
    history = []
    for step in range(1, 65):
        address = pc
        instruction = memory[pc]
        op, n = instruction >> 4, instruction & 15
        pc = (pc + 1) % len(code)
        before = memory.copy() if trace else None
        old_a = accumulator
        output = None
        if op == 1: accumulator = n
        elif op == 2: accumulator = (accumulator + n) & 255
        elif op == 3: accumulator = (accumulator - n) & 255
        elif op == 4: accumulator = memory[n]
        elif op == 5: memory[n] = accumulator
        elif op == 6: accumulator = (accumulator + memory[n]) & 255
        elif op == 7: pc = n % len(code)
        elif op == 8:
            if accumulator == 0: pc = n % len(code)
        elif op == 9: memory[n] = (memory[n] + 1) & 255
        elif op == 10: memory[n] = (memory[n] - 1) & 255
        elif op == 11:
            k = n & 7
            accumulator = ((accumulator << k) | (accumulator >> (8 - k))) & 255
        elif op == 12: accumulator ^= memory[n]
        elif op == 13: output = memory[n]
        elif op == 14: memory[n] ^= 255
        if trace:
            history.append({"step": step, "pc": address, "byte": f"{instruction:02X}",
                            "instruction": f"{OPS[op]} {n:X}", "a_before": old_a, "a_after": accumulator,
                            "writes": [{"cell": i, "before": before[i], "after": value}
                                       for i, value in enumerate(memory) if before[i] != value], "output": output})
        if op in (13, 15):
            return output, history
    return None, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-xor", action="store_true", help="Check the XOR example on all 65,536 input pairs")
    args = parser.parse_args()
    for program, inputs, expected in [("17C5915071", {5: 3, 6: 5}, 3 ^ 5 ^ 7),
                                      ("67A05CE2", {5: 1, 6: 2, 7: 3}, 6)]:
        output, history = run(program, inputs, trace=True)
        print(f"\nProgram {program}; inputs {inputs}; expected first output {expected}")
        for row in history:
            writes = ", ".join(f"M[{w['cell']:X}] {w['before']:02X}->{w['after']:02X}" for w in row['writes'])
            print(f"{row['step']:2} PC={row['pc']} {row['byte']} {row['instruction']:6} "
                  f"A={row['a_before']:3}->{row['a_after']:3} {writes}")
        assert output == expected, (program, output, expected)
        print(f"First output: {output}")
    if args.verify_xor:
        from nano_ref import first_out
        for a in range(256):
            for b in range(256):
                assert first_out(0x17C5915071, 5, a, b) == a ^ b ^ 7
        print("XOR example: all 65,536 pairs passed in nano_ref.first_out (64-step limit).")


if __name__ == "__main__":
    main()
