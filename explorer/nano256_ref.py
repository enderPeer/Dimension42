"""Experimental NANO-256: banked 256-byte memory and a default 512-step budget.

This is a different ISA from the active 16-byte NANO mapping. No existing
interpreter, training script, OS image, or cluster process imports this module.
"""
import argparse
import json

ISA = "nano256-bank-v1"
OPS = "BANK LDI ADDI SUBI LD ST ADD JMP JZ INC DEC ROL XOR OUT NOT HLT".split()
CLASSES = "IDLE HALT ACTIVE SELF-MOD OUTPUT DRAW".split()


def execute(program, *, max_steps=512, initial_memory=None, first_output=False, trace=False):
    program = bytes(program)
    if not 1 <= len(program) <= 16:
        raise ValueError("Program must contain 1..16 bytes")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or not 1 <= max_steps <= 65535:
        raise ValueError("max_steps must be an integer from 1 to 65535")
    memory = list(program) + [0] * (256 - len(program))
    for address, value in (initial_memory or {}).items():
        if not isinstance(address, int) or not len(program) <= address < 256:
            raise ValueError("Initial data must use an address outside the program, below 256")
        if not isinstance(value, int) or not 0 <= value <= 255:
            raise ValueError("Initial memory values must be bytes")
        memory[address] = value
    seen = memory.copy()
    a = bank = pc = trace_a = trace_bank = output_count = steps = 0
    outputs, history = [], []
    halted = self_modified = False
    reason = "step_limit"
    for steps in range(1, max_steps + 1):
        at = pc
        instruction = memory[pc]
        op, n = instruction >> 4, instruction & 15
        pc = (pc + 1) % len(program)
        address = bank * 16 + n
        old_a, old_bank = a, bank
        write = output = None
        if op == 0: bank = n
        elif op == 1: a = n
        elif op == 2: a = (a + n) & 255
        elif op == 3: a = (a - n) & 255
        elif op == 4: a = memory[address]
        elif op == 5: write = a
        elif op == 6: a = (a + memory[address]) & 255
        elif op == 7: pc = n % len(program)
        elif op == 8:
            if a == 0: pc = n % len(program)
        elif op == 9: write = (memory[address] + 1) & 255
        elif op == 10: write = (memory[address] - 1) & 255
        elif op == 11:
            k = n & 7
            a = ((a << k) | (a >> (8 - k))) & 255
        elif op == 12: a ^= memory[address]
        elif op == 13:
            output = memory[address]
            output_count += 1
            if len(outputs) < 8: outputs.append(output)
        elif op == 14: write = memory[address] ^ 255
        elif op == 15:
            halted = True
            reason = "halt"
        change = None
        if write is not None:
            if memory[address] != write:
                self_modified |= address < len(program)
                change = {"address": address, "before": memory[address], "after": write}
            memory[address] = write
            seen[address] |= write
        trace_a |= a
        trace_bank |= bank
        if trace:
            history.append({"step": steps, "pc": at, "byte": f"{instruction:02X}",
                            "instruction": f"{OPS[op]} {n:X}", "a_before": old_a, "a_after": a,
                            "bank_before": old_bank, "bank_after": bank, "next_pc": pc,
                            "write": change, "output": output})
        if halted: break
        if first_output and output is not None:
            reason = "first_output"
            break
    final_code_changed = memory[:len(program)] != list(program)
    active = trace_a or trace_bank or any(seen[len(program):8]) or any(seen[16:])
    cls = 6 if any(seen[8:16]) else 5 if output_count else 4 if final_code_changed else 3 if active else 2 if halted else 1
    # New, explicitly versioned checksum layout; counters use 32-bit LE encoding.
    payload = (b"NANO256v1" + bytes([len(program), a, bank, pc, int(halted), trace_a, trace_bank, cls, int(self_modified)])
               + steps.to_bytes(4, "little") + output_count.to_bytes(4, "little")
               + bytes(memory) + bytes(seen) + bytes(outputs + [0] * (8 - len(outputs))))
    checksum = 2166136261
    for value in payload:
        checksum = ((checksum ^ value) * 16777619) & 0xFFFFFFFF
    result = {"isa": ISA, "program_hex": program.hex().upper(), "memory_bytes": 256,
              "max_steps": max_steps, "steps": steps, "a": a, "bank": bank, "pc": pc,
              "halted": halted, "stop_reason": reason, "class": CLASSES[cls - 1],
              "self_modified": self_modified, "output_count": output_count, "outputs_first8": outputs,
              "memory": memory, "seen_memory_bits": seen, "state_hash": checksum}
    if trace: result["trace"] = history
    return result


def parse_initial(entries):
    memory = {}
    for entry in entries:
        address, value = entry.split("=", 1)
        address, value = int(address, 0), int(value, 0)
        if address in memory:
            raise ValueError("Duplicate initial-memory address")
        memory[address] = value
    return memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program", help="Hex bytes, e.g. '0F 11 5F DF F0'")
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--set", action="append", default=[], metavar="ADDRESS=VALUE", help="Initial data; decimal or 0x-prefixed hex")
    parser.add_argument("--first-output", action="store_true")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    try:
        result = execute(bytes.fromhex(args.program), max_steps=args.steps,
                         initial_memory=parse_initial(args.set), first_output=args.first_output, trace=args.trace)
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
