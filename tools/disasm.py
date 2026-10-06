"""Disassemble the built image so the hand-written bytes can be checked.

usage: python tools/disasm.py boot|kernel [start_hex end_hex]
Needs: pip install capstone
"""
import pathlib
import sys

from capstone import CS_ARCH_X86, CS_MODE_16, Cs

ROOT = pathlib.Path(__file__).resolve().parent.parent
image = (ROOT / "build" / "dimension42.img").read_bytes()

part = sys.argv[1] if len(sys.argv) > 1 else "kernel"
base, data = (0x7C00, image[:512]) if part == "boot" else (0x8000, image[512:512 + 0x1080])
start = int(sys.argv[2], 16) if len(sys.argv) > 2 else 0
end = int(sys.argv[3], 16) if len(sys.argv) > 3 else len(data)

md = Cs(CS_ARCH_X86, CS_MODE_16)
for ins in md.disasm(data[start:end], base + start):
    print(f"{ins.address - base:04X}  {ins.bytes.hex(' ').upper():24s} {ins.mnemonic} {ins.op_str}")
