"""Dimension42 build: turns the hex machine-code sources into a bootable disk image.

No assembler, no compiler: every byte in the image is written by hand in src/*.hex.
This script only converts hex text to bytes and checks the =XXXX offset markers.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SECTOR = 512
KERNEL_SECTORS = 32          # must match the disk address packet in boot.hex
IMAGE_SIZE = 1024 * 1024     # 1 MiB disk image


def load_hex(path):
    out = bytearray()
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        where = f"{path.name}:{lineno}"
        if line[0] in "=>":
            target = int(line[1:], 16)
            if line[0] == "=" and len(out) != target:
                sys.exit(f"{where}: offset is {len(out):04X}, expected {target:04X}")
            if line[0] == ">":
                if len(out) > target:
                    sys.exit(f"{where}: already at {len(out):04X}, cannot pad to {target:04X}")
                out += bytes(target - len(out))
            continue
        for tok in line.split():
            if len(tok) != 2:
                sys.exit(f"{where}: bad byte '{tok}'")
            out.append(int(tok, 16))
    return bytes(out)


def main():
    boot = load_hex(ROOT / "src" / "boot.hex")
    kernel = load_hex(ROOT / "src" / "kernel.hex")
    if len(boot) != SECTOR or boot[-2:] != b"\x55\xAA":
        sys.exit("boot sector must be exactly 512 bytes ending in 55 AA")
    if len(kernel) > KERNEL_SECTORS * SECTOR:
        sys.exit(f"kernel is {len(kernel)} bytes, max {KERNEL_SECTORS * SECTOR}")

    image = bytearray(boot + kernel)
    image += bytes(IMAGE_SIZE - len(image))
    out = ROOT / "build" / "dimension42.img"
    out.parent.mkdir(exist_ok=True)
    out.write_bytes(image)
    print(f"boot   {len(boot):5d} bytes")
    print(f"kernel {len(kernel):5d} bytes")
    print(f"image  {out}  ({len(image)} bytes)")


if __name__ == "__main__":
    main()
