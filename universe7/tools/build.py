"""Universe 7 build: turns src/u7.hex into the executable build/u7.

No assembler, no compiler. This script only converts hex text to bytes
and checks the =XXXX offset markers.
"""
import os, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_hex(path):
    out = bytearray()
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        where = f"{path.name}:{lineno}"
        if line[0] == "=":
            target = int(line[1:], 16)
            if len(out) != target:
                sys.exit(f"{where}: offset is {len(out):04X}, expected {target:04X}")
            continue
        for tok in line.split():
            if len(tok) != 2:
                sys.exit(f"{where}: bad byte '{tok}'")
            out.append(int(tok, 16))
    return bytes(out)


def main():
    code = load_hex(ROOT / "src" / "u7.hex")
    out = ROOT / "build" / "u7"
    out.parent.mkdir(exist_ok=True)
    out.write_bytes(code)
    os.chmod(out, 0o755)
    print(f"u7  {len(code)} bytes  ->  {out}")


if __name__ == "__main__":
    main()
