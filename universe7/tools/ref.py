"""Universe 7 reference model and checker (host-side observer, not part of the universe).

    python tools/ref.py 0110        # print the screen of one program
    python tools/ref.py --check 12  # compare build/u7 with this model on all programs up to 12 bits
"""
import pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TABLE = [("flip", 4), ("flip", 8), ("set", 3), ("clear", 3)]   # ops 00 01 10 11
STEPS = 127


def run(bits, steps=STEPS):
    L = len(bits); scr = 0; head = 0; pc = 0
    for _ in range(steps):
        b1 = int(bits[pc]); pc = (pc + 1) % L
        b0 = int(bits[pc]); pc = (pc + 1) % L
        pen, stride = TABLE[b1 * 2 + b0]
        m = 1 << head
        if pen == "flip": scr ^= m
        elif pen == "set": scr |= m
        else: scr &= ~m
        head = (head + stride) & 63
    return scr


def picture(scr):
    return "".join(("#" if scr >> i & 1 else ".") + ("\n" if i % 8 == 7 else "") for i in range(64))


def check(lmax):
    exe = ROOT / "build" / "u7"; n = bad = 0
    for L in range(1, lmax + 1):
        for p in range(1 << L):
            bits = format(p, f"0{L}b"); n += 1
            out = subprocess.run([str(exe), bits], capture_output=True, text=True).stdout
            if out != picture(run(bits)):
                bad += 1; print("MISMATCH", bits)
    print(f"checked {n} programs, {bad} mismatches")
    return bad == 0


if __name__ == "__main__":
    if sys.argv[1] == "--check":
        sys.exit(0 if check(int(sys.argv[2])) else 1)
    print(picture(run(sys.argv[1])), end="")
