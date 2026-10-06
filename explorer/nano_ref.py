"""Slow, plain reference model of the NANO CPU (L-byte programs, 64 steps).

Used only to check the fast C / CUDA / Vulkan versions on samples.
Returns (class, hash) with the same definitions as nano_cpu.c.
"""


def fnv(h, b):
    return ((h ^ (b & 255)) * 16777619) & 0xFFFFFFFF


def first_out(p, L, a, b):
    """First value output with OUT when a is in M5 and b in M6 (None if nothing within 64 steps)."""
    M = [(p >> (8 * (L - 1 - i))) & 255 for i in range(L)] + [0] * (16 - L)
    M[5], M[6] = a, b
    A = pc = 0
    for _ in range(64):
        ins = M[pc]
        pc = (pc + 1) % L
        op, n = ins >> 4, ins & 15
        if op == 1: A = n
        elif op == 2: A = (A + n) & 255
        elif op == 3: A = (A - n) & 255
        elif op == 4: A = M[n]
        elif op == 5: M[n] = A
        elif op == 6: A = (A + M[n]) & 255
        elif op == 7: pc = n % L
        elif op == 8:
            if A == 0: pc = n % L
        elif op == 9: M[n] = (M[n] + 1) & 255
        elif op == 10: M[n] = (M[n] - 1) & 255
        elif op == 11:
            k = n & 7
            A = ((A << k) | (A >> (8 - k))) & 255
        elif op == 12: A ^= M[n]
        elif op == 13: return M[n]
        elif op == 14: M[n] ^= 255
        elif op == 15: return None
    return None


def run(p, L):
    prog = [(p >> (8 * (L - 1 - i))) & 255 for i in range(L)]
    M = prog + [0] * (16 - L)
    T = [0] * 16
    O = [0] * 8
    A = tA = pc = steps = outc = halted = 0
    while True:
        T = [t | m for t, m in zip(T, M)]
        tA |= A
        if steps >= 64:
            break
        steps += 1
        ins = M[pc]
        pc = (pc + 1) % L
        op, n = ins >> 4, ins & 15
        if op == 1: A = n
        elif op == 2: A = (A + n) & 255
        elif op == 3: A = (A - n) & 255
        elif op == 4: A = M[n]
        elif op == 5: M[n] = A
        elif op == 6: A = (A + M[n]) & 255
        elif op == 7: pc = n % L
        elif op == 8:
            if A == 0: pc = n % L
        elif op == 9: M[n] = (M[n] + 1) & 255
        elif op == 10: M[n] = (M[n] - 1) & 255
        elif op == 11:
            k = n & 7
            A = ((A << k) | (A >> (8 - k))) & 255
        elif op == 12: A ^= M[n]
        elif op == 13:
            if outc < 8: O[outc] = M[n]
            outc += 1
        elif op == 14: M[n] ^= 255
        elif op == 15:
            halted = 1
            break
    if any(T[8:]): cls = 6
    elif outc: cls = 5
    elif M[:L] != prog: cls = 4
    elif tA or any(T[L:8]): cls = 3
    elif halted: cls = 2
    else: cls = 1
    h = 2166136261
    for b in [A, steps, outc, halted, tA] + M + T + O:
        h = fnv(h, b)
    return cls, h
