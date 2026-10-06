/* NANO explorer, CPU version (C + OpenMP).
 *
 * Runs every NANO program of L bytes in [start, start+count): 64 steps each,
 * exactly the semantics of the Dimension42 kernel (src/kernel.hex), generalised
 * to programs of L bytes (PC wraps at L, JMP/JZ go to n mod L).
 *
 * usage: nano_cpu L start count threads [--hashes | --nooutput]
 *   stdout: one class byte per program (default), or one uint32 hash per
 *           program (--hashes), or nothing (--nooutput)
 *   stderr: "programs=N hashsum=S counts=c1,..,c6 secs=T device=..."
 *
 * Program p is loaded big-endian: M[0] = first byte = top byte of p.
 * Classes: 6 DRAW, 5 OUTPUT, 4 SELF-MOD, 3 ACTIVE, 2 HALT, 1 IDLE (see kernel).
 * hash = FNV-1a over A, steps, outcount, halted, traceA, M[16], trace[16], outputs[8].
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

static inline uint32_t fnv(uint32_t h, uint8_t b) { return (h ^ b) * 16777619u; }

static uint8_t run(uint64_t p, int L, uint32_t *hash_out)
{
    uint8_t M[16] = {0}, T[16] = {0}, O[8] = {0};
    for (int i = 0; i < L; i++) M[i] = (uint8_t)(p >> (8 * (L - 1 - i)));
    uint8_t A = 0, tA = 0;
    int pc = 0, steps = 0, outc = 0, halted = 0;

    for (;;) {
        for (int i = 0; i < 16; i++) T[i] |= M[i];
        tA |= A;
        if (steps >= 64) break;
        steps++;
        uint8_t ins = M[pc];
        if (++pc >= L) pc = 0;
        unsigned op = ins >> 4, n = ins & 15;
        switch (op) {
        case 0x0: break;
        case 0x1: A = (uint8_t)n; break;
        case 0x2: A = (uint8_t)(A + n); break;
        case 0x3: A = (uint8_t)(A - n); break;
        case 0x4: A = M[n]; break;
        case 0x5: M[n] = A; break;
        case 0x6: A = (uint8_t)(A + M[n]); break;
        case 0x7: pc = (int)(n % L); break;
        case 0x8: if (A == 0) pc = (int)(n % L); break;
        case 0x9: M[n]++; break;
        case 0xA: M[n]--; break;
        case 0xB: { unsigned k = n & 7; A = (uint8_t)((A << k) | (A >> (8 - k))); } break;
        case 0xC: A ^= M[n]; break;
        case 0xD: if (outc < 8) O[outc] = M[n]; outc++; break;
        case 0xE: M[n] = (uint8_t)~M[n]; break;
        case 0xF: halted = 1; goto done;
        }
    }
done:;
    uint8_t cls = 1;
    int draw = 0, selfmod = 0, active = tA != 0;
    for (int i = 8; i < 16; i++) draw |= T[i];
    for (int i = 0; i < L; i++) selfmod |= M[i] != (uint8_t)(p >> (8 * (L - 1 - i)));
    for (int i = L; i < 8; i++) active |= T[i] != 0;
    if (draw) cls = 6;
    else if (outc) cls = 5;
    else if (selfmod) cls = 4;
    else if (active) cls = 3;
    else if (halted) cls = 2;

    uint32_t h = 2166136261u;
    h = fnv(h, A); h = fnv(h, (uint8_t)steps); h = fnv(h, (uint8_t)outc);
    h = fnv(h, (uint8_t)halted); h = fnv(h, tA);
    for (int i = 0; i < 16; i++) h = fnv(h, M[i]);
    for (int i = 0; i < 16; i++) h = fnv(h, T[i]);
    for (int i = 0; i < 8; i++) h = fnv(h, O[i]);
    *hash_out = h;
    return cls;
}

int main(int argc, char **argv)
{
    if (argc < 5) { fprintf(stderr, "usage: nano_cpu L start count threads [--hashes|--nooutput]\n"); return 2; }
    int L = atoi(argv[1]);
    uint64_t start = strtoull(argv[2], 0, 0), count = strtoull(argv[3], 0, 0);
    int threads = atoi(argv[4]);
    int mode = argc > 5 ? (strcmp(argv[5], "--hashes") == 0 ? 1 : 2) : 0;

    uint8_t *cls = malloc(mode == 2 ? 1 : count);
    uint32_t *hs = mode == 1 ? malloc(count * 4) : NULL;
    uint64_t hashsum = 0, counts[7] = {0};

    double t0 = omp_get_wtime();
    #pragma omp parallel num_threads(threads)
    {
        uint64_t lc[7] = {0}, lh = 0;
        #pragma omp for schedule(static)
        for (uint64_t i = 0; i < count; i++) {
            uint32_t h;
            uint8_t c = run(start + i, L, &h);
            lh += h;
            lc[c]++;
            if (mode == 0) cls[i] = c;
            else if (mode == 1) hs[i] = h;
        }
        #pragma omp critical
        { hashsum += lh; for (int k = 0; k < 7; k++) counts[k] += lc[k]; }
    }
    double secs = omp_get_wtime() - t0;

    if (mode == 0) fwrite(cls, 1, count, stdout);
    if (mode == 1) fwrite(hs, 4, count, stdout);
    fflush(stdout);
    fprintf(stderr, "programs=%llu hashsum=%llu counts=%llu,%llu,%llu,%llu,%llu,%llu secs=%.6f device=CPU-%dthreads\n",
            (unsigned long long)count, (unsigned long long)hashsum,
            (unsigned long long)counts[1], (unsigned long long)counts[2], (unsigned long long)counts[3],
            (unsigned long long)counts[4], (unsigned long long)counts[5], (unsigned long long)counts[6],
            secs, threads);
    return 0;
}
