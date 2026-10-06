/* "Which programs answer 2 + 2 = 4?"  Plain CPU version of pass 1 (independent of the GPU code).
 * usage: nano_search_cpu L start count threads [ca cb]     prints the same "done ..." line as nano_search.cu
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

static const uint8_t PAIRS[32][2] = {
    {0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 3}, {3, 2}, {3, 5}, {5, 3}, {1, 255}, {255, 1}, {128, 128}, {100, 55},
    {7, 250}, {250, 7}, {17, 4}, {4, 17}, {9, 9}, {0, 200}, {200, 0}, {13, 31}, {31, 13}, {255, 255}, {64, 32},
    {32, 64}, {6, 1}, {1, 6}, {77, 123}, {123, 77}, {2, 0}, {0, 2}, {11, 2}, {2, 11}};

/* first output of program p (L bytes) with a in M5, b in M6; -1 if it outputs nothing in 64 steps */
static int CA = 5, CB = 6;
static int first_out(uint64_t p, int L, uint8_t a, uint8_t b)
{
    uint8_t M[16] = {0}, A = 0;
    for (int i = 0; i < L; i++) M[i] = (uint8_t)(p >> (8 * (L - 1 - i)));
    M[CA] = a; M[CB] = b;
    int pc = 0;
    for (int steps = 0; steps < 64; steps++) {
        uint8_t ins = M[pc];
        if (++pc >= L) pc = 0;
        unsigned op = ins >> 4, n = ins & 15;
        switch (op) {
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
        case 0xD: return M[n];
        case 0xE: M[n] = (uint8_t)~M[n]; break;
        case 0xF: return -1;
        }
    }
    return -1;
}

int main(int argc, char **argv)
{
    if (argc < 5) { fprintf(stderr, "usage: nano_search_cpu L start count threads [ca cb]\n"); return 2; }
    int L = atoi(argv[1]), threads = atoi(argv[4]);
    if (argc > 6) { CA = atoi(argv[5]); CB = atoi(argv[6]); }
    uint64_t start = strtoull(argv[2], 0, 0), count = strtoull(argv[3], 0, 0);
    uint64_t c[8] = {0};
    uint64_t *cand = malloc(sizeof(uint64_t) * (1u << 24));
    uint64_t ncand = 0;
    double t0 = omp_get_wtime();
    #pragma omp parallel for num_threads(threads) schedule(dynamic, 65536) reduction(+:c[:8])
    for (uint64_t i = 0; i < count; i++) {
        uint64_t p = start + i;
        if (first_out(p, L, 2, 2) != 4) continue;
        c[1]++;
        int fadd = 1, fconst = 1, fmul = 1, f2a = 1, f2b = 1;
        for (int t = 0; t < 32 && (fadd | fconst | fmul | f2a | f2b); t++) {
            unsigned a = PAIRS[t][0], b = PAIRS[t][1];
            int o = first_out(p, L, (uint8_t)a, (uint8_t)b);
            fadd &= o == (int)((a + b) & 255); fconst &= o == 4; fmul &= o == (int)((a * b) & 255);
            f2a &= o == (int)((2 * a) & 255); f2b &= o == (int)((2 * b) & 255);
        }
        if (fadd) {
            c[2]++;
            uint64_t at;
            #pragma omp atomic capture
            at = ncand++;
            if (at < (1u << 24)) cand[at] = p;
        } else if (fconst) c[3]++;
        else if (fmul) c[4]++;
        else if (f2a) c[5]++;
        else if (f2b) c[6]++;
        else c[7]++;
    }
    printf("done L=%d start=%llu count=%llu says4=%llu add=%llu const=%llu mul=%llu twoa=%llu twob=%llu other=%llu overflow=0 secs=%.3f device=CPU-%d cands=",
           L, (unsigned long long)start, (unsigned long long)count, (unsigned long long)c[1], (unsigned long long)c[2],
           (unsigned long long)c[3], (unsigned long long)c[4], (unsigned long long)c[5], (unsigned long long)c[6],
           (unsigned long long)c[7], omp_get_wtime() - t0, threads);
    for (uint64_t i = 0; i < ncand && i < (1u << 24); i++) printf("%s%llx", i ? "," : "", (unsigned long long)cand[i]);
    printf("\n");
    return 0;
}
