/* "Which programs add THREE numbers?"  Plain CPU version of pass 1 (independent of the GPU code).
 * usage: nano_search3_cpu L start count threads ca cb cc     prints the same "done ..." line as nano_search3.cu
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

static const uint8_t TRI[32][3] = {
    {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 2, 3}, {3, 2, 1}, {2, 3, 1}, {5, 0, 7}, {255, 1, 0}, {0, 255, 1},
    {1, 0, 255}, {128, 64, 64}, {100, 55, 1}, {7, 250, 3}, {9, 9, 9}, {200, 0, 100}, {13, 31, 77}, {255, 255, 255},
    {64, 32, 16}, {16, 32, 64}, {6, 1, 0}, {0, 6, 1}, {77, 123, 45}, {45, 77, 123}, {2, 0, 0}, {0, 2, 0}, {0, 0, 2},
    {11, 2, 30}, {30, 11, 2}, {2, 30, 11}, {90, 90, 90}, {1, 1, 1}};
static int CA, CB, CC;

static int first_out(uint64_t p, int L, uint8_t a, uint8_t b, uint8_t c)
{
    uint8_t M[16] = {0}, A = 0;
    for (int i = 0; i < L; i++) M[i] = (uint8_t)(p >> (8 * (L - 1 - i)));
    M[CA] = a; M[CB] = b; M[CC] = c;
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
    if (argc < 8) { fprintf(stderr, "usage: nano_search3_cpu L start count threads ca cb cc\n"); return 2; }
    int L = atoi(argv[1]), threads = atoi(argv[4]);
    uint64_t start = strtoull(argv[2], 0, 0), count = strtoull(argv[3], 0, 0);
    CA = atoi(argv[5]); CB = atoi(argv[6]); CC = atoi(argv[7]);
    uint64_t c[5] = {0}, ncand = 0;
    uint64_t *cand = malloc(sizeof(uint64_t) * (1u << 22));
    double t0 = omp_get_wtime();
    #pragma omp parallel for num_threads(threads) schedule(dynamic, 65536) reduction(+:c[:5])
    for (uint64_t i = 0; i < count; i++) {
        uint64_t p = start + i;
        if (first_out(p, L, 2, 2, 2) != 6) continue;
        c[1]++;
        int fadd = 1, fconst = 1;
        for (int t = 0; t < 32 && (fadd | fconst); t++) {
            unsigned a = TRI[t][0], b = TRI[t][1], cc = TRI[t][2];
            int o = first_out(p, L, (uint8_t)a, (uint8_t)b, (uint8_t)cc);
            fadd &= o == (int)((a + b + cc) & 255); fconst &= o == 6;
        }
        if (fadd) {
            c[2]++;
            uint64_t at;
            #pragma omp atomic capture
            at = ncand++;
            if (at < (1u << 22)) cand[at] = p;
        } else if (fconst) c[3]++;
        else c[4]++;
    }
    printf("done L=%d start=%llu count=%llu says=%llu add=%llu const=%llu other=%llu overflow=0 secs=%.3f device=CPU-%d cands=",
           L, (unsigned long long)start, (unsigned long long)count, (unsigned long long)c[1], (unsigned long long)c[2],
           (unsigned long long)c[3], (unsigned long long)c[4], omp_get_wtime() - t0, threads);
    for (uint64_t i = 0; i < ncand && i < (1u << 22); i++) printf("%s%llx", i ? "," : "", (unsigned long long)cand[i]);
    printf("\n");
    return 0;
}
