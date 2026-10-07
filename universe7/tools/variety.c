/* Universe 7 variety counter (host-side observer, not part of the universe).
 *
 * Runs every program of 1..Lmax bits on the Universe 7 machine and counts
 * how many distinct final screens they produce. The machine here must match
 * src/u7.hex exactly; tools/ref.py checks the built binary against it.
 *
 *   gcc -O2 -fopenmp tools/variety.c -o build/variety
 *   ./build/variety 22            # all programs up to 22 bits (a few minutes)
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static const int PEN[4]    = {0, 0, 1, 2};   /* 0 flip, 1 set, 2 clear */
static const int STRIDE[4] = {4, 8, 3, 3};
#define STEPS 127

static uint64_t run(uint64_t p, int L) {
    uint64_t scr = 0; int h = 0, pc = 0;
    for (int s = 0; s < STEPS; s++) {
        int b1 = (p >> (L - 1 - pc)) & 1; pc = (pc + 1) % L;
        int b0 = (p >> (L - 1 - pc)) & 1; pc = (pc + 1) % L;
        int op = b1 * 2 + b0;
        if (PEN[op] == 0) scr ^= 1ULL << h;
        else if (PEN[op] == 1) scr |= 1ULL << h;
        else scr &= ~(1ULL << h);
        h = (h + STRIDE[op]) & 63;
    }
    return scr;
}
static int cmp(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }
int main(int argc, char **argv) {
    int Lmax = argc > 1 ? atoi(argv[1]) : 16;
    long tot = 0, totP = 0;
    printf("bits   programs       distinct screens   share\n");
    for (int L = 1; L <= Lmax; L++) {
        long n = 1L << L; uint64_t *S = malloc(n * sizeof *S);
        #pragma omp parallel for schedule(static)
        for (long p = 0; p < n; p++) S[p] = run((uint64_t)p, L);
        qsort(S, n, sizeof *S, cmp);
        long d = 1; for (long i = 1; i < n; i++) if (S[i] != S[i - 1]) d++;
        free(S); tot += d; totP += n;
        printf("%4d %12ld %18ld %8.2f%%\n", L, n, d, 100.0 * d / n);
    }
    printf("1..%d %10ld %18ld %8.2f%%\n", Lmax, totP, tot, 100.0 * tot / totP);
    return 0;
}
