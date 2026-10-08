// Artificial-life toolkit search over all 5-byte NANO programs (no inputs, 64 steps). One pass, flags per program:
//   REP      exact copy of its 5 bytes appears at M[k..k+4], k >= 5, at some step (and >= 2 non-zero bytes)
//   MUTREP   no exact copy ever, but at the end a copy with exactly one differing byte stands at some k >= 5,
//            and that differing byte is non-zero (it was written, not just missing)
//   HEALER   writes into its own code during the run (changing it) and the code is exactly original at the end
//   WALKER   at least 4 times changes only the operand (low nibble) of one of its own instructions
//   STABLE   never halts in 64 steps, does work (A non-zero at some step or M5..MF written), code intact at end
//   REPAIR   bitmask of byte positions j it repairs: for 3 damaged values of byte j set BEFORE the start,
//            the code is exactly original at the end every time (only positions the program writes are tested)
// Stored: all REP / MUTREP and repairers of >= 2 positions; a 1-in-4,194,304 hash sample of everything else.
//
// usage: nano_alife serve gpu     stdin "start count"     stdout "done start=.. <counts> cands=" list prog-flags
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

typedef unsigned long long ull;
#define L 5
enum { F_REP = 1, F_MUTREP = 2, F_HEALER = 4, F_WALKER = 8, F_STABLE = 16 };   // repair mask in bits 8..12

__device__ __forceinline__ unsigned getb(const unsigned *m, unsigned n) { return (m[n >> 2] >> ((n & 3) * 8)) & 255u; }
__device__ __forceinline__ void setb(unsigned *m, unsigned n, unsigned v)
{
    m[n >> 2] = (m[n >> 2] & ~(255u << ((n & 3) * 8))) | ((v & 255u) << ((n & 3) * 8));
}

// mismatches between M[k..k+4] and the original bytes (stops counting at 2)
__device__ __forceinline__ int diff_at(const unsigned *m, const unsigned char *p0, int k, int *where)
{
    int d = 0;
    for (int i = 0; i < L && d < 2; i++)
        if (getb(m, k + i) != p0[i]) { d++; *where = k + i; }
    return d;
}

struct Run { unsigned flags, wrote_own; };

// run from initial memory m (16 bytes); p0 = the ORIGINAL program (for copy / intact checks)
__device__ Run run(unsigned *m, const unsigned char *p0, bool full)
{
    unsigned A = 0, pc = 0, wrote_own = 0, operand_edits = 0;
    bool halted = false, work = false, rep = false, changed_own = false;
    for (int steps = 0; steps < 64; steps++) {
        unsigned ins = getb(m, pc);
        pc = pc + 1 >= L ? 0 : pc + 1;
        unsigned op = ins >> 4, n = ins & 15;
        if (op == 15) { halted = true; break; }
        unsigned val = getb(m, n), k3 = n & 7u, nA = A;
        switch (op) {
        case 1: nA = n; break;
        case 2: nA = A + n; break;
        case 3: nA = A - n; break;
        case 4: nA = val; break;
        case 6: nA = A + val; break;
        case 11: nA = (A << k3) | (A >> (8 - k3)); break;
        case 12: nA = A ^ val; break;
        }
        if (op == 5 || op == 9 || op == 10 || op == 14) {
            unsigned wr = (op == 5 ? A : op == 9 ? val + 1 : op == 10 ? val - 1 : ~val) & 255u;
            if (n < L) {
                wrote_own |= 1u << n;
                if (wr != val) changed_own = true;
                if ((wr >> 4) == (val >> 4) && wr != val) operand_edits++;
            } else work = true;
            setb(m, n, wr);
            if (full && !rep && n >= L) {
                int w;
                for (int k = L; k + L <= 16 && !rep; k++) rep = diff_at(m, p0, k, &w) == 0;
            }
        }
        if (op == 7 || (op == 8 && A == 0)) pc = n % L;
        A = nA & 255u;
        if (A) work = true;
    }
    bool intact = true;
    for (int i = 0; i < L; i++) intact &= getb(m, i) == p0[i];
    unsigned f = 0;
    if (full) {
        if (rep) f |= F_REP;
        else {
            for (int k = L; k + L <= 16; k++) {
                int w = -1;
                if (diff_at(m, p0, k, &w) == 1 && getb(m, w) != 0) { f |= F_MUTREP; break; }
            }
        }
        if (changed_own && intact) f |= F_HEALER;
        if (operand_edits >= 4) f |= F_WALKER;
        if (!halted && work && intact) f |= F_STABLE;
    } else if (intact) f = 1;                         // damage run: 1 = code exactly restored
    Run r = {f, wrote_own};
    return r;
}

__device__ __forceinline__ ull mix(ull x) { x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; return x; }

// counters: 0 rep 1 mutrep 2 healer 3 walker 4 stable 5 repair_any 6 repair_multi 7 overflow
__global__ void search(ull start, ull count, ull *out, unsigned cap, unsigned *nout, ull *cnt)
{
    unsigned c[7] = {0, 0, 0, 0, 0, 0, 0};
    for (ull i = blockIdx.x * (ull)blockDim.x + threadIdx.x; i < count; i += (ull)gridDim.x * blockDim.x) {
        ull p = start + i;
        unsigned char p0[L];
        int nonzero = 0;
        unsigned m[4] = {0, 0, 0, 0};
        for (int b = 0; b < L; b++) {
            p0[b] = (unsigned char)(p >> (8 * (L - 1 - b)));
            nonzero += p0[b] != 0;
            setb(m, b, p0[b]);
        }
        Run r = run(m, p0, true);
        unsigned flags = nonzero >= 2 ? r.flags : (r.flags & ~(F_REP | F_MUTREP));
        // repair test: only positions the program writes itself
        unsigned repair = 0;
        for (int j = 0; j < L; j++) {
            if (!(r.wrote_own & (1u << j))) continue;
            const unsigned char dmg[3] = {(unsigned char)(p0[j] ^ 0x01), (unsigned char)(p0[j] ^ 0x80), (unsigned char)(p0[j] ^ 0xFF)};
            bool ok = true;
            for (int d = 0; d < 3 && ok; d++) {
                unsigned mm[4] = {0, 0, 0, 0};
                for (int b = 0; b < L; b++) setb(mm, b, p0[b]);
                setb(mm, j, dmg[d]);
                ok = run(mm, p0, false).flags == 1;
            }
            if (ok) repair |= 1u << j;
        }
        flags |= repair << 8;
        c[0] += !!(flags & F_REP); c[1] += !!(flags & F_MUTREP); c[2] += !!(flags & F_HEALER);
        c[3] += !!(flags & F_WALKER); c[4] += !!(flags & F_STABLE); c[5] += repair != 0; c[6] += __popc(repair) >= 2;
        bool keep = (flags & (F_REP | F_MUTREP)) || __popc(repair) >= 2 || (flags && (mix(p) & 0x3FFFFF) == 0);
        if (keep) {
            unsigned idx = atomicAdd(nout, 1u);
            if (idx < cap) { out[2 * idx] = p; out[2 * idx + 1] = flags; } else atomicAdd(&cnt[7], 1ull);
        }
    }
    for (int k = 0; k < 7; k++) atomicAdd(&cnt[k], (ull)c[k]);
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: nano_alife serve gpu\n"); return 2; }
    int gpu = atoi(argv[2]);
    cudaSetDevice(gpu);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpu);
    const unsigned CAP = 1u << 22;
    ull *d_out, *d_cnt;
    unsigned *d_n;
    cudaMalloc(&d_out, 2ull * CAP * 8); cudaMalloc(&d_cnt, 8 * 8); cudaMalloc(&d_n, 4);
    std::vector<ull> out(2ull * CAP);
    char line[256];
    ull start, count;
    while (fgets(line, sizeof line, stdin)) {
        if (sscanf(line, "%llu %llu", &start, &count) != 2) continue;
        auto t0 = std::chrono::steady_clock::now();
        cudaMemset(d_cnt, 0, 64); cudaMemset(d_n, 0, 4);
        const ull SUB = 1ull << 28;
        for (ull off = 0; off < count; off += SUB) {
            ull n = count - off < SUB ? count - off : SUB;
            search<<<prop.multiProcessorCount * 16, 256>>>(start + off, n, d_out, CAP, d_n, d_cnt);
        }
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) { printf("ERROR %s\n", cudaGetErrorString(err)); fflush(stdout); return 1; }
        ull cnt[8];
        unsigned nn;
        cudaMemcpy(cnt, d_cnt, 64, cudaMemcpyDeviceToHost);
        cudaMemcpy(&nn, d_n, 4, cudaMemcpyDeviceToHost);
        unsigned stored = nn < CAP ? nn : CAP;
        cudaMemcpy(out.data(), d_out, 2ull * stored * 8, cudaMemcpyDeviceToHost);
        printf("done start=%llu rep=%llu mutrep=%llu healer=%llu walker=%llu stable=%llu repair=%llu repair2=%llu overflow=%llu secs=%.2f cands=",
               start, cnt[0], cnt[1], cnt[2], cnt[3], cnt[4], cnt[5], cnt[6], cnt[7],
               std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        for (unsigned i = 0; i < stored; i++) printf("%s%llx-%llx", i ? "," : "", out[2 * i], out[2 * i + 1]);
        printf("\n");
        fflush(stdout);
    }
    return 0;
}
