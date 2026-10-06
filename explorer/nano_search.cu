// "Which programs answer 2 + 2 = 4?"  CUDA search over NANO programs.
//
// Input:  a in M[ca], b in M[cb] (default M5, M6) before the program starts (everything else as in nano_cpu.c).
// Answer: the first value the program outputs with OUT (no OUT within 64 steps = no answer).
//
// Pass 1 (search): run every program with a=2, b=2. If the answer is 4, run it on 32 more
// input pairs and decide what it really computes:
//   ADD a+b | CONST always 4 | MUL a*b | TWOA 2a | TWOB 2b | OTHER (4 for 2+2 by coincidence)
// ADD candidates are listed.
// Pass 2 (verify): every ADD candidate is run on all 65,536 pairs (a, b = 0..255) and must
// answer (a+b) mod 256 every time.
//
// usage: nano_search serve gpu            stdin: "L start count"   stdout: one result line per range
//        nano_search verify gpu           stdin: "L program" lines stdout: one line per program
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>

typedef unsigned long long ull;

__constant__ unsigned char PAIRS[32][2] = {
    {0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 3}, {3, 2}, {3, 5}, {5, 3}, {1, 255}, {255, 1}, {128, 128}, {100, 55},
    {7, 250}, {250, 7}, {17, 4}, {4, 17}, {9, 9}, {0, 200}, {200, 0}, {13, 31}, {31, 13}, {255, 255}, {64, 32},
    {32, 64}, {6, 1}, {1, 6}, {77, 123}, {123, 77}, {2, 0}, {0, 2}, {11, 2}, {2, 11}};

__device__ __forceinline__ unsigned getb(unsigned m0, unsigned m1, unsigned m2, unsigned m3, unsigned n)
{
    unsigned w = n < 4 ? m0 : n < 8 ? m1 : n < 12 ? m2 : m3;
    return (w >> ((n & 3) * 8)) & 255u;
}

// Runs program p (L bytes) with inputs a, b. Returns the first output, or -1 if none.
// full=false stops at the first output; full=true runs to the end and fills outc/halted/steps.
__device__ int run_io(ull p, int L, unsigned a, unsigned b, bool full, unsigned ca, unsigned cb,
                      unsigned *outc_r = nullptr, unsigned *halted_r = nullptr, unsigned *steps_r = nullptr)
{
    unsigned prog0 = 0, prog1 = 0;
    for (int i = 0; i < L; i++) {
        unsigned v = (unsigned)(p >> (8 * (L - 1 - i))) & 255u;
        if (i < 4) prog0 |= v << (8 * i); else prog1 |= v << (8 * (i - 4));
    }
    unsigned m[4] = {prog0, prog1, 0, 0};
    m[ca >> 2] = (m[ca >> 2] & ~(255u << ((ca & 3) * 8))) | (a << ((ca & 3) * 8));   // input a -> M[ca]
    m[cb >> 2] = (m[cb >> 2] & ~(255u << ((cb & 3) * 8))) | (b << ((cb & 3) * 8));   // input b -> M[cb]
    unsigned m0 = m[0], m1 = m[1], m2 = m[2], m3 = m[3];
    unsigned A = 0, pc = 0, steps = 0, outc = 0, halted = 0;
    int first = -1;
    while (steps < 64) {
        steps++;
        unsigned ins = ((pc < 4 ? m0 : m1) >> ((pc & 3) * 8)) & 255u;
        pc = pc + 1 >= (unsigned)L ? 0 : pc + 1;
        unsigned op = ins >> 4, n = ins & 15u;
        if (op == 15) { halted = 1; break; }
        unsigned val = getb(m0, m1, m2, m3, n);
        if (op == 13) {
            if (outc == 0) { first = (int)val; if (!full) return first; }
            outc++;
            continue;
        }
        unsigned k = n & 7u;
        unsigned rol = ((A << k) | (A >> (8 - k))) & 255u;
        unsigned nA = A;
        nA = op == 1 ? n : nA;
        nA = op == 2 ? A + n : nA;
        nA = op == 3 ? A - n : nA;
        nA = op == 4 ? val : nA;
        nA = op == 6 ? A + val : nA;
        nA = op == 11 ? rol : nA;
        nA = op == 12 ? (A ^ val) : nA;
        unsigned wr = op == 5 ? A : op == 9 ? val + 1 : op == 10 ? val - 1 : ~val;
        bool dow = op == 5 || op == 9 || op == 10 || op == 14;
        unsigned sh = (n & 3) * 8, keep = ~(255u << sh), put = (wr & 255u) << sh, wi = n >> 2;
        m0 = dow && wi == 0 ? (m0 & keep) | put : m0;
        m1 = dow && wi == 1 ? (m1 & keep) | put : m1;
        m2 = dow && wi == 2 ? (m2 & keep) | put : m2;
        m3 = dow && wi == 3 ? (m3 & keep) | put : m3;
        pc = (op == 7 || (op == 8 && A == 0)) ? n % (unsigned)L : pc;
        A = nA & 255u;
    }
    if (outc_r) { *outc_r = outc; *halted_r = halted; *steps_r = steps; }
    return first;
}

// counters: 0 programs, 1 says4, 2 ADD, 3 CONST, 4 MUL, 5 TWOA, 6 TWOB, 7 OTHER, 8 candidate overflow
__global__ void search(int L, ull start, ull count, ull *cand, unsigned cap, unsigned *ncand, ull *cnt, unsigned ca, unsigned cb)
{
    unsigned c_says = 0, c_add = 0, c_const = 0, c_mul = 0, c_2a = 0, c_2b = 0, c_other = 0;
    for (ull i = blockIdx.x * (ull)blockDim.x + threadIdx.x; i < count; i += (ull)gridDim.x * blockDim.x) {
        ull p = start + i;
        if (run_io(p, L, 2, 2, false, ca, cb) != 4) continue;
        c_says++;
        bool fadd = true, fconst = true, fmul = true, f2a = true, f2b = true;
        for (int t = 0; t < 32 && (fadd || fconst || fmul || f2a || f2b); t++) {
            unsigned a = PAIRS[t][0], b = PAIRS[t][1];
            int o = run_io(p, L, a, b, false, ca, cb);
            fadd &= o == (int)((a + b) & 255u);
            fconst &= o == 4;
            fmul &= o == (int)((a * b) & 255u);
            f2a &= o == (int)((2 * a) & 255u);
            f2b &= o == (int)((2 * b) & 255u);
        }
        if (fadd) {
            c_add++;
            unsigned idx = atomicAdd(ncand, 1u);
            if (idx < cap) cand[idx] = p; else atomicAdd(&cnt[8], 1ull);
        } else if (fconst) c_const++;
        else if (fmul) c_mul++;
        else if (f2a) c_2a++;
        else if (f2b) c_2b++;
        else c_other++;
    }
    atomicAdd(&cnt[1], (ull)c_says); atomicAdd(&cnt[2], (ull)c_add); atomicAdd(&cnt[3], (ull)c_const);
    atomicAdd(&cnt[4], (ull)c_mul); atomicAdd(&cnt[5], (ull)c_2a); atomicAdd(&cnt[6], (ull)c_2b);
    atomicAdd(&cnt[7], (ull)c_other);
}

// one thread per (candidate, a): checks all 256 values of b
__global__ void verify(const ull *progs, const int *lens, const int *cells, int n, unsigned *fails, unsigned *first_fail)
{
    unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (unsigned)n * 256) return;
    int c = t / 256;
    unsigned a = t % 256;
    for (unsigned b = 0; b < 256; b++) {
        if (run_io(progs[c], lens[c], a, b, false, cells[2 * c], cells[2 * c + 1]) != (int)((a + b) & 255u)) {
            atomicAdd(&fails[c], 1u);
            atomicMin(&first_fail[c], a * 256 + b);
        }
    }
}

__global__ void detail(const ull *progs, const int *lens, const int *cells, int n, unsigned *info)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    unsigned outc, halted, steps;
    run_io(progs[c], lens[c], 2, 2, true, cells[2 * c], cells[2 * c + 1], &outc, &halted, &steps);
    info[c * 3] = outc; info[c * 3 + 1] = halted; info[c * 3 + 2] = steps;
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: nano_search serve|verify gpu\n"); return 2; }
    cudaSetDevice(atoi(argv[2]));
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, atoi(argv[2]));

    if (strcmp(argv[1], "serve") == 0) {
        const unsigned CAP = 1u << 24;
        ull *d_cand, *d_cnt;
        unsigned *d_ncand;
        cudaMalloc(&d_cand, CAP * sizeof(ull));
        cudaMalloc(&d_cnt, 9 * sizeof(ull));
        cudaMalloc(&d_ncand, 4);
        std::vector<ull> cand(CAP);
        int L;
        ull start, count;
        char line[256];
        while (fgets(line, sizeof line, stdin)) {
            unsigned ca = 5, cb = 6;
            if (sscanf(line, "%d %llu %llu %u %u", &L, &start, &count, &ca, &cb) < 3) continue;
            auto t0 = std::chrono::steady_clock::now();
            cudaMemset(d_cnt, 0, 9 * sizeof(ull));
            cudaMemset(d_ncand, 0, 4);
            const ull SUB = 1ull << 30;                    // keep each launch short
            for (ull off = 0; off < count; off += SUB) {
                ull n = count - off < SUB ? count - off : SUB;
                search<<<prop.multiProcessorCount * 16, 256>>>(L, start + off, n, d_cand, CAP, d_ncand, d_cnt, ca, cb);
            }
            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess) { printf("ERROR %s\n", cudaGetErrorString(err)); fflush(stdout); return 1; }
            ull cnt[9];
            unsigned nc;
            cudaMemcpy(cnt, d_cnt, sizeof cnt, cudaMemcpyDeviceToHost);
            cudaMemcpy(&nc, d_ncand, 4, cudaMemcpyDeviceToHost);
            unsigned stored = nc < CAP ? nc : CAP;
            cudaMemcpy(cand.data(), d_cand, stored * sizeof(ull), cudaMemcpyDeviceToHost);
            double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            printf("done L=%d start=%llu count=%llu says4=%llu add=%llu const=%llu mul=%llu twoa=%llu twob=%llu other=%llu overflow=%llu secs=%.3f device=%s cands=",
                   L, start, count, cnt[1], cnt[2], cnt[3], cnt[4], cnt[5], cnt[6], cnt[7], cnt[8], secs, prop.name);
            for (unsigned i = 0; i < stored; i++) printf("%s%llx", i ? "," : "", cand[i]);
            printf("\n");
            fflush(stdout);
        }
        return 0;
    }

    // verify: read "L program_hex" lines, check each on all 65,536 input pairs
    std::vector<ull> progs;
    std::vector<int> lens, cells;
    char vline[256];
    while (fgets(vline, sizeof vline, stdin)) {
        int L; ull p; unsigned ca = 5, cb = 6;
        if (sscanf(vline, "%d %llx %u %u", &L, &p, &ca, &cb) < 2) continue;
        lens.push_back(L); progs.push_back(p); cells.push_back(ca); cells.push_back(cb);
    }
    int n = (int)progs.size();
    if (n == 0) return 0;
    ull *d_p; int *d_l, *d_c; unsigned *d_f, *d_ff, *d_info;
    cudaMalloc(&d_p, n * sizeof(ull)); cudaMalloc(&d_l, n * sizeof(int)); cudaMalloc(&d_c, 2 * n * sizeof(int));
    cudaMalloc(&d_f, n * 4); cudaMalloc(&d_ff, n * 4); cudaMalloc(&d_info, n * 12);
    cudaMemcpy(d_p, progs.data(), n * sizeof(ull), cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, lens.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, cells.data(), 2 * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_f, 0, n * 4);
    cudaMemset(d_ff, 0xFF, n * 4);
    auto t0 = std::chrono::steady_clock::now();
    const int CH = 4096;                                   // candidates per launch
    for (int off = 0; off < n; off += CH) {
        int m = n - off < CH ? n - off : CH;
        verify<<<(m * 256 + 255) / 256, 256>>>(d_p + off, d_l + off, d_c + 2 * off, m, d_f + off, d_ff + off);
        cudaDeviceSynchronize();
    }
    detail<<<(n + 255) / 256, 256>>>(d_p, d_l, d_c, n, d_info);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "ERROR %s\n", cudaGetErrorString(err)); return 1; }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::vector<unsigned> f(n), ff(n), info(n * 3);
    cudaMemcpy(f.data(), d_f, n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(ff.data(), d_ff, n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(info.data(), d_info, n * 12, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++)
        printf("%d %llx %d %d fails=%u first_fail=%u outputs=%u halts=%u steps=%u\n", lens[i], progs[i], cells[2 * i], cells[2 * i + 1], f[i],
               f[i] ? ff[i] : 0, info[i * 3], info[i * 3 + 1], info[i * 3 + 2]);
    fprintf(stderr, "verified %d programs x 65536 input pairs in %.2f s on %s\n", n, secs, prop.name);
    return 0;
}
