// "Which programs add THREE numbers?"  CUDA search over NANO programs (sibling of nano_search.cu).
//
// Input:  a in M[ca], b in M[cb], c in M[cc] before the program starts.
// Answer: the first value the program outputs with OUT.
// Pass 1: answer for 2+2+2 must be 6, then 32 more triples must give (a+b+c) mod 256.
// Pass 2 (verify): candidates are run on all 16,777,216 triples.
//
// usage: nano_search3 serve gpu      stdin "L start count ca cb cc"
//        nano_search3 verify gpu     stdin "L program ca cb cc"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

typedef unsigned long long ull;

__constant__ unsigned char TRI[32][3] = {
    {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 2, 3}, {3, 2, 1}, {2, 3, 1}, {5, 0, 7}, {255, 1, 0}, {0, 255, 1},
    {1, 0, 255}, {128, 64, 64}, {100, 55, 1}, {7, 250, 3}, {9, 9, 9}, {200, 0, 100}, {13, 31, 77}, {255, 255, 255},
    {64, 32, 16}, {16, 32, 64}, {6, 1, 0}, {0, 6, 1}, {77, 123, 45}, {45, 77, 123}, {2, 0, 0}, {0, 2, 0}, {0, 0, 2},
    {11, 2, 30}, {30, 11, 2}, {2, 30, 11}, {90, 90, 90}, {1, 1, 1}};

__device__ __forceinline__ unsigned getb(unsigned m0, unsigned m1, unsigned m2, unsigned m3, unsigned n)
{
    unsigned w = n < 4 ? m0 : n < 8 ? m1 : n < 12 ? m2 : m3;
    return (w >> ((n & 3) * 8)) & 255u;
}

__device__ __forceinline__ void put(unsigned *m, unsigned cell, unsigned v)
{
    m[cell >> 2] = (m[cell >> 2] & ~(255u << ((cell & 3) * 8))) | (v << ((cell & 3) * 8));
}

__device__ int run3(ull p, int L, unsigned a, unsigned b, unsigned c, unsigned ca, unsigned cb, unsigned cc, bool full,
                    unsigned *outc_r = nullptr, unsigned *halted_r = nullptr, unsigned *steps_r = nullptr)
{
    unsigned m[4] = {0, 0, 0, 0};
    for (int i = 0; i < L; i++) put(m, i, (unsigned)(p >> (8 * (L - 1 - i))) & 255u);
    put(m, ca, a); put(m, cb, b); put(m, cc, c);
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
        unsigned sh = (n & 3) * 8, keep = ~(255u << sh), pv = (wr & 255u) << sh, wi = n >> 2;
        m0 = dow && wi == 0 ? (m0 & keep) | pv : m0;
        m1 = dow && wi == 1 ? (m1 & keep) | pv : m1;
        m2 = dow && wi == 2 ? (m2 & keep) | pv : m2;
        m3 = dow && wi == 3 ? (m3 & keep) | pv : m3;
        pc = (op == 7 || (op == 8 && A == 0)) ? n % (unsigned)L : pc;
        A = nA & 255u;
    }
    if (outc_r) { *outc_r = outc; *halted_r = halted; *steps_r = steps; }
    return first;
}

// cnt: 1 says6  2 add  3 const6  4 other  5 overflow
__global__ void search(int L, ull start, ull count, unsigned ca, unsigned cb, unsigned cc,
                       ull *cand, unsigned cap, unsigned *ncand, ull *cnt)
{
    unsigned c_says = 0, c_add = 0, c_const = 0, c_other = 0;
    for (ull i = blockIdx.x * (ull)blockDim.x + threadIdx.x; i < count; i += (ull)gridDim.x * blockDim.x) {
        ull p = start + i;
        if (run3(p, L, 2, 2, 2, ca, cb, cc, false) != 6) continue;
        c_says++;
        bool fadd = true, fconst = true;
        for (int t = 0; t < 32 && (fadd || fconst); t++) {
            unsigned a = TRI[t][0], b = TRI[t][1], c = TRI[t][2];
            int o = run3(p, L, a, b, c, ca, cb, cc, false);
            fadd &= o == (int)((a + b + c) & 255u);
            fconst &= o == 6;
        }
        if (fadd) {
            c_add++;
            unsigned idx = atomicAdd(ncand, 1u);
            if (idx < cap) cand[idx] = p; else atomicAdd(&cnt[5], 1ull);
        } else if (fconst) c_const++;
        else c_other++;
    }
    atomicAdd(&cnt[1], (ull)c_says); atomicAdd(&cnt[2], (ull)c_add);
    atomicAdd(&cnt[3], (ull)c_const); atomicAdd(&cnt[4], (ull)c_other);
}

// one thread per (candidate, a, b): checks all 256 values of c
__global__ void verify(const ull *progs, const int *meta, int n, unsigned *fails)
{
    ull t = blockIdx.x * (ull)blockDim.x + threadIdx.x;
    if (t >= (ull)n * 65536) return;
    int k = (int)(t / 65536);
    unsigned a = (t % 65536) / 256, b = t % 256;
    const int *m = meta + 4 * k;                       // L, ca, cb, cc
    unsigned bad = 0;
    for (unsigned c = 0; c < 256; c++)
        bad += run3(progs[k], m[0], a, b, c, m[1], m[2], m[3], false) != (int)((a + b + c) & 255u);
    if (bad) atomicAdd(&fails[k], bad);
}

__global__ void detail(const ull *progs, const int *meta, int n, unsigned *info)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const int *m = meta + 4 * k;
    unsigned outc, halted, steps;
    run3(progs[k], m[0], 2, 2, 2, m[1], m[2], m[3], true, &outc, &halted, &steps);
    info[3 * k] = outc; info[3 * k + 1] = halted; info[3 * k + 2] = steps;
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: nano_search3 serve|verify gpu\n"); return 2; }
    int gpu = atoi(argv[2]);
    cudaSetDevice(gpu);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpu);

    if (strcmp(argv[1], "serve") == 0) {
        const unsigned CAP = 1u << 22;
        ull *d_cand, *d_cnt;
        unsigned *d_ncand;
        cudaMalloc(&d_cand, CAP * sizeof(ull));
        cudaMalloc(&d_cnt, 6 * sizeof(ull));
        cudaMalloc(&d_ncand, 4);
        std::vector<ull> cand(CAP);
        char line[256];
        int L;
        ull start, count;
        unsigned ca, cb, cc;
        while (fgets(line, sizeof line, stdin)) {
            if (sscanf(line, "%d %llu %llu %u %u %u", &L, &start, &count, &ca, &cb, &cc) != 6) continue;
            auto t0 = std::chrono::steady_clock::now();
            cudaMemset(d_cnt, 0, 6 * sizeof(ull));
            cudaMemset(d_ncand, 0, 4);
            const ull SUB = 1ull << 30;
            for (ull off = 0; off < count; off += SUB) {
                ull n = count - off < SUB ? count - off : SUB;
                search<<<prop.multiProcessorCount * 16, 256>>>(L, start + off, n, ca, cb, cc, d_cand, CAP, d_ncand, d_cnt);
            }
            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess) { printf("ERROR %s\n", cudaGetErrorString(err)); fflush(stdout); return 1; }
            ull cnt[6];
            unsigned nc;
            cudaMemcpy(cnt, d_cnt, sizeof cnt, cudaMemcpyDeviceToHost);
            cudaMemcpy(&nc, d_ncand, 4, cudaMemcpyDeviceToHost);
            unsigned stored = nc < CAP ? nc : CAP;
            cudaMemcpy(cand.data(), d_cand, stored * sizeof(ull), cudaMemcpyDeviceToHost);
            double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            printf("done L=%d start=%llu count=%llu says=%llu add=%llu const=%llu other=%llu overflow=%llu secs=%.3f device=%s cands=",
                   L, start, count, cnt[1], cnt[2], cnt[3], cnt[4], cnt[5], secs, prop.name);
            for (unsigned i = 0; i < stored; i++) printf("%s%llx", i ? "," : "", cand[i]);
            printf("\n");
            fflush(stdout);
        }
        return 0;
    }

    std::vector<ull> progs;
    std::vector<int> meta;
    char line[256];
    while (fgets(line, sizeof line, stdin)) {
        int L; ull p; unsigned ca, cb, cc;
        if (sscanf(line, "%d %llx %u %u %u", &L, &p, &ca, &cb, &cc) != 5) continue;
        progs.push_back(p);
        meta.insert(meta.end(), {L, (int)ca, (int)cb, (int)cc});
    }
    int n = (int)progs.size();
    if (!n) return 0;
    ull *d_p; int *d_m; unsigned *d_f, *d_i;
    cudaMalloc(&d_p, n * sizeof(ull)); cudaMalloc(&d_m, 4 * n * sizeof(int));
    cudaMalloc(&d_f, n * 4); cudaMalloc(&d_i, 12 * n);
    cudaMemcpy(d_p, progs.data(), n * sizeof(ull), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, meta.data(), 4 * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_f, 0, n * 4);
    auto t0 = std::chrono::steady_clock::now();
    const int CH = 64;                                       // candidates per launch (64 x 65,536 threads)
    for (int off = 0; off < n; off += CH) {
        int m = n - off < CH ? n - off : CH;
        verify<<<(unsigned)(((ull)m * 65536 + 255) / 256), 256>>>(d_p + off, d_m + 4 * off, m, d_f + off);
        cudaDeviceSynchronize();
    }
    detail<<<(n + 255) / 256, 256>>>(d_p, d_m, n, d_i);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { fprintf(stderr, "ERROR %s\n", cudaGetErrorString(err)); return 1; }
    std::vector<unsigned> f(n), info(3 * n);
    cudaMemcpy(f.data(), d_f, n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(info.data(), d_i, 12 * n, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++)
        printf("%d %llx %d %d %d fails=%u outputs=%u halts=%u steps=%u\n", meta[4 * i], progs[i], meta[4 * i + 1],
               meta[4 * i + 2], meta[4 * i + 3], f[i], info[3 * i], info[3 * i + 1], info[3 * i + 2]);
    fprintf(stderr, "verified %d programs x 16,777,216 triples in %.2f s on %s\n", n,
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), prop.name);
    return 0;
}
