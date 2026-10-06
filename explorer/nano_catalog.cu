// Behavior catalog: run every 5-byte NANO program on 16 probe inputs (a in M5, b in M6) and collect
// every distinct behavior (the 16 answers) that is useful: answers every probe and depends on the input.
// Per range a GPU hash table keeps: behavior hash, number of programs, one example program.
//
// usage: nano_catalog serve gpu     stdin "start count"   stdout "done start=.. programs=.. useful=.. entries=N overflow=.."
//                                   followed by N lines "hash count example"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

typedef unsigned long long ull;
#define L 5
// 16 fixed probe inputs (also used by the Python side, ai/catalog.py)
__constant__ unsigned char PA[16] = {3, 200, 17, 0, 255, 91, 128, 44, 7, 160, 66, 1, 230, 99, 12, 181};
__constant__ unsigned char PB[16] = {5, 13, 250, 0, 1, 77, 128, 210, 100, 33, 66, 254, 9, 140, 31, 2};

__device__ __forceinline__ unsigned getb(unsigned m0, unsigned m1, unsigned m2, unsigned m3, unsigned n)
{
    unsigned w = n < 4 ? m0 : n < 8 ? m1 : n < 12 ? m2 : m3;
    return (w >> ((n & 3) * 8)) & 255u;
}

__device__ int run_io(ull p, unsigned a, unsigned b)
{
    unsigned prog0 = 0, prog1 = 0;
    for (int i = 0; i < L; i++) {
        unsigned v = (unsigned)(p >> (8 * (L - 1 - i))) & 255u;
        if (i < 4) prog0 |= v << (8 * i); else prog1 |= v << (8 * (i - 4));
    }
    unsigned m0 = prog0, m1 = prog1 | (a << 8) | (b << 16), m2 = 0, m3 = 0, A = 0, pc = 0;
    for (int steps = 0; steps < 64; steps++) {
        unsigned ins = ((pc < 4 ? m0 : m1) >> ((pc & 3) * 8)) & 255u;
        pc = pc + 1 >= L ? 0 : pc + 1;
        unsigned op = ins >> 4, n = ins & 15u;
        if (op == 15) return -1;
        unsigned val = getb(m0, m1, m2, m3, n);
        if (op == 13) return (int)val;
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
        pc = (op == 7 || (op == 8 && A == 0)) ? n % L : pc;
        A = nA & 255u;
    }
    return -1;
}

__global__ void catalog(ull start, ull count, ull *keys, unsigned *cnts, ull *ex, unsigned mask, ull *stat)
{
    ull my_useful = 0, my_over = 0;
    for (ull i = blockIdx.x * (ull)blockDim.x + threadIdx.x; i < count; i += (ull)gridDim.x * blockDim.x) {
        ull p = start + i;
        ull h = 1469598103934665603ull;                 // FNV-1a 64 over the 16 answers
        int first = -2;
        bool useful = true, varies = false;
        for (int t = 0; t < 16; t++) {
            int o = run_io(p, PA[t], PB[t]);
            if (o < 0) { useful = false; break; }
            if (first == -2) first = o; else varies |= o != first;
            h = (h ^ (ull)o) * 1099511628211ull;
        }
        if (!useful || !varies) continue;
        my_useful++;
        if (h == 0) h = 1;
        unsigned slot = (unsigned)h & mask;
        bool placed = false;
        for (int probe = 0; probe < 256; probe++) {
            ull prev = atomicCAS(&keys[slot], 0ull, h);
            if (prev == 0ull) { ex[slot] = p; atomicAdd(&cnts[slot], 1u); placed = true; break; }
            if (prev == h) { atomicAdd(&cnts[slot], 1u); placed = true; break; }
            slot = (slot + 1) & mask;
        }
        if (!placed) my_over++;
    }
    atomicAdd(&stat[0], my_useful);
    atomicAdd(&stat[1], my_over);
}

int main(int argc, char **argv)
{
    if (argc < 3 || strcmp(argv[1], "serve") != 0) { fprintf(stderr, "usage: nano_catalog serve gpu\n"); return 2; }
    int gpu = atoi(argv[2]);
    cudaSetDevice(gpu);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpu);
    const unsigned SIZE = 1u << 22, MASK = SIZE - 1;
    ull *d_keys, *d_ex, *d_stat;
    unsigned *d_cnt;
    cudaMalloc(&d_keys, SIZE * 8); cudaMalloc(&d_ex, SIZE * 8); cudaMalloc(&d_cnt, SIZE * 4); cudaMalloc(&d_stat, 16);
    std::vector<ull> keys(SIZE), ex(SIZE);
    std::vector<unsigned> cnt(SIZE);
    char line[256];
    ull start, count;
    while (fgets(line, sizeof line, stdin)) {
        if (sscanf(line, "%llu %llu", &start, &count) != 2) continue;
        auto t0 = std::chrono::steady_clock::now();
        cudaMemset(d_keys, 0, SIZE * 8); cudaMemset(d_cnt, 0, SIZE * 4); cudaMemset(d_stat, 0, 16);
        const ull SUB = 1ull << 28;
        for (ull off = 0; off < count; off += SUB) {
            ull n = count - off < SUB ? count - off : SUB;
            catalog<<<prop.multiProcessorCount * 16, 256>>>(start + off, n, d_keys, d_cnt, d_ex, MASK, d_stat);
        }
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) { printf("ERROR %s\n", cudaGetErrorString(err)); fflush(stdout); return 1; }
        ull stat[2];
        cudaMemcpy(stat, d_stat, 16, cudaMemcpyDeviceToHost);
        cudaMemcpy(keys.data(), d_keys, SIZE * 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(ex.data(), d_ex, SIZE * 8, cudaMemcpyDeviceToHost);
        cudaMemcpy(cnt.data(), d_cnt, SIZE * 4, cudaMemcpyDeviceToHost);
        unsigned entries = 0;
        for (unsigned i = 0; i < SIZE; i++) entries += keys[i] != 0;
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("done start=%llu programs=%llu useful=%llu entries=%u overflow=%llu secs=%.3f device=%s\n",
               start, count, stat[0], entries, stat[1], secs, prop.name);
        for (unsigned i = 0; i < SIZE; i++)
            if (keys[i]) printf("%llx %u %llx\n", keys[i], cnt[i], ex[i]);
        fflush(stdout);
    }
    return 0;
}
