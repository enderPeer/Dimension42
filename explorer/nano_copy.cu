// "Which programs copy themselves?"  CUDA search over NANO programs (no inputs, 64 steps).
//
// A program of L bytes is a COPIER if at some step an exact copy of its original L bytes stands in
// memory at another place M[k..k+L-1] that does not overlap the program (k >= L), and at least two of
// its bytes are non-zero (so empty memory cannot pass as a "copy" of zeros).
// Reported per copier: first step the copy exists, offset k, whether the original was intact at that
// moment (real reproduction) and whether the copy still exists at the end.
//
// usage: nano_copy serve gpu     stdin "L start count"   stdout "done ... copiers=N" + "cands=" list
//        entries: program-L-k-step-intact-persists (hex program, decimal rest)
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

typedef unsigned long long ull;

__device__ __forceinline__ unsigned getb(const unsigned *m, unsigned n) { return (m[n >> 2] >> ((n & 3) * 8)) & 255u; }

// copy of the L program bytes (p0) at offset k?
__device__ __forceinline__ bool copy_at(const unsigned *m, const unsigned char *p0, int L, int k)
{
    for (int i = 0; i < L; i++)
        if (getb(m, k + i) != p0[i]) return false;
    return true;
}

__device__ int find_copy(const unsigned *m, const unsigned char *p0, int L)
{
    for (int k = L; k + L <= 16; k++)
        if (copy_at(m, p0, L, k)) return k;
    return -1;
}

__global__ void search(int L, ull start, ull count, ull *out, unsigned cap, unsigned *nout, ull *stat)
{
    ull my = 0;
    for (ull i = blockIdx.x * (ull)blockDim.x + threadIdx.x; i < count; i += (ull)gridDim.x * blockDim.x) {
        ull p = start + i;
        unsigned char p0[8];
        int nonzero = 0;
        unsigned m[4] = {0, 0, 0, 0};
        for (int b = 0; b < L; b++) {
            p0[b] = (unsigned char)(p >> (8 * (L - 1 - b)));
            nonzero += p0[b] != 0;
            m[b >> 2] |= (unsigned)p0[b] << ((b & 3) * 8);
        }
        if (nonzero < 2) continue;
        unsigned A = 0, pc = 0;
        int first_step = -1, first_k = -1, intact = 0;
        int steps = 0;
        for (; steps < 64; ) {
            steps++;
            unsigned ins = getb(m, pc);
            pc = pc + 1 >= (unsigned)L ? 0 : pc + 1;
            unsigned op = ins >> 4, n = ins & 15;
            if (op == 15) break;
            unsigned val = getb(m, n);
            unsigned k3 = n & 7u;
            unsigned nA = A;
            switch (op) {
            case 1: nA = n; break;
            case 2: nA = A + n; break;
            case 3: nA = A - n; break;
            case 4: nA = val; break;
            case 6: nA = A + val; break;
            case 11: nA = (A << k3) | (A >> (8 - k3)); break;
            case 12: nA = A ^ val; break;
            }
            bool dow = op == 5 || op == 9 || op == 10 || op == 14;
            if (dow) {
                unsigned wr = op == 5 ? A : op == 9 ? val + 1 : op == 10 ? val - 1 : ~val;
                m[n >> 2] = (m[n >> 2] & ~(255u << ((n & 3) * 8))) | ((wr & 255u) << ((n & 3) * 8));
                if (first_step < 0) {
                    int k = find_copy(m, p0, L);
                    if (k >= 0) { first_step = steps; first_k = k; intact = copy_at(m, p0, L, 0); }
                }
            }
            if (op == 7 || (op == 8 && A == 0)) pc = n % (unsigned)L;
            A = nA & 255u;
        }
        if (first_step < 0) continue;
        my++;
        int persists = find_copy(m, p0, L) >= 0;
        unsigned idx = atomicAdd(nout, 1u);
        if (idx < cap) {
            out[2 * idx] = p;
            out[2 * idx + 1] = ((ull)L << 32) | ((ull)first_k << 24) | ((ull)first_step << 16) | ((ull)intact << 8) | (ull)persists;
        } else atomicAdd(&stat[1], 1ull);
    }
    atomicAdd(&stat[0], my);
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: nano_copy serve gpu\n"); return 2; }
    int gpu = atoi(argv[2]);
    cudaSetDevice(gpu);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpu);
    const unsigned CAP = 1u << 22;
    ull *d_out, *d_stat;
    unsigned *d_n;
    cudaMalloc(&d_out, 2ull * CAP * 8); cudaMalloc(&d_stat, 16); cudaMalloc(&d_n, 4);
    std::vector<ull> out(2ull * CAP);
    char line[256];
    int L;
    ull start, count;
    while (fgets(line, sizeof line, stdin)) {
        if (sscanf(line, "%d %llu %llu", &L, &start, &count) != 3) continue;
        auto t0 = std::chrono::steady_clock::now();
        cudaMemset(d_stat, 0, 16); cudaMemset(d_n, 0, 4);
        const ull SUB = 1ull << 30;
        for (ull off = 0; off < count; off += SUB) {
            ull n = count - off < SUB ? count - off : SUB;
            search<<<prop.multiProcessorCount * 16, 256>>>(L, start + off, n, d_out, CAP, d_n, d_stat);
        }
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) { printf("ERROR %s\n", cudaGetErrorString(err)); fflush(stdout); return 1; }
        ull stat[2];
        unsigned nn;
        cudaMemcpy(stat, d_stat, 16, cudaMemcpyDeviceToHost);
        cudaMemcpy(&nn, d_n, 4, cudaMemcpyDeviceToHost);
        unsigned stored = nn < CAP ? nn : CAP;
        cudaMemcpy(out.data(), d_out, 2ull * stored * 8, cudaMemcpyDeviceToHost);
        printf("done L=%d start=%llu count=%llu copiers=%llu overflow=%llu secs=%.3f device=%s cands=", L, start, count,
               stat[0], stat[1], std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), prop.name);
        for (unsigned i = 0; i < stored; i++) {
            ull info = out[2 * i + 1];
            printf("%s%llx-%llu-%llu-%llu-%llu-%llu", i ? "," : "", out[2 * i], info >> 32, (info >> 24) & 255,
                   (info >> 16) & 255, (info >> 8) & 255, info & 255);
        }
        printf("\n");
        fflush(stdout);
    }
    return 0;
}
