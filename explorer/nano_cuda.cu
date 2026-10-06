// NANO explorer, CUDA version. Same semantics, classes and hash as nano_cpu.c (programs up to 8 bytes).
// Memory is held in 4 registers (16 bytes); every step evaluates the instruction with selects
// instead of a switch, so threads of a warp running different programs do not diverge.
//
// usage: nano_cuda L start count device [--nooutput]     classes to stdout
//        nano_cuda serve L device zstd_threads           read "start count file" lines from stdin,
//                                                        write classes zstd-compressed to file,
//                                                        answer one "done ..." line per range
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

typedef unsigned long long ull;

__device__ __forceinline__ unsigned getb(unsigned m0, unsigned m1, unsigned m2, unsigned m3, unsigned n)
{
    unsigned w = n < 4 ? m0 : n < 8 ? m1 : n < 12 ? m2 : m3;
    return (w >> ((n & 3) * 8)) & 255u;
}

__device__ __forceinline__ unsigned fnv(unsigned h, unsigned b) { return (h ^ (b & 255u)) * 16777619u; }

__device__ __forceinline__ unsigned fnvw(unsigned h, unsigned w)
{
    h = fnv(h, w); h = fnv(h, w >> 8); h = fnv(h, w >> 16); return fnv(h, w >> 24);
}

__global__ void explore(int L, ull start, ull count, unsigned char *cls_out, ull *hashsum, ull *counts)
{
    __shared__ ull sh_sum;
    __shared__ unsigned sh_cnt[7];
    if (threadIdx.x == 0) { sh_sum = 0; for (int k = 0; k < 7; k++) sh_cnt[k] = 0; }
    __syncthreads();

    ull my_sum = 0;
    unsigned my_cnt[7] = {0, 0, 0, 0, 0, 0, 0};
    for (ull i = blockIdx.x * (ull)blockDim.x + threadIdx.x; i < count; i += (ull)gridDim.x * blockDim.x) {
        ull p = start + i;
        unsigned prog0 = 0, prog1 = 0;                   // program bytes 0..3 and 4..7
        for (int b = 0; b < L; b++) {
            unsigned v = (unsigned)(p >> (8 * (L - 1 - b))) & 255u;
            if (b < 4) prog0 |= v << (8 * b); else prog1 |= v << (8 * (b - 4));
        }
        unsigned m0 = prog0, m1 = prog1, m2 = 0, m3 = 0;
        unsigned t0 = 0, t1 = 0, t2 = 0, t3 = 0, o0 = 0, o1 = 0;
        unsigned A = 0, tA = 0, pc = 0, steps = 0, outc = 0, halted = 0;

        for (;;) {
            t0 |= m0; t1 |= m1; t2 |= m2; t3 |= m3; tA |= A;
            if (steps >= 64) break;
            steps++;
            unsigned ins = ((pc < 4 ? m0 : m1) >> ((pc & 3) * 8)) & 255u;
            pc = pc + 1 >= (unsigned)L ? 0 : pc + 1;
            unsigned op = ins >> 4, n = ins & 15u;
            if (op == 15) { halted = 1; break; }
            unsigned val = getb(m0, m1, m2, m3, n);
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
            bool doo = op == 13 && outc < 8;
            unsigned osh = (outc & 3) * 8, oput = val << osh, okeep = ~(255u << osh);
            o0 = doo && outc < 4 ? (o0 & okeep) | oput : o0;
            o1 = doo && outc >= 4 ? (o1 & okeep) | oput : o1;
            outc += op == 13;
            A = nA & 255u;
        }

        unsigned pm0 = L >= 4 ? 0xFFFFFFFFu : ((1u << (8 * L)) - 1);
        unsigned pm1 = L <= 4 ? 0u : L >= 8 ? 0xFFFFFFFFu : ((1u << (8 * (L - 4))) - 1);
        bool selfmod = (m0 & pm0) != prog0 || (m1 & pm1) != prog1;
        bool active = tA || (t0 & ~pm0) || (t1 & ~pm1);
        unsigned cls = (t2 | t3) ? 6 : outc ? 5 : selfmod ? 4 : active ? 3 : halted ? 2 : 1;

        unsigned h = 2166136261u;
        h = fnv(h, A); h = fnv(h, steps); h = fnv(h, outc); h = fnv(h, halted); h = fnv(h, tA);
        h = fnvw(h, m0); h = fnvw(h, m1); h = fnvw(h, m2); h = fnvw(h, m3);
        h = fnvw(h, t0); h = fnvw(h, t1); h = fnvw(h, t2); h = fnvw(h, t3);
        h = fnvw(h, o0); h = fnvw(h, o1);
        my_sum += h;
        my_cnt[1] += cls == 1; my_cnt[2] += cls == 2; my_cnt[3] += cls == 3;
        my_cnt[4] += cls == 4; my_cnt[5] += cls == 5; my_cnt[6] += cls == 6;
        if (cls_out) cls_out[i] = (unsigned char)cls;
    }
    atomicAdd(&sh_sum, my_sum);
    for (int k = 1; k < 7; k++) atomicAdd(&sh_cnt[k], my_cnt[k]);
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(hashsum, sh_sum);
        for (int k = 1; k < 7; k++) atomicAdd(&counts[k], (ull)sh_cnt[k]);
    }
}

// ---- host: compute a range in sub-ranges, double-buffered, streaming classes to `out` ----
static const ull SUB = 1ull << 27;                      // 128 Mi programs per launch
static unsigned char *d_buf[2], *h_buf[2];
static ull *d_sum, *d_cnt;
static cudaStream_t stream;
static cudaEvent_t ev[2];
static int blocks;

struct Res { ull sum, cnt[7]; double secs; };

static Res process(int L, ull start, ull count, FILE *out)
{
    auto t0 = std::chrono::steady_clock::now();
    cudaMemsetAsync(d_sum, 0, 8, stream);
    cudaMemsetAsync(d_cnt, 0, 7 * 8, stream);
    ull nsub = (count + SUB - 1) / SUB, len[2] = {0, 0};
    int b = 0;
    for (ull i = 0; i < nsub; i++) {
        ull s = start + i * SUB, n = count - i * SUB < SUB ? count - i * SUB : SUB;
        explore<<<blocks, 256, 0, stream>>>(L, s, n, out ? d_buf[b] : nullptr, d_sum, d_cnt);
        if (out) {
            cudaMemcpyAsync(h_buf[b], d_buf[b], n, cudaMemcpyDeviceToHost, stream);
            cudaEventRecord(ev[b], stream);
            len[b] = n;
            if (i > 0) { cudaEventSynchronize(ev[1 - b]); fwrite(h_buf[1 - b], 1, len[1 - b], out); }
        }
        b ^= 1;
    }
    if (out && nsub) { cudaEventSynchronize(ev[1 - b]); fwrite(h_buf[1 - b], 1, len[1 - b], out); }
    cudaStreamSynchronize(stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); exit(1); }
    Res r;
    cudaMemcpy(&r.sum, d_sum, 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(r.cnt, d_cnt, 7 * 8, cudaMemcpyDeviceToHost);
    r.secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    // self-check: every single program must have been classified
    if (r.cnt[1] + r.cnt[2] + r.cnt[3] + r.cnt[4] + r.cnt[5] + r.cnt[6] != count) {
        fprintf(stderr, "ERROR: only %llu of %llu programs were computed\n",
                r.cnt[1] + r.cnt[2] + r.cnt[3] + r.cnt[4] + r.cnt[5] + r.cnt[6], count);
        exit(1);
    }
    return r;
}

static void report(FILE *f, const char *what, ull start, ull count, const Res &r, const char *dev)
{
    fprintf(f, "%s start=%llu programs=%llu hashsum=%llu counts=%llu,%llu,%llu,%llu,%llu,%llu secs=%.6f device=%s\n",
            what, start, count, r.sum, r.cnt[1], r.cnt[2], r.cnt[3], r.cnt[4], r.cnt[5], r.cnt[6], r.secs, dev);
    fflush(f);
}

int main(int argc, char **argv)
{
    bool serve = argc > 1 && strcmp(argv[1], "serve") == 0;
    if ((serve && argc < 5) || (!serve && argc < 5)) {
        fprintf(stderr, "usage: nano_cuda L start count device [--nooutput] | nano_cuda serve L device zstd_threads\n");
        return 2;
    }
    int L = atoi(argv[serve ? 2 : 1]);
    int dev = atoi(argv[serve ? 3 : 4]);
    if (L < 1 || L > 8) { fprintf(stderr, "L must be 1..8\n"); return 2; }

    cudaSetDevice(dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    blocks = prop.multiProcessorCount * 32;
    cudaStreamCreate(&stream);
    for (int b = 0; b < 2; b++) {
        cudaMalloc(&d_buf[b], SUB);
        cudaMallocHost(&h_buf[b], SUB);
        cudaEventCreateWithFlags(&ev[b], cudaEventDisableTiming);
    }
    cudaMalloc(&d_sum, 8);
    cudaMalloc(&d_cnt, 7 * 8);

    if (!serve) {
        ull start = strtoull(argv[2], 0, 0), count = strtoull(argv[3], 0, 0);
        bool out = !(argc > 5 && strcmp(argv[5], "--nooutput") == 0);
        Res r = process(L, start, count, out ? stdout : nullptr);
        fflush(stdout);
        report(stderr, "", start, count, r, prop.name);
        return 0;
    }
    int zt = atoi(argv[4]);
    char line[1024], path[900], cmd[2048];
    ull start, count;
    while (fgets(line, sizeof line, stdin)) {
        if (sscanf(line, "%llu %llu %899s", &start, &count, path) != 3) continue;
        /* NANO_SINK: send the compressed stream to another node instead of a local file
           (e.g. "ssh -i key ender@host"; the receiving side stores it under the given path) */
        const char *sink = getenv("NANO_SINK");
        if (sink) snprintf(cmd, sizeof cmd, "zstd -q -3 -T%d -c | %s %s", zt, sink, path);
        else snprintf(cmd, sizeof cmd, "zstd -q -3 -T%d -f -o %s", zt, path);
        FILE *z = popen(cmd, "w");
        if (!z) { fprintf(stdout, "ERROR cannot start zstd\n"); fflush(stdout); return 1; }
        Res r = process(L, start, count, z);
        if (pclose(z) != 0) { fprintf(stdout, "ERROR zstd failed for %s\n", path); fflush(stdout); return 1; }
        report(stdout, "done", start, count, r, prop.name);
    }
    return 0;
}
