/* NANO explorer, Vulkan host (AMD GPUs). Runs nano.spv (compiled from nano.comp). Programs up to 8 bytes.
 *
 * usage: nano_vk L start count gpu_index [--nooutput]     classes to stdout
 *        nano_vk serve L gpu_index zstd_threads           read "start count file" lines from stdin,
 *                                                         write classes zstd-compressed to file,
 *                                                         answer one "done ..." line per range
 * gpu_index counts discrete GPUs only.
 *
 * Work is submitted in sub-ranges of 32 Mi programs (one dispatch each, far below any GPU
 * watchdog), double-buffered: the GPU computes sub-range i while sub-range i-1 is written out.
 * Uses a compute-only queue when available (the graphics ring has a ~2 s watchdog on amdgpu).
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s failed: %d\n", #x, r_); exit(1); } } while (0)

typedef unsigned long long ull;
typedef struct { uint32_t L, start_lo, start_hi, count, write_cls, inv_base, wg_base; } Push;
typedef struct { ull sum, cnt[7]; double secs; } Res;

#define SUB (1ull << 25)                       /* programs per submission */
#define GROUPS (SUB / 1024)                    /* 256 invocations x 4 programs per group */

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

static VkPhysicalDevice phys;
static VkDevice dev;
static VkQueue queue;
static VkPipeline pipe;
static VkPipelineLayout pl;
static VkDescriptorSet ds[2];
static VkCommandBuffer cb[2];
static VkFence fence[2];
static void *p_cls[2], *p_part[2], *p_cnt[2];
static VkPhysicalDeviceProperties props;

static uint32_t find_mem(uint32_t bits, VkMemoryPropertyFlags want)
{
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    fprintf(stderr, "no suitable memory type\n");
    exit(1);
}

static uint32_t find_mem_opt(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags fallback)
{
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    return find_mem(bits, fallback);
}

static VkBuffer make_buffer(VkDeviceSize size, void **map)
{
    VkBuffer buf;
    VkDeviceMemory mem;
    VkBufferCreateInfo bi = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, size,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL };
    CK(vkCreateBuffer(dev, &bi, NULL, &buf));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, buf, &mr);
    /* HOST_CACHED: the CPU reads every result byte back; uncached (write-combined) memory
       makes those reads crawl. Fall back to plain host-visible memory if there is no cached type. */
    VkMemoryPropertyFlags hv = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t type = getenv("NANO_UNCACHED") ? find_mem(mr.memoryTypeBits, hv)
                  : find_mem_opt(mr.memoryTypeBits, hv | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, hv);
    VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, type };
    CK(vkAllocateMemory(dev, &ai, NULL, &mem));
    CK(vkBindBufferMemory(dev, buf, mem, 0));
    CK(vkMapMemory(dev, mem, 0, VK_WHOLE_SIZE, 0, map));
    return buf;
}

static void init(int want)
{
    FILE *f = fopen("nano.spv", "rb");
    if (!f) { fprintf(stderr, "nano.spv not found\n"); exit(1); }
    fseek(f, 0, SEEK_END); long spv_len = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t *spv = malloc(spv_len);
    if (fread(spv, 1, spv_len, f) != (size_t)spv_len) exit(1);
    fclose(f);

    VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "nano", 1, "none", 1, VK_API_VERSION_1_2 };
    VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app, 0, NULL, 0, NULL };
    VkInstance inst;
    CK(vkCreateInstance(&ici, NULL, &inst));
    uint32_t nphys = 0;
    vkEnumeratePhysicalDevices(inst, &nphys, NULL);
    VkPhysicalDevice *all = malloc(nphys * sizeof *all);
    vkEnumeratePhysicalDevices(inst, &nphys, all);
    int seen = -1;
    for (uint32_t i = 0; i < nphys; i++) {
        vkGetPhysicalDeviceProperties(all[i], &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && ++seen == want) { phys = all[i]; break; }
    }
    if (!phys) { fprintf(stderr, "discrete GPU %d not found\n", want); exit(1); }
    vkGetPhysicalDeviceProperties(phys, &props);

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, NULL);
    VkQueueFamilyProperties *qf = malloc(nq * sizeof *qf);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qf);
    uint32_t qfi = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++)
        if ((qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) { qfi = i; break; }
    if (qfi == UINT32_MAX)
        for (uint32_t i = 0; i < nq; i++) if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfi = i; break; }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, qfi, 1, &prio };
    VkPhysicalDeviceFeatures feat = {0};
    feat.shaderInt64 = VK_TRUE;
    VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, NULL, 0, 1, &qci, 0, NULL, 0, NULL, &feat };
    CK(vkCreateDevice(phys, &dci, NULL, &dev));
    vkGetDeviceQueue(dev, qfi, 0, &queue);

    VkDescriptorSetLayoutBinding binds[3];
    for (int i = 0; i < 3; i++)
        binds[i] = (VkDescriptorSetLayoutBinding){ (uint32_t)i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL };
    VkDescriptorSetLayoutCreateInfo dli = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 3, binds };
    VkDescriptorSetLayout dsl;
    CK(vkCreateDescriptorSetLayout(dev, &dli, NULL, &dsl));
    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push) };
    VkPipelineLayoutCreateInfo pli = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &dsl, 1, &pcr };
    CK(vkCreatePipelineLayout(dev, &pli, NULL, &pl));
    VkShaderModuleCreateInfo smi = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, (size_t)spv_len, spv };
    VkShaderModule sm;
    CK(vkCreateShaderModule(dev, &smi, NULL, &sm));
    VkComputePipelineCreateInfo cpi = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL },
        pl, VK_NULL_HANDLE, 0 };
    CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, &pipe));

    VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6 };
    VkDescriptorPoolCreateInfo dpi = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 2, 1, &ps };
    VkDescriptorPool dp;
    CK(vkCreateDescriptorPool(dev, &dpi, NULL, &dp));
    VkCommandPoolCreateInfo cpci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL,
                                     VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, qfi };
    VkCommandPool cp;
    CK(vkCreateCommandPool(dev, &cpci, NULL, &cp));
    VkCommandBufferAllocateInfo cbai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, cp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 2 };
    CK(vkAllocateCommandBuffers(dev, &cbai, cb));

    for (int b = 0; b < 2; b++) {
        VkBuffer bufs[3] = { make_buffer(SUB, &p_cls[b]), make_buffer(GROUPS * 8, &p_part[b]),
                             make_buffer(GROUPS * 32, &p_cnt[b]) };
        VkDescriptorSetAllocateInfo dai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, dp, 1, &dsl };
        CK(vkAllocateDescriptorSets(dev, &dai, &ds[b]));
        VkDescriptorBufferInfo dbi[3];
        VkWriteDescriptorSet wds[3];
        for (int i = 0; i < 3; i++) {
            dbi[i] = (VkDescriptorBufferInfo){ bufs[i], 0, VK_WHOLE_SIZE };
            wds[i] = (VkWriteDescriptorSet){ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, ds[b], (uint32_t)i, 0, 1,
                                             VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &dbi[i], NULL };
        }
        vkUpdateDescriptorSets(dev, 3, wds, 0, NULL);
        VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, 0 };
        CK(vkCreateFence(dev, &fci, NULL, &fence[b]));
    }
}

/* wait for buffer b, add its partial sums, write its classes */
static void collect(int b, ull n, FILE *out, Res *r)
{
    CK(vkWaitForFences(dev, 1, &fence[b], VK_TRUE, UINT64_MAX));
    CK(vkResetFences(dev, 1, &fence[b]));
    ull groups = (n + 1023) / 1024;
    for (ull g = 0; g < groups; g++) {
        r->sum += ((uint64_t *)p_part[b])[g];
        for (int k = 1; k < 7; k++) r->cnt[k] += ((uint32_t *)p_cnt[b])[g * 8 + k];
    }
    if (out) fwrite(p_cls[b], 1, n, out);
}

static Res process(uint32_t L, ull start, ull count, FILE *out)
{
    Res r;
    memset(&r, 0, sizeof r);
    double t0 = now();
    ull nsub = (count + SUB - 1) / SUB, len[2] = {0, 0};
    int b = 0;
    for (ull i = 0; i < nsub; i++) {
        ull s = start + i * SUB, n = count - i * SUB < SUB ? count - i * SUB : SUB;
        Push push = { L, (uint32_t)s, (uint32_t)(s >> 32), (uint32_t)n, out != NULL, 0, 0 };
        VkCommandBufferBeginInfo cbbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL };
        CK(vkResetCommandBuffer(cb[b], 0));
        CK(vkBeginCommandBuffer(cb[b], &cbbi));
        vkCmdBindPipeline(cb[b], VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cb[b], VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds[b], 0, NULL);
        vkCmdPushConstants(cb[b], pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof push, &push);
        vkCmdDispatch(cb[b], (uint32_t)((n + 1023) / 1024), 1, 1);
        CK(vkEndCommandBuffer(cb[b]));
        VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb[b], 0, NULL };
        CK(vkQueueSubmit(queue, 1, &si, fence[b]));
        len[b] = n;
        if (i > 0) collect(1 - b, len[1 - b], out, &r);
        b ^= 1;
    }
    if (nsub) collect(1 - b, len[1 - b], out, &r);
    r.secs = now() - t0;
    /* self-check: every single program must have been classified */
    ull total = r.cnt[1] + r.cnt[2] + r.cnt[3] + r.cnt[4] + r.cnt[5] + r.cnt[6];
    if (total != count) {
        fprintf(stderr, "ERROR: only %llu of %llu programs were computed (GPU lost work)\n", total, count);
        printf("ERROR lost work\n");
        exit(1);
    }
    return r;
}

static void report(FILE *f, const char *what, ull start, ull count, const Res *r)
{
    fprintf(f, "%s start=%llu programs=%llu hashsum=%llu counts=%llu,%llu,%llu,%llu,%llu,%llu secs=%.6f device=%s\n",
            what, start, count, r->sum, r->cnt[1], r->cnt[2], r->cnt[3], r->cnt[4], r->cnt[5], r->cnt[6],
            r->secs, props.deviceName);
    fflush(f);
}

int main(int argc, char **argv)
{
    int serve = argc > 1 && strcmp(argv[1], "serve") == 0;
    if (argc < 5) {
        fprintf(stderr, "usage: nano_vk L start count gpu_index [--nooutput] | nano_vk serve L gpu_index zstd_threads\n");
        return 2;
    }
    uint32_t L = (uint32_t)atoi(argv[serve ? 2 : 1]);
    if (L < 1 || L > 8) { fprintf(stderr, "L must be 1..8\n"); return 2; }
    init(atoi(argv[serve ? 3 : 4]));

    if (!serve) {
        ull start = strtoull(argv[2], 0, 0), count = strtoull(argv[3], 0, 0);
        int out = !(argc > 5 && strcmp(argv[5], "--nooutput") == 0);
        Res r = process(L, start, count, out ? stdout : NULL);
        fflush(stdout);
        report(stderr, "", start, count, &r);
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
        if (!z) { printf("ERROR cannot start zstd\n"); fflush(stdout); return 1; }
        Res r = process(L, start, count, z);
        if (pclose(z) != 0) { printf("ERROR zstd failed for %s\n", path); fflush(stdout); return 1; }
        report(stdout, "done", start, count, &r);
    }
    return 0;
}
