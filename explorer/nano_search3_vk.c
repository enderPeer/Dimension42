/* "Which programs add THREE numbers?"  Vulkan host for nano_search3.comp (pass 1), AMD GPUs.
 * usage: nano_search3_vk serve gpu_index     stdin: "L start count ca cb cc"; same output line as nano_search3.cu
 * Submissions of 32 Mi programs each (far below the GPU watchdog), compute-only queue.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { printf("ERROR %s failed: %d\n", #x, r_); fflush(stdout); exit(1); } } while (0)
typedef unsigned long long ull;
typedef struct { uint32_t L, start_lo, start_hi, count, cap, ca, cb, cc; } Push;
#define SUB (1ull << 25)
#define CAP (1u << 22)

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static VkPhysicalDevice phys;
static VkDevice dev;

static VkBuffer make_buffer(VkDeviceSize size, void **map)
{
    VkBuffer buf;
    VkDeviceMemory mem;
    VkBufferCreateInfo bi = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              VK_SHARING_MODE_EXCLUSIVE, 0, NULL };
    CK(vkCreateBuffer(dev, &bi, NULL, &buf));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, buf, &mr);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    VkMemoryPropertyFlags want[2] = {
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT };
    uint32_t type = UINT32_MAX;
    for (int w = 0; w < 2 && type == UINT32_MAX; w++)
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
            if ((mr.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want[w]) == want[w]) { type = i; break; }
    VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, mr.size, type };
    CK(vkAllocateMemory(dev, &ai, NULL, &mem));
    CK(vkBindBufferMemory(dev, buf, mem, 0));
    CK(vkMapMemory(dev, mem, 0, VK_WHOLE_SIZE, 0, map));
    return buf;
}

int main(int argc, char **argv)
{
    if (argc < 3 || strcmp(argv[1], "serve") != 0) { fprintf(stderr, "usage: nano_search3_vk serve gpu_index\n"); return 2; }
    int want = atoi(argv[2]);
    FILE *f = fopen("nano_search3.spv", "rb");
    if (!f) { printf("ERROR nano_search3.spv not found\n"); return 1; }
    fseek(f, 0, SEEK_END); long len = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t *spv = malloc(len);
    if (fread(spv, 1, len, f) != (size_t)len) return 1;
    fclose(f);

    VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "nano", 1, "none", 1, VK_API_VERSION_1_2 };
    VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app, 0, NULL, 0, NULL };
    VkInstance inst;
    CK(vkCreateInstance(&ici, NULL, &inst));
    uint32_t np = 0;
    vkEnumeratePhysicalDevices(inst, &np, NULL);
    VkPhysicalDevice *all = malloc(np * sizeof *all);
    vkEnumeratePhysicalDevices(inst, &np, all);
    VkPhysicalDeviceProperties props;
    int seen = -1;
    for (uint32_t i = 0; i < np; i++) {
        vkGetPhysicalDeviceProperties(all[i], &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && ++seen == want) { phys = all[i]; break; }
    }
    if (!phys) { printf("ERROR discrete GPU %d not found\n", want); return 1; }
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
    VkQueue queue;
    vkGetDeviceQueue(dev, qfi, 0, &queue);

    void *p_cand, *p_cnt;
    VkBuffer bufs[2] = { make_buffer((VkDeviceSize)CAP * 8, &p_cand), make_buffer(64, &p_cnt) };
    VkDescriptorSetLayoutBinding binds[2];
    for (int i = 0; i < 2; i++)
        binds[i] = (VkDescriptorSetLayoutBinding){ (uint32_t)i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL };
    VkDescriptorSetLayoutCreateInfo dli = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 2, binds };
    VkDescriptorSetLayout dsl;
    CK(vkCreateDescriptorSetLayout(dev, &dli, NULL, &dsl));
    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push) };
    VkPipelineLayoutCreateInfo pli = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &dsl, 1, &pcr };
    VkPipelineLayout pl;
    CK(vkCreatePipelineLayout(dev, &pli, NULL, &pl));
    VkShaderModuleCreateInfo smi = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, (size_t)len, spv };
    VkShaderModule sm;
    CK(vkCreateShaderModule(dev, &smi, NULL, &sm));
    VkComputePipelineCreateInfo cpi = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
        { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", NULL }, pl, VK_NULL_HANDLE, 0 };
    VkPipeline pipe;
    CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, &pipe));
    VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 };
    VkDescriptorPoolCreateInfo dpi = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 1, 1, &ps };
    VkDescriptorPool dp;
    CK(vkCreateDescriptorPool(dev, &dpi, NULL, &dp));
    VkDescriptorSetAllocateInfo dai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, dp, 1, &dsl };
    VkDescriptorSet ds;
    CK(vkAllocateDescriptorSets(dev, &dai, &ds));
    VkDescriptorBufferInfo dbi[2] = { { bufs[0], 0, VK_WHOLE_SIZE }, { bufs[1], 0, VK_WHOLE_SIZE } };
    VkWriteDescriptorSet wds[2];
    for (int i = 0; i < 2; i++)
        wds[i] = (VkWriteDescriptorSet){ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, ds, (uint32_t)i, 0, 1,
                                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &dbi[i], NULL };
    vkUpdateDescriptorSets(dev, 2, wds, 0, NULL);
    VkCommandPoolCreateInfo cpci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, qfi };
    VkCommandPool cp;
    CK(vkCreateCommandPool(dev, &cpci, NULL, &cp));
    VkCommandBufferAllocateInfo cbai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, cp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1 };
    VkCommandBuffer cb;
    CK(vkAllocateCommandBuffers(dev, &cbai, &cb));
    VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, 0 };
    VkFence fence;
    CK(vkCreateFence(dev, &fci, NULL, &fence));

    uint32_t *cnt = p_cnt;
    char line[256];
    int L;
    ull start, count;
    while (fgets(line, sizeof line, stdin)) {
        unsigned cell_a = 5, cell_b = 6, cell_c = 7;
        if (sscanf(line, "%d %llu %llu %u %u %u", &L, &start, &count, &cell_a, &cell_b, &cell_c) != 6) continue;
        double t0 = now();
        ull tot[6] = {0};
        memset(cnt, 0, 64);
        for (ull off = 0; off < count; off += SUB) {
            ull n = count - off < SUB ? count - off : SUB, s = start + off;
            for (int q = 1; q < 6; q++) cnt[q] = 0;
            Push push = { (uint32_t)L, (uint32_t)s, (uint32_t)(s >> 32), (uint32_t)n, CAP, cell_a, cell_b, cell_c };
            VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL };
            CK(vkResetCommandBuffer(cb, 0));
            CK(vkBeginCommandBuffer(cb, &bi));
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
            vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof push, &push);
            vkCmdDispatch(cb, (uint32_t)((n + 1023) / 1024), 1, 1);
            CK(vkEndCommandBuffer(cb));
            VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb, 0, NULL };
            CK(vkQueueSubmit(queue, 1, &si, fence));
            CK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
            CK(vkResetFences(dev, 1, &fence));
            for (int q = 1; q < 6; q++) tot[q] += cnt[q];
        }
        uint32_t nc = cnt[0], stored = nc < CAP ? nc : CAP;
        printf("done L=%d start=%llu count=%llu says=%llu add=%llu const=%llu other=%llu overflow=%llu secs=%.3f device=%s cands=",
               L, start, count, tot[1], tot[2], tot[3], tot[4], tot[5], now() - t0, props.deviceName);
        for (uint32_t i = 0; i < stored; i++) printf("%s%llx", i ? "," : "", (ull)((uint64_t *)p_cand)[i]);
        printf("\n");
        fflush(stdout);
    }
    return 0;
}
