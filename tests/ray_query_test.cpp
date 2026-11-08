// Ray queries end to end (DeviceCaps::ray_query; Vulkan and Metal): a bottom level of four boxes, a top level of two instances of it
// (the second moved +100 in x: the transform's layout), rays traced in a compute kernel with their own masks and t ranges; the kernel
// commits each candidate box's entry, the nearest wins. The expected hits are computed here.
#include <NoGraphicsAPI/NoGraphicsAPI.hpp>
#include "ray_query_shared.h"
#include "shader_code.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

namespace
{
struct Expected
{
    bool hit;
    uint32 primitive;
    uint32 instance;
    float t;
};
}

int main()
{
    const gpu::DeviceInit initialized = gpu::create_device();
    if (initialized.error == gpu::Error::unsupported) return 77;
    if (initialized.error != gpu::Error::none) return 1;
    gpu::Device* device = initialized.device;
    if (!gpu::get_device_caps(device).ray_query)
    {
        printf("Ray queries: not supported on %s, skipped\n", gpu::get_device_caps(device).device_name);
        gpu::destroy_device(device);
        return 77;
    }
    const gpu::Span<byte> code = load_test_shader(NOGRAPHICSAPI_RAY_QUERY_SHADER);
    if (!code.data) return 1;
    gpu::PSO* pso = gpu::create_compute_pso(device, {.code = {code.data, code.size}, .entry_point = "computeMain",
                                                     .threadgroup_size = {.x = ray_query_thread_count, .y = 1, .z = 1}});
    free(code.data);
    if (!pso) return 1;

    // The CPU-visible data: boxes at 0, instances at 256, rays at 1024, the root at 2048
    const gpu::GpuHeap data = gpu::create_gpu_heap(device, 4096);
    float* boxes = reinterpret_cast<float*>(data.range.cpu);
    for (uint32 i = 0; i < 4; ++i)
    {
        const float box[6] = {-1.0f, -1.0f, 10.0f * float(i) + 5.0f, 1.0f, 1.0f, 10.0f * float(i) + 6.0f};
        memcpy(boxes + i * 6, box, sizeof(box));
    }
    const gpu::AccelerationStructureDesc bottom{
        .type = gpu::AccelerationStructureType::bottom_level,
        .aabbs = {.aabbs = {.gpu = data.range.gpu, .size = 4 * 24}, .count = 4, .stride = 24},
    };
    const gpu::AccelerationStructureDesc top{
        .type = gpu::AccelerationStructureType::top_level, .instances = {.gpu = data.range.gpu + 256, .size = 2 * 64}, .instance_count = 2};
    const gpu::AccelerationStructureSizes bottom_sizes = gpu::get_acceleration_structure_sizes(device, bottom);
    const gpu::AccelerationStructureSizes top_sizes = gpu::get_acceleration_structure_sizes(device, top);
    gpu::AccelerationStructure* blas = gpu::create_acceleration_structure(device, gpu::AccelerationStructureType::bottom_level, bottom_sizes.storage);
    gpu::AccelerationStructure* tlas = gpu::create_acceleration_structure(device, gpu::AccelerationStructureType::top_level, top_sizes.storage);
    if (!blas || !tlas || !bottom_sizes.scratch || !top_sizes.scratch) return 1;
    const uint64 alignment = bottom_sizes.scratch_alignment > top_sizes.scratch_alignment ? bottom_sizes.scratch_alignment : top_sizes.scratch_alignment;
    const uint64 top_scratch_offset = (bottom_sizes.scratch + alignment - 1) / alignment * alignment;
    const gpu::GpuHeap scratch = gpu::create_gpu_heap(device, top_scratch_offset + top_sizes.scratch + alignment, gpu::MemoryType::gpu_only);
    byte* scratch_base = reinterpret_cast<byte*>((reinterpret_cast<uintptr>(scratch.range.gpu) + alignment - 1) / alignment * alignment);

    gpu::AccelerationStructureInstance instances[2]{};
    instances[0].custom_index_and_mask = 7u | 0x01u << 24;
    instances[0].acceleration_structure = gpu::get_acceleration_structure_address(blas);
    instances[1] = instances[0];
    instances[1].transform[0][3] = 100.0f; // row-major: the translation is each row's last column
    instances[1].custom_index_and_mask = 9u | 0x02u << 24;
    memcpy(data.range.cpu + 256, instances, sizeof(instances));

    // Rays along +z; each with its mask, t range and expected hit
    constexpr uint32 ray_count = 8;
    const RayQueryRay rays[ray_count] = {
        {.origin = {0, 0, 0}, .t_min = 0, .direction = {0, 0, 1}, .t_max = 1000, .mask = 1},
        {.origin = {0, 0, 0}, .t_min = 0, .direction = {0, 0, 1}, .t_max = 1000, .mask = 2},
        {.origin = {100, 0, 0}, .t_min = 0, .direction = {0, 0, 1}, .t_max = 1000, .mask = 2},
        {.origin = {100, 0, 0}, .t_min = 0, .direction = {0, 0, 1}, .t_max = 1000, .mask = 1},
        {.origin = {0, 0, 0}, .t_min = 7, .direction = {0, 0, 1}, .t_max = 1000, .mask = 3},
        {.origin = {50, 0, 0}, .t_min = 0, .direction = {0, 0, 1}, .t_max = 1000, .mask = 3},
        {.origin = {0.5f, 0.5f, -20}, .t_min = 0, .direction = {0, 0, 1}, .t_max = 1000, .mask = 1},
        {.origin = {100.5f, 0, 32}, .t_min = 0, .direction = {0, 0, 1}, .t_max = 2, .mask = 3},
    };
    const Expected expected[ray_count] = {
        {true, 0, 7, 5}, {false}, {true, 0, 9, 5}, {false}, {true, 1, 7, 15}, {false}, {true, 0, 7, 25}, {false},
    };
    memcpy(data.range.cpu + 1024, rays, sizeof(rays));

    const gpu::GpuHeap hits = gpu::create_gpu_heap(device, 256, gpu::MemoryType::gpu_only);
    const gpu::GpuHeap readback = gpu::create_gpu_heap(device, 256, gpu::MemoryType::readback);
    memset(readback.range.cpu, 0xA5, readback.range.size);
    const uint64 address = gpu::get_acceleration_structure_address(tlas);
    const RayQueryRoot root{.boxes = reinterpret_cast<float*>(data.range.gpu), .rays = reinterpret_cast<RayQueryRay*>(data.range.gpu + 1024),
                            .hits = reinterpret_cast<RayQueryHit*>(hits.range.gpu), .tlas = {uint32(address), uint32(address >> 32)},
                            .ray_count = ray_count};
    memcpy(data.range.cpu + 2048, &root, sizeof(root));

    gpu::TimelineSemaphore* timeline = gpu::create_timeline_semaphore(device);
    gpu::CommandPool* pool = gpu::create_command_pool(device);
    gpu::CommandBuffer* commands = gpu::begin_commands(pool);
    gpu::build_acceleration_structure(commands, blas, bottom, {.gpu = scratch_base, .size = bottom_sizes.scratch});
    gpu::barrier(commands, gpu::Stage::acceleration_structure_build, gpu::Access::acceleration_structure_write,
                 gpu::Stage::acceleration_structure_build, gpu::Access::acceleration_structure_read);
    gpu::build_acceleration_structure(commands, tlas, top, {.gpu = scratch_base + top_scratch_offset, .size = top_sizes.scratch});
    gpu::barrier(commands, gpu::Stage::acceleration_structure_build, gpu::Access::acceleration_structure_write, gpu::Stage::compute,
                 gpu::Access::acceleration_structure_read);
    gpu::bind_pso(commands, pso);
    gpu::dispatch(commands, data.range.gpu + 2048, {.x = 1, .y = 1, .z = 1});
    gpu::barrier(commands, gpu::Stage::compute, gpu::Access::shader_write, gpu::Stage::transfer, gpu::Access::transfer_read);
    gpu::copy_memory(commands, {.gpu = hits.range.gpu, .size = ray_count * sizeof(RayQueryHit)},
                     {.gpu = readback.range.gpu, .size = ray_count * sizeof(RayQueryHit)});
    gpu::barrier(commands, gpu::Stage::transfer, gpu::Access::transfer_write, gpu::Stage::host, gpu::Access::host_read);
    gpu::end_commands(commands);
    gpu::submit(device, {.commands = {commands}, .completion = {.semaphore = timeline, .value = 1}});
    gpu::wait_timeline({.semaphore = timeline, .value = 1});

    bool valid = true;
    const RayQueryHit* results = reinterpret_cast<const RayQueryHit*>(readback.range.cpu);
    for (uint32 i = 0; i < ray_count; ++i)
    {
        const RayQueryHit& hit = results[i];
        const Expected& want = expected[i];
        const bool ok = want.hit ? hit.status == 1 && hit.primitive == want.primitive && hit.instance == want.instance && fabsf(hit.t - want.t) < 1e-4f
                                 : hit.status == 0;
        if (!ok)
            printf("ray %u: status %u primitive %u instance %u t %g; expected %s %u %u %g\n", i, hit.status, hit.primitive, hit.instance,
                   double(hit.t), want.hit ? "a hit" : "a miss", want.primitive, want.instance, double(want.t));
        valid &= ok;
    }
    printf("Ray queries on %s: 4 boxes, 2 instances, %u rays: %s\n", gpu::get_device_caps(device).device_name, ray_count,
           valid ? "passed" : "failed");

    gpu::wait_idle(device);
    gpu::destroy_command_pool(pool);
    gpu::destroy_timeline_semaphore(timeline);
    gpu::destroy_acceleration_structure(tlas);
    gpu::destroy_acceleration_structure(blas);
    gpu::destroy_gpu_heap(readback);
    gpu::destroy_gpu_heap(hits);
    gpu::destroy_gpu_heap(scratch);
    gpu::destroy_gpu_heap(data);
    gpu::destroy_pso(pso);
    gpu::destroy_device(device);
    return valid ? 0 : 1;
}
