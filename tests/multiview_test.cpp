#include <NoGraphicsAPI/NoGraphicsAPI.hpp>
#include "multiview_read_shared.h"
#include "shader_code.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Renders two views into the two layers of an array texture: view 0 fills the left half of layer 0 in red, view 1 the right half
// of layer 1 in green. Positions and colors come from SV_ViewID. Covers vertex and mesh PSOs, and a 4x multisample array.
namespace
{
constexpr uint32 view_mask = (1u << multiview_test_views) - 1u;
constexpr uint32 pixel_count = multiview_test_size * multiview_test_size * multiview_test_views;
constexpr uint64 single_bytes = uint64(pixel_count) * 4;
constexpr uint64 sample_bytes = uint64(pixel_count) * multiview_test_samples * sizeof(float4);

struct Target
{
    gpu::TextureHeap heap{};
    gpu::Texture* texture = nullptr;
    gpu::RenderView* view = nullptr;
};

bool create_target(gpu::Device* device, uint8 sample_count, Target& target) noexcept
{
    const gpu::TextureDesc desc{
        .type = gpu::TextureType::two_d_array,
        .extent = {.x = multiview_test_size, .y = multiview_test_size, .z = 1},
        .layer_count = multiview_test_views,
        .format = gpu::Format::rgba8_unorm,
        .sample_count = sample_count,
        .usage = gpu::TextureUsage::color_attachment | (sample_count == 1 ? gpu::TextureUsage::transfer_source : gpu::TextureUsage::sampled),
    };
    const gpu::SizeAlign placement = gpu::get_texture_size_align(device, desc);
    if (!placement.size) return false;
    target.heap = gpu::create_texture_heap(device, placement.size);
    if (!target.heap.owner) return false;
    target.texture = gpu::create_texture(device, desc, target.heap, 0);
    if (!target.texture) return false;
    target.view = gpu::create_render_view(target.texture, {.layer_count = multiview_test_views});
    return target.view != nullptr;
}

void destroy_target(Target& target) noexcept
{
    gpu::destroy_render_view(target.view);
    gpu::destroy_texture(target.texture);
    gpu::destroy_texture_heap(target.heap);
}

bool expected_covered(uint32 layer, uint32 x) noexcept
{
    return layer == 0 ? x < multiview_test_size / 2 : x >= multiview_test_size / 2;
}

bool check_single(const uint8* pixels, const char* name) noexcept
{
    for (uint32 layer = 0; layer < multiview_test_views; ++layer)
        for (uint32 y = 0; y < multiview_test_size; ++y)
            for (uint32 x = 0; x < multiview_test_size; ++x)
            {
                const uint8* pixel = pixels + ((layer * multiview_test_size + y) * multiview_test_size + x) * 4;
                const bool covered = expected_covered(layer, x);
                const bool valid = covered ? pixel[0] == (layer == 0 ? 255 : 0) && pixel[1] == (layer == 1 ? 255 : 0) && pixel[2] == 0 && pixel[3] == 255
                                           : pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0 && pixel[3] == 0;
                if (!valid)
                {
                    fprintf(stderr, "%s multiview: layer %u pixel (%u, %u) = (%u, %u, %u, %u).\n", name, layer, x, y, pixel[0], pixel[1], pixel[2],
                            pixel[3]);
                    return false;
                }
            }
    return true;
}

bool check_samples(const float4* samples) noexcept
{
    for (uint32 layer = 0; layer < multiview_test_views; ++layer)
        for (uint32 y = 0; y < multiview_test_size; ++y)
            for (uint32 x = 0; x < multiview_test_size; ++x)
                for (uint32 sample = 0; sample < multiview_test_samples; ++sample)
                {
                    const float4& value = samples[((layer * multiview_test_size + y) * multiview_test_size + x) * multiview_test_samples + sample];
                    const bool covered = expected_covered(layer, x);
                    const bool valid = covered ? value.x == (layer == 0 ? 1.0f : 0.0f) && value.y == (layer == 1 ? 1.0f : 0.0f) && value.z == 0.0f &&
                                                     value.w == 1.0f
                                               : value.x == 0.0f && value.y == 0.0f && value.z == 0.0f && value.w == 0.0f;
                    if (!valid)
                    {
                        fprintf(stderr, "4x multiview: layer %u pixel (%u, %u) sample %u = (%g, %g, %g, %g).\n", layer, x, y, sample, value.x,
                                value.y, value.z, value.w);
                        return false;
                    }
                }
    return true;
}
}

int main()
{
    const gpu::DeviceInit initialized = gpu::create_device({.timestamp_query_count = 0});
    if (initialized.error == gpu::Error::unsupported) return 77;
    if (initialized.error != gpu::Error::none) return 1;
    gpu::Device* device = initialized.device;
    const gpu::DeviceCaps& caps = gpu::get_device_caps(device);
    printf("Multiview views: %u, mesh multiview views: %u, sample counts 0x%x\n", caps.max_multiview_view_count, caps.max_mesh_multiview_view_count,
           caps.sample_counts);
    if (caps.max_multiview_view_count < multiview_test_views)
    {
        printf("Multiview unavailable; skipping.\n");
        gpu::destroy_device(device);
        return 77;
    }
    const bool test_mesh = caps.max_mesh_multiview_view_count >= multiview_test_views;
    const bool test_msaa = (caps.sample_counts & multiview_test_samples) != 0;
    if (!test_mesh) printf("Mesh multiview unavailable; testing vertex PSOs only.\n");
    if (!test_msaa) printf("4x MSAA unavailable; skipping the multisample case.\n");

    const gpu::Span<byte> vertex = load_test_shader(NOGRAPHICSAPI_MULTIVIEW_VERTEX);
    const gpu::Span<byte> mesh = load_test_shader(NOGRAPHICSAPI_MULTIVIEW_MESH);
    const gpu::Span<byte> fragment = load_test_shader(NOGRAPHICSAPI_MULTIVIEW_FRAGMENT);
    const gpu::Span<byte> read = load_test_shader(NOGRAPHICSAPI_MULTIVIEW_READ);
    if (!vertex.data || !mesh.data || !fragment.data || !read.data) return 1;
    const gpu::ColorTargetDesc color_target{.format = gpu::Format::rgba8_unorm};
    gpu::PSO* vertex_pso = gpu::create_graphics_pso(device, {
        .vertex = {.code = {vertex.data, vertex.size}, .entry_point = "vertexMain"},
        .fragment = {.code = {fragment.data, fragment.size}, .entry_point = "fragmentMain"},
        .color_targets = {&color_target, 1},
        .view_mask = view_mask,
    });
    gpu::PSO* mesh_pso = test_mesh ? gpu::create_mesh_pso(device, {
        .mesh = {.code = {mesh.data, mesh.size}, .entry_point = "meshMain", .threadgroup_size = {.x = 4, .y = 1, .z = 1}},
        .fragment = {.code = {fragment.data, fragment.size}, .entry_point = "fragmentMain"},
        .color_targets = {&color_target, 1},
        .view_mask = view_mask,
    }) : nullptr;
    gpu::PSO* msaa_pso = test_msaa ? gpu::create_graphics_pso(device, {
        .vertex = {.code = {vertex.data, vertex.size}, .entry_point = "vertexMain"},
        .fragment = {.code = {fragment.data, fragment.size}, .entry_point = "fragmentMain"},
        .color_targets = {&color_target, 1},
        .sample_count = multiview_test_samples,
        .view_mask = view_mask,
    }) : nullptr;
    gpu::PSO* read_pso = gpu::create_compute_pso(device, {.code = {read.data, read.size}, .entry_point = "computeMain",
                                                          .threadgroup_size = {.x = 8, .y = 8, .z = 1}});
    free(vertex.data);
    free(mesh.data);
    free(fragment.data);
    free(read.data);
    if (!vertex_pso || (test_mesh && !mesh_pso) || (test_msaa && !msaa_pso) || !read_pso) return 1;

    Target targets[3]{};
    if (!create_target(device, 1, targets[0]) || !create_target(device, 1, targets[1]) ||
        (test_msaa && !create_target(device, multiview_test_samples, targets[2])))
        return 1;
    gpu::TextureDescriptorHeap* descriptors = gpu::create_texture_descriptor_heap(device, 1);
    const gpu::GpuHeap roots = gpu::create_gpu_heap(device, 256);
    const gpu::GpuHeap output = gpu::create_gpu_heap(device, sample_bytes, gpu::MemoryType::gpu_only);
    const gpu::GpuHeap readback = gpu::create_gpu_heap(device, 2 * single_bytes + sample_bytes, gpu::MemoryType::readback);
    gpu::TimelineSemaphore* timeline = gpu::create_timeline_semaphore(device);
    gpu::CommandPool* pool = gpu::create_command_pool(device);
    if (!descriptors || !roots.owner || !output.owner || !readback.owner || !timeline || !pool) return 1;
    memset(readback.range.cpu, 0xa5, size_t(readback.range.size));
    if (test_msaa) gpu::write_texture_descriptor(descriptors, 0, targets[2].texture, gpu::TextureDescriptorType::sampled);
    const MultiviewReadRoot read_root{.output = reinterpret_cast<float4*>(output.range.gpu)};
    memcpy(roots.range.cpu, &read_root, sizeof(read_root));

    gpu::CommandBuffer* commands = gpu::begin_commands(pool);
    gpu::PSO* const draw_psos[3]{vertex_pso, mesh_pso, msaa_pso};
    for (uint32 index = 0; index < 3; ++index)
    {
        if (!draw_psos[index]) continue;
        gpu::initialize_texture(commands, targets[index].texture);
        gpu::begin_render_pass(commands, {.colors = {{.render_view = targets[index].view, .load = gpu::LoadOp::clear, .clear = {.w = 0.0f}}},
                                          .view_mask = view_mask});
        gpu::bind_pso(commands, draw_psos[index]);
        if (index == 1) gpu::draw_meshlets(commands, nullptr, {.x = 1, .y = 1, .z = 1});
        else gpu::draw(commands, nullptr, 6);
        gpu::end_render_pass(commands);
    }
    gpu::barrier(commands, gpu::Stage::color_output, gpu::Access::color_write, gpu::Stage::transfer | gpu::Stage::compute,
                 gpu::Access::transfer_read | gpu::Access::shader_read);
    gpu::copy_texture_to_memory(commands, targets[0].texture, {.gpu = readback.range.gpu, .size = single_bytes});
    if (test_mesh) gpu::copy_texture_to_memory(commands, targets[1].texture, {.gpu = readback.range.gpu + single_bytes, .size = single_bytes});
    if (test_msaa)
    {
        gpu::set_texture_descriptor_heap(commands, descriptors);
        gpu::bind_pso(commands, read_pso);
        gpu::dispatch(commands, roots.range.gpu, {.x = 1, .y = 1, .z = multiview_test_views});
        gpu::barrier(commands, gpu::Stage::compute, gpu::Access::shader_write, gpu::Stage::transfer, gpu::Access::transfer_read);
        gpu::copy_memory(commands, gpu::gpu_range(output), {.gpu = readback.range.gpu + 2 * single_bytes, .size = sample_bytes});
    }
    gpu::barrier(commands, gpu::Stage::transfer, gpu::Access::transfer_write, gpu::Stage::host, gpu::Access::host_read);
    gpu::end_commands(commands);
    gpu::submit(device, {.commands = {commands}, .completion = {.semaphore = timeline, .value = 1}});
    gpu::wait_timeline({.semaphore = timeline, .value = 1});

    bool valid = check_single(reinterpret_cast<const uint8*>(readback.range.cpu), "Vertex");
    if (test_mesh) valid &= check_single(reinterpret_cast<const uint8*>(readback.range.cpu + single_bytes), "Mesh");
    if (test_msaa) valid &= check_samples(reinterpret_cast<const float4*>(readback.range.cpu + 2 * single_bytes));

    gpu::wait_idle(device);
    gpu::destroy_command_pool(pool);
    gpu::destroy_timeline_semaphore(timeline);
    gpu::destroy_gpu_heap(readback);
    gpu::destroy_gpu_heap(output);
    gpu::destroy_gpu_heap(roots);
    gpu::destroy_texture_descriptor_heap(descriptors);
    for (Target& target : targets)
        if (target.texture) destroy_target(target);
    gpu::destroy_pso(read_pso);
    if (msaa_pso) gpu::destroy_pso(msaa_pso);
    if (mesh_pso) gpu::destroy_pso(mesh_pso);
    gpu::destroy_pso(vertex_pso);
    gpu::destroy_device(device);
    printf("%s: multiview into 2-layer textures through SV_ViewID (vertex%s%s)\n", valid ? "PASS" : "FAIL", test_mesh ? ", mesh" : "",
           test_msaa ? ", 4x MSAA" : "");
    return valid ? 0 : 1;
}
