#include <NoGraphicsAPI/NoGraphicsAPI.hpp>
#include "msaa_shared.h"
#include "msaa_read_shared.h"
#include "shader_code.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Renders into 4x float color, r32_uint, rg32_uint and d32_float targets, then reads every sample in a compute shader.
// Pass 0 checks coverage at a vertical edge through pixel centers; passes 1-3 compare alpha-to-coverage with plain coverage.
namespace
{
constexpr uint32 pass_count = 4;
constexpr uint32 samples_per_pass = msaa_test_size * msaa_test_size * msaa_test_samples;
constexpr uint64 pass_bytes = uint64(samples_per_pass) * sizeof(MsaaSample);
constexpr uint32 target_count = 4;

struct Pass
{
    float alpha;
    uint32 full_screen;
    bool alpha_to_coverage;
    bool depth_test;
};

constexpr Pass passes[pass_count]{
    {.alpha = 1.0f, .full_screen = 0, .alpha_to_coverage = false, .depth_test = true},
    {.alpha = 0.5f, .full_screen = 1, .alpha_to_coverage = true, .depth_test = false},
    {.alpha = 0.0f, .full_screen = 1, .alpha_to_coverage = true, .depth_test = false},
    {.alpha = 0.5f, .full_screen = 1, .alpha_to_coverage = false, .depth_test = false},
};

bool covered(const MsaaSample& value, float alpha) noexcept
{
    return value.color.x == 1.0f && value.color.w == alpha && value.id == msaa_test_id && value.id2_x == msaa_test_id2_x &&
           value.id2_y == msaa_test_id2_y;
}

bool cleared(const MsaaSample& value) noexcept
{
    return value.color.x == 0.0f && value.color.w == 0.0f && value.id == 0 && value.id2_x == 0 && value.id2_y == 0;
}

// Standard 4x locations put samples 0 and 2 left of the pixel center and samples 1 and 3 right of it.
bool check_edge(const MsaaSample* values) noexcept
{
    for (uint32 y = 0; y < msaa_test_size; ++y)
        for (uint32 x = 0; x < msaa_test_size; ++x)
            for (uint32 sample = 0; sample < msaa_test_samples; ++sample)
            {
                const MsaaSample& value = values[(y * msaa_test_size + x) * msaa_test_samples + sample];
                const bool inside = x < msaa_test_edge || (x == msaa_test_edge && (sample == 0 || sample == 2));
                if (inside ? !(covered(value, 1.0f) && value.depth == 0.25f) : !(cleared(value) && value.depth == 1.0f))
                {
                    fprintf(stderr, "MSAA edge: pixel (%u, %u) sample %u is %s, expected %s.\n", x, y, sample,
                            cleared(value) ? "clear" : "covered or invalid", inside ? "covered" : "clear");
                    return false;
                }
            }
    return true;
}

uint32 count_covered(const MsaaSample* values, float alpha, bool& valid) noexcept
{
    uint32 count = 0;
    for (uint32 index = 0; index < samples_per_pass; ++index)
    {
        if (covered(values[index], alpha)) ++count;
        else if (!cleared(values[index])) valid = false;
    }
    return count;
}
}

int main()
{
    const gpu::DeviceInit initialized = gpu::create_device({.timestamp_query_count = 0});
    if (initialized.error == gpu::Error::unsupported) return 77;
    if (initialized.error != gpu::Error::none) return 1;
    gpu::Device* device = initialized.device;
    const gpu::DeviceCaps& caps = gpu::get_device_caps(device);
    printf("Sample counts: 0x%x, integer 0x%x\n", caps.sample_counts, caps.integer_sample_counts);
    if (!(caps.sample_counts & msaa_test_samples) || !(caps.integer_sample_counts & msaa_test_samples))
    {
        printf("4x MSAA unavailable; skipping.\n");
        gpu::destroy_device(device);
        return 77;
    }

    const gpu::Span<byte> vertex = load_test_shader(NOGRAPHICSAPI_MSAA_VERTEX);
    const gpu::Span<byte> fragment = load_test_shader(NOGRAPHICSAPI_MSAA_FRAGMENT);
    const gpu::Span<byte> read = load_test_shader(NOGRAPHICSAPI_MSAA_READ);
    if (!vertex.data || !fragment.data || !read.data) return 1;
    const gpu::ColorTargetDesc color_targets[]{{.format = gpu::Format::rgba16_float}, {.format = gpu::Format::r32_uint},
                                               {.format = gpu::Format::rg32_uint}};
    gpu::PSO* draw_psos[2]{};
    for (uint32 alpha_to_coverage = 0; alpha_to_coverage < 2; ++alpha_to_coverage)
        draw_psos[alpha_to_coverage] = gpu::create_graphics_pso(device, {
            .vertex = {.code = {vertex.data, vertex.size}, .entry_point = "vertexMain"},
            .fragment = {.code = {fragment.data, fragment.size}, .entry_point = "fragmentMain"},
            .color_targets = color_targets,
            .depth_format = gpu::Format::d32_float,
            .sample_count = msaa_test_samples,
            .alpha_to_coverage = alpha_to_coverage != 0,
        });
    gpu::PSO* read_pso = gpu::create_compute_pso(device, {.code = {read.data, read.size}, .entry_point = "computeMain",
                                                          .threadgroup_size = {.x = 8, .y = 8, .z = 1}});
    free(vertex.data);
    free(fragment.data);
    free(read.data);
    if (!draw_psos[0] || !draw_psos[1] || !read_pso) return 1;

    const gpu::Format formats[target_count]{gpu::Format::rgba16_float, gpu::Format::r32_uint, gpu::Format::rg32_uint, gpu::Format::d32_float};
    gpu::TextureDesc descs[target_count]{};
    uint64 offsets[target_count]{};
    uint64 heap_size = 0;
    for (uint32 index = 0; index < target_count; ++index)
    {
        descs[index] = {
            .extent = {.x = msaa_test_size, .y = msaa_test_size, .z = 1},
            .format = formats[index],
            .sample_count = msaa_test_samples,
            .usage = gpu::TextureUsage::sampled | (index == 3 ? gpu::TextureUsage::depth_stencil_attachment : gpu::TextureUsage::color_attachment),
        };
        const gpu::SizeAlign placement = gpu::get_texture_size_align(device, descs[index]);
        if (!placement.size) return 1;
        offsets[index] = (heap_size + placement.align - 1) / placement.align * placement.align;
        heap_size = offsets[index] + placement.size;
    }
    const gpu::TextureHeap texture_heap = gpu::create_texture_heap(device, heap_size);
    gpu::TextureDescriptorHeap* descriptors = gpu::create_texture_descriptor_heap(device, target_count);
    const gpu::GpuHeap roots = gpu::create_gpu_heap(device, 512);
    const gpu::GpuHeap output = gpu::create_gpu_heap(device, pass_count * pass_bytes, gpu::MemoryType::gpu_only);
    const gpu::GpuHeap readback = gpu::create_gpu_heap(device, pass_count * pass_bytes, gpu::MemoryType::readback);
    gpu::TimelineSemaphore* timeline = gpu::create_timeline_semaphore(device);
    gpu::CommandPool* pool = gpu::create_command_pool(device);
    if (!texture_heap.owner || !descriptors || !roots.owner || !output.owner || !readback.owner || !timeline || !pool) return 1;
    gpu::Texture* textures[target_count]{};
    gpu::RenderView* views[target_count]{};
    gpu::CommandBuffer* commands = gpu::begin_commands(pool);
    for (uint32 index = 0; index < target_count; ++index)
    {
        textures[index] = gpu::create_texture(device, descs[index], texture_heap, offsets[index]);
        if (!textures[index]) return 1;
        gpu::initialize_texture(commands, textures[index]);
        views[index] = gpu::create_render_view(textures[index]);
        if (!views[index]) return 1;
        gpu::write_texture_descriptor(descriptors, index, textures[index], gpu::TextureDescriptorType::sampled);
    }

    gpu::set_texture_descriptor_heap(commands, descriptors);
    const gpu::ColorAttachment colors[]{{.render_view = views[0], .load = gpu::LoadOp::clear, .clear = {.w = 0.0f}},
                                        {.render_view = views[1], .load = gpu::LoadOp::clear, .clear = {.w = 0.0f}},
                                        {.render_view = views[2], .load = gpu::LoadOp::clear, .clear = {.w = 0.0f}}};
    for (uint32 pass = 0; pass < pass_count; ++pass)
    {
        const MsaaDrawRoot draw_root{.color = {.x = 1.0f, .w = passes[pass].alpha}, .full_screen = passes[pass].full_screen};
        const MsaaReadRoot read_root{
            .output = reinterpret_cast<MsaaSample*>(output.range.gpu + pass * pass_bytes),
            .color_texture = 0,
            .id_texture = 1,
            .id2_texture = 2,
            .depth_texture = 3,
        };
        memcpy(roots.range.cpu + pass * 64, &draw_root, sizeof(draw_root));
        memcpy(roots.range.cpu + pass * 64 + 32, &read_root, sizeof(read_root));
        if (pass)
            gpu::barrier(commands, gpu::Stage::compute, gpu::Access::shader_read, gpu::Stage::depth_stencil_tests | gpu::Stage::color_output,
                         gpu::Access::depth_stencil_write | gpu::Access::color_write);
        gpu::begin_render_pass(commands, {.colors = colors, .depth = {.render_view = views[3], .load = gpu::LoadOp::clear, .clear = 1.0f}});
        if (passes[pass].depth_test) gpu::set_depth_stencil(commands, {.depth_test = true, .depth_write = true, .depth_compare = gpu::CompareOp::less});
        gpu::bind_pso(commands, draw_psos[passes[pass].alpha_to_coverage ? 1 : 0]);
        gpu::draw(commands, roots.range.gpu + pass * 64, 6);
        gpu::end_render_pass(commands);
        gpu::barrier(commands, gpu::Stage::depth_stencil_tests | gpu::Stage::color_output, gpu::Access::depth_stencil_write | gpu::Access::color_write,
                     gpu::Stage::compute, gpu::Access::shader_read);
        gpu::bind_pso(commands, read_pso);
        gpu::dispatch(commands, roots.range.gpu + pass * 64 + 32, {.x = (msaa_test_size + 7) / 8, .y = (msaa_test_size + 7) / 8, .z = 1});
    }
    gpu::barrier(commands, gpu::Stage::compute, gpu::Access::shader_write, gpu::Stage::transfer, gpu::Access::transfer_read);
    gpu::copy_memory(commands, gpu::gpu_range(output), gpu::gpu_range(readback));
    gpu::barrier(commands, gpu::Stage::transfer, gpu::Access::transfer_write, gpu::Stage::host, gpu::Access::host_read);
    gpu::end_commands(commands);
    gpu::submit(device, {.commands = {commands}, .completion = {.semaphore = timeline, .value = 1}});
    gpu::wait_timeline({.semaphore = timeline, .value = 1});

    const MsaaSample* values = reinterpret_cast<const MsaaSample*>(readback.range.cpu);
    bool valid = check_edge(values);
    bool consistent = true;
    const uint32 partial = count_covered(values + 1 * samples_per_pass, 0.5f, consistent);
    const uint32 none = count_covered(values + 2 * samples_per_pass, 0.0f, consistent);
    const uint32 full = count_covered(values + 3 * samples_per_pass, 0.5f, consistent);
    printf("Alpha-to-coverage covered samples: alpha 0.5 %u, alpha 0 %u; without alpha-to-coverage %u of %u.\n", partial, none, full,
           samples_per_pass);
    valid &= consistent && none == 0 && full == samples_per_pass && partial > 0 && partial < samples_per_pass;

    gpu::wait_idle(device);
    gpu::destroy_command_pool(pool);
    gpu::destroy_timeline_semaphore(timeline);
    gpu::destroy_gpu_heap(readback);
    gpu::destroy_gpu_heap(output);
    gpu::destroy_gpu_heap(roots);
    for (uint32 index = 0; index < target_count; ++index)
    {
        gpu::destroy_render_view(views[index]);
        gpu::destroy_texture(textures[index]);
    }
    gpu::destroy_texture_descriptor_heap(descriptors);
    gpu::destroy_texture_heap(texture_heap);
    gpu::destroy_pso(read_pso);
    gpu::destroy_pso(draw_psos[1]);
    gpu::destroy_pso(draw_psos[0]);
    gpu::destroy_device(device);
    printf("%s: 4x MSAA float, r32_uint, rg32_uint and depth samples, edge coverage and alpha-to-coverage\n", valid ? "PASS" : "FAIL");
    return valid ? 0 : 1;
}
