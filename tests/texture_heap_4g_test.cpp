// No texture past byte 4 GiB of its heap: NVIDIA 621.24 binds a texture at or past that byte 4 GiB lower, over another texture (Surfbird
// T241, docs/known-driver-issues.md). create_texture refuses one that ends past 4 GiB, with an error; one that ends at 4 GiB keeps its memory
#include <NoGraphicsAPI/NoGraphicsAPI.hpp>
#include <stdio.h>
#include <string.h>
using namespace gpu;

namespace
{

constexpr uint32 side = 1024;
constexpr uint64 texture_bytes = uint64(side) * side * 4;
constexpr uint64 four_gib = uint64(1) << 32;
int refusals = 0;

void count_refusal(const char* message, bool fatal) noexcept
{
    if (!fatal && strstr(message, "past 4 GiB")) ++refusals;
    else fprintf(stderr, "Unexpected error: %s\n", message);
}

uint32 pattern(uint32 which, uint32 i)
{
    return which * 0x9E3779B9u ^ (i * 2654435761u + 12345u);
}

struct Copy
{
    Texture* texture;
    GpuRange range;
    bool upload;
};

void run(Device* device, TimelineSemaphore* timeline, uint64& value, const Copy& copy)
{
    CommandPool* pool = create_command_pool(device);
    CommandBuffer* commands = begin_commands(pool);
    if (copy.upload)
    {
        initialize_texture(commands, copy.texture);
        copy_memory_to_texture(commands, copy.range, copy.texture);
    }
    else copy_texture_to_memory(commands, copy.texture, copy.range);
    end_commands(commands);
    submit(device, {.commands = {commands}, .completion = {.semaphore = timeline, .value = ++value}});
    wait_timeline({.semaphore = timeline, .value = value});
    destroy_command_pool(pool);
}

// The control: A at the heap's start, B ending at 4 GiB; A written, then B, then both read back unchanged
bool check_control(Device* device, TimelineSemaphore* timeline, uint64& value, const TextureDesc& desc, uint64 b_size)
{
    const TextureHeap heap = create_texture_heap(device, four_gib);
    if (!heap.owner)
    {
        printf("No texture heap of 4 GiB: control skipped\n");
        return true;
    }
    Texture* a = create_texture(device, desc, heap, 0);
    Texture* b = create_texture(device, desc, heap, four_gib - b_size);
    bool valid = a && b;
    const GpuHeap upload = create_gpu_heap(device, 2 * texture_bytes);
    const GpuHeap readback = create_gpu_heap(device, 2 * texture_bytes, MemoryType::readback);
    if (valid)
    {
        uint32* const up = reinterpret_cast<uint32*>(upload.range.cpu);
        for (uint32 i = 0; i < side * side; ++i)
        {
            up[i] = pattern(1, i);
            up[side * side + i] = pattern(2, i);
        }
        memset(readback.range.cpu, 0, 2 * texture_bytes);
        const Copy steps[] = {{a, {.gpu = upload.range.gpu, .size = texture_bytes}, true},
            {b, {.gpu = upload.range.gpu + texture_bytes, .size = texture_bytes}, true},
            {a, {.gpu = readback.range.gpu, .size = texture_bytes}, false},
            {b, {.gpu = readback.range.gpu + texture_bytes, .size = texture_bytes}, false}};
        for (const Copy& step : steps) run(device, timeline, value, step);
        const uint32* const back = reinterpret_cast<const uint32*>(readback.range.cpu);
        uint32 bad = 0;
        for (uint32 i = 0; i < side * side; ++i) bad += (back[i] != pattern(1, i)) + (back[side * side + i] != pattern(2, i));
        printf("B ending at 4 GiB: %u wrong texels\n", bad);
        valid = bad == 0;
    }
    destroy_gpu_heap(upload);
    destroy_gpu_heap(readback);
    if (a) destroy_texture(a);
    if (b) destroy_texture(b);
    destroy_texture_heap(heap);
    return valid;
}

// B at 4 GiB and B across 4 GiB: both refused, with an error each
bool check_refused(Device* device, const TextureDesc& desc, uint64 b_size, uint64 align)
{
    const TextureHeap heap = create_texture_heap(device, four_gib + b_size);
    if (!heap.owner)
    {
        printf("No texture heap past 4 GiB: refusals skipped\n");
        return true;
    }
    refusals = 0;
    set_error_callback(count_refusal);
    Texture* at = create_texture(device, desc, heap, four_gib);
    Texture* across = create_texture(device, desc, heap, (four_gib - b_size / 2) / align * align);
    set_error_callback(nullptr);
    printf("B at 4 GiB: %s; B across 4 GiB: %s; errors %d\n", at ? "made" : "refused", across ? "made" : "refused", refusals);
    const bool valid = !at && !across && refusals == 2;
    if (at) destroy_texture(at);
    if (across) destroy_texture(across);
    destroy_texture_heap(heap);
    return valid;
}

}

int main()
{
    const DeviceInit initialized = create_device({.timestamp_query_count = 0});
    if (initialized.error == Error::unsupported) return 77;
    if (initialized.error != Error::none) return 1;
    Device* device = initialized.device;
    TimelineSemaphore* timeline = create_timeline_semaphore(device);
    uint64 value = 0;
    const TextureDesc desc{.type = TextureType::two_d, .extent = {.x = side, .y = side, .z = 1}, .format = Format::r32_uint,
        .usage = TextureUsage::transfer_source | TextureUsage::transfer_destination};
    const SizeAlign required = get_texture_size_align(device, desc);
    const uint64 align = required.align > 0 ? required.align : 1;
    const bool valid = check_control(device, timeline, value, desc, required.size) && check_refused(device, desc, required.size, align);
    wait_idle(device);
    destroy_timeline_semaphore(timeline);
    destroy_device(device);
    printf("No texture past 4 GiB of its heap: %s\n", valid ? "passed" : "failed");
    return valid ? 0 : 1;
}
