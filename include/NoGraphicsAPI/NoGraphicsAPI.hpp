#pragma once

#include <NoGraphicsAPI/span.hpp>

namespace gpu
{

struct uint32x2
{
    uint32 x = 0;
    uint32 y = 0;
};

struct uint32x3
{
    uint32 x = 0;
    uint32 y = 0;
    uint32 z = 0;
};

struct Device;
struct CommandPool;
struct Texture;
struct RenderView;
struct PSO;
struct CommandBuffer;
struct TimelineSemaphore;
struct GpuHeapOwner;
struct TextureHeapOwner;
struct TextureDescriptorHeap;
struct SamplerDescriptorHeap;
struct AccelerationStructure;

enum class Error : uint8
{
    none,
    unsupported,
    device_lost,
    driver_error,
    out_of_memory,
};

using ErrorCallback = void (*)(const char* message, bool fatal) noexcept;

// Synchronous diagnostics on the calling thread; message is valid only during the callback. Do not call graphics APIs recursively.
// The thread-local callback defaults to nullptr. Fatal execution errors still abort after the callback returns.
void set_error_callback(ErrorCallback callback) noexcept;

enum class MemoryType : uint8
{
    cpu_visible,
    gpu_only,
    readback,
};

struct SizeAlign
{
    uint64 size = 0;
    uint64 align = 0;
};

struct GpuRange
{
    void* gpu = nullptr;
    uint64 size = 0;
};

template<typename T>
struct GpuCpuRange
{
    T* cpu = nullptr;
    T* gpu = nullptr;
    uint64 size = 0; // Bytes, independent of T.
};

struct GpuHeap
{
    // Copies alias the same allocation. Pass one unchanged copy to destroy_gpu_heap exactly once.
    GpuCpuRange<byte> range{};
    const GpuHeapOwner* owner = nullptr;
};

struct TextureHeap
{
    // Copies alias the same allocation. Pass one unchanged copy to destroy_texture_heap exactly once.
    uint64 size = 0;
    const TextureHeapOwner* owner = nullptr;
};

struct TimelinePoint
{
    TimelineSemaphore* semaphore = nullptr;
    uint64 value = 0;
};

struct SubmitDesc
{
    Span<CommandBuffer* const> commands = {};
    Span<const TimelinePoint> waits = {}; // GPU waits cover all command stages.
    TimelinePoint completion = {}; // Required for every submission.
};

enum class Format : uint8
{
    r8_srgb,
    rg8_srgb,
    rgba8_srgb,
    bgra8_srgb,

    rgba4_unorm,
    r5g5b5a1_unorm,
    r5g6b5_unorm,

    r8_unorm,
    rg8_unorm,
    rgba8_unorm,
    bgra8_unorm,
    r16_unorm,
    rg16_unorm,
    rgba16_unorm,

    r8_uint,
    rg8_uint,
    rgba8_uint,
    bgra8_uint,
    r16_uint,
    rg16_uint,
    rgba16_uint,
    r32_uint,
    rg32_uint,
    rgba32_uint,

    r16_float,
    rg16_float,
    rgba16_float,
    r32_float,
    rg32_float,
    rgba32_float,

    rgb10a2_unorm,
    rg11b10_float,

    d16_unorm,
    d24_unorm_s8_uint,
    d32_float,
    s8_uint,
    d32_float_s8_uint,

    eac_rg,
    astc_4x4_srgb,
    astc_4x4_unorm,
    bc3_srgb,
    bc3_unorm,
    bc4_r, // one channel, 8 bytes per block
    bc5_rg,
    bc6h_ufloat,
    bc6h_sfloat,
    bc7_srgb,
    bc7_unorm,

    undefined, // Must remain last; preceding values are concrete texture formats.
};

struct TextureFormatInfo
{
    uint32x2 block_extent = {};
    uint32 bytes_per_block = 0;
    bool depth = false;
    bool stencil = false;
};

[[nodiscard]] constexpr TextureFormatInfo get_texture_format_info(Format format) noexcept
{
    switch (format)
    {
    case Format::r8_srgb:
    case Format::r8_unorm:
    case Format::r8_uint:
    case Format::s8_uint:
        return {
            .block_extent = {.x = 1, .y = 1},
            .bytes_per_block = 1,
            .depth = false,
            .stencil = format == Format::s8_uint,
        };
    case Format::rg8_srgb:
    case Format::rgba4_unorm:
    case Format::r5g5b5a1_unorm:
    case Format::r5g6b5_unorm:
    case Format::rg8_unorm:
    case Format::r16_unorm:
    case Format::rg8_uint:
    case Format::r16_uint:
    case Format::r16_float:
    case Format::d16_unorm:
        return {
            .block_extent = {.x = 1, .y = 1},
            .bytes_per_block = 2,
            .depth = format == Format::d16_unorm,
            .stencil = false,
        };
    case Format::rgba8_srgb:
    case Format::bgra8_srgb:
    case Format::rgba8_unorm:
    case Format::bgra8_unorm:
    case Format::rg16_unorm:
    case Format::rgba8_uint:
    case Format::bgra8_uint:
    case Format::rg16_uint:
    case Format::r32_uint:
    case Format::rg16_float:
    case Format::r32_float:
    case Format::rgb10a2_unorm:
    case Format::rg11b10_float:
    case Format::d24_unorm_s8_uint:
    case Format::d32_float:
        return {
            .block_extent = {.x = 1, .y = 1},
            .bytes_per_block = 4,
            .depth = format == Format::d24_unorm_s8_uint || format == Format::d32_float,
            .stencil = format == Format::d24_unorm_s8_uint,
        };
    case Format::rgba16_unorm:
    case Format::rgba16_uint:
    case Format::rg32_uint:
    case Format::rgba16_float:
    case Format::rg32_float:
    case Format::d32_float_s8_uint:
        return {
            .block_extent = {.x = 1, .y = 1},
            .bytes_per_block = 8,
            .depth = format == Format::d32_float_s8_uint,
            .stencil = format == Format::d32_float_s8_uint,
        };
    case Format::rgba32_uint:
    case Format::rgba32_float:
        return {
            .block_extent = {.x = 1, .y = 1},
            .bytes_per_block = 16,
        };
    case Format::bc4_r:
        return {
            .block_extent = {.x = 4, .y = 4},
            .bytes_per_block = 8,
        };
    case Format::eac_rg:
    case Format::astc_4x4_srgb:
    case Format::astc_4x4_unorm:
    case Format::bc3_srgb:
    case Format::bc3_unorm:
    case Format::bc5_rg:
    case Format::bc6h_ufloat:
    case Format::bc6h_sfloat:
    case Format::bc7_srgb:
    case Format::bc7_unorm:
        return {
            .block_extent = {.x = 4, .y = 4},
            .bytes_per_block = 16,
        };
    case Format::undefined: return {};
    }
    return {};
}

enum class TextureType : uint8
{
    one_d,
    two_d,
    three_d,
    cube,
    two_d_array,
    cube_array,
};

enum class TextureUsage : uint32
{
    none = 0,
    sampled = 1u << 0u,
    storage = 1u << 1u,
    color_attachment = 1u << 2u,
    depth_stencil_attachment = 1u << 3u,
    transfer_source = 1u << 4u,
    transfer_destination = 1u << 5u,
};

enum class TextureDescriptorType : uint8
{
    sampled,
    storage,
};

enum class TextureAspect : uint8
{
    automatic,
    color,
    depth,
    stencil,
};

constexpr TextureUsage operator|(TextureUsage lhs, TextureUsage rhs) noexcept
{
    return static_cast<TextureUsage>(static_cast<uint32>(lhs) |  static_cast<uint32>(rhs));
}

enum class Filter : uint8
{
    nearest,
    linear,
};

enum class AddressMode : uint8
{
    repeat,
    mirrored_repeat,
    clamp_to_edge,
};

enum class CompareOp : uint8
{
    never,
    less,
    equal,
    less_equal,
    greater,
    not_equal,
    greater_equal,
    always,
};

enum class CullMode : uint8
{
    none,
    clockwise,
    counter_clockwise,
};

enum class BlendFactor : uint8
{
    zero,
    one,
    source_color,
    one_minus_source_color,
    destination_color,
    one_minus_destination_color,
    source_alpha,
    one_minus_source_alpha,
    destination_alpha,
    one_minus_destination_alpha,
    source_alpha_saturate,
};

enum class BlendOp : uint8
{
    add,
    subtract,
    reverse_subtract,
    minimum,
    maximum,
};

enum class IndexType : uint8
{
    uint16,
    uint32,
};

enum class LoadOp : uint8
{
    load,
    clear,
    discard,
};

enum class StoreOp : uint8
{
    store,
    discard,
};

enum class StencilOp : uint8
{
    keep,
    zero,
    replace,
    increment_clamp,
    decrement_clamp,
    invert,
    increment_wrap,
    decrement_wrap,
};

// Ordered by Vulkan's logical execution order where stages are comparable.
// A barrier's before execution scope includes the selected and logically earlier
// stages; its after execution scope includes the selected and logically later
// stages. Vertex and task/mesh are alternative graphics branches, depth_stencil_tests
// spans early and late tests around fragment, compute and transfer are separate
// pipelines, host is a pseudo-stage, and none/all_commands are special masks.
enum class Stage : uint64
{
    none = 0,
    indirect = 1ull << 0u,
    index_input = 1ull << 1u,
    vertex = 1ull << 2u,
    task = 1ull << 3u,
    mesh = 1ull << 4u,
    depth_stencil_tests = 1ull << 5u,
    fragment = 1ull << 6u,
    color_output = 1ull << 7u,
    compute = 1ull << 8u,
    transfer = 1ull << 9u,
    host = 1ull << 10u, // Barrier destination only, paired with host_read.
    all_commands = 1ull << 11u, // All GPU command stages; excludes host.
    acceleration_structure_build = 1ull << 12u, // build_acceleration_structure (DeviceCaps::ray_query).
};

constexpr Stage operator|(Stage lhs, Stage rhs) noexcept
{
    return static_cast<Stage>(static_cast<uint64>(lhs) | static_cast<uint64>(rhs));
}

enum class Access : uint64
{
    none = 0,
    transfer_read = 1ull << 0u,
    transfer_write = 1ull << 1u,
    shader_read = 1ull << 2u,
    shader_write = 1ull << 3u,
    color_read = 1ull << 4u,
    color_write = 1ull << 5u,
    depth_stencil_read = 1ull << 6u,
    depth_stencil_write = 1ull << 7u,
    indirect_read = 1ull << 8u,
    index_read = 1ull << 9u,
    host_read = 1ull << 10u,
    descriptor_read = 1ull << 11u,
    // Acceleration structures (DeviceCaps::ray_query): a build writes the structure and reads others (a top level reads its bottom
    // levels); ray queries in shaders read it.
    acceleration_structure_read = 1ull << 12u,
    acceleration_structure_write = 1ull << 13u,
};

constexpr Access operator|(Access lhs, Access rhs) noexcept
{
    return static_cast<Access>(static_cast<uint64>(lhs) | static_cast<uint64>(rhs));
}

struct DeviceCaps
{
    const char* device_name = nullptr;
    // Flat queue indices: general first, then compute-only, then copy-only. Queue zero supports presentation.
    uint32 queue_count = 0;
    uint32 general_queue_count = 0;
    uint32 compute_queue_count = 0;
    uint32 copy_queue_count = 0;
    // Copy-only texture offsets/extents align to these texel-block counts, except at mip edges. Zero means whole mip levels only.
    uint32x3 copy_texture_granularity = {.x = 1, .y = 1, .z = 1};
    uint64 max_push_data_size = 0;
    // Common element size for suballocating TextureHeap storage; every SizeAlign::align divides this value.
    uint64 texture_heap_alignment = 0;
    float timestamp_period_ns = 0.0f; // Nanoseconds per timestamp tick; zero when profiling is unavailable.
    uint32 sub_texel_precision_bits = 0; // Fractional filtering precision, for conservative sampled-field bounds.
    bool texture_compression_bc = false;
    bool texture_compression_astc = false;
    bool storage_input_output16 = false;
    bool indirect_mesh_draw = false;
    // Bit masks of MSAA sample counts (bit value equals count: 1, 2, 4, 8) for float/normalized color and depth/stencil attachments,
    // and for integer color attachments, that are also sampleable. Formats can support fewer; get_texture_size_align() checks them.
    uint32 sample_counts = 1;
    uint32 integer_sample_counts = 1;
    uint32 max_multiview_view_count = 0; // Views in RenderingDesc::view_mask; zero when multiview rendering is unavailable.
    uint32 max_mesh_multiview_view_count = 0; // Views in MeshPSODesc::view_mask; zero when mesh PSOs cannot use multiview.
    float max_sampler_anisotropy = 1.0f; // The largest SamplerDesc::max_anisotropy that takes effect.
    // Acceleration structures and inline ray queries in every shader stage (Vulkan: VK_KHR_acceleration_structure, VK_KHR_ray_query).
    // False on Metal for now, and on Vulkan devices without the extensions.
    bool ray_query = false;
};

enum class ColorSpace : uint32 { srgb, extended_srgb_linear };

// Vulkan only: a host that must make the Vulkan instance and device itself, such as OpenXR's XR_KHR_vulkan_enable2
// (xrCreateVulkanInstanceKHR, xrGetVulkanGraphicsDevice2KHR, xrCreateVulkanDeviceKHR). create_device fills its create infos as
// usual (layers, extensions, features, queues) and passes them to these callbacks instead of calling vkCreateInstance and
// vkCreateDevice; the host may add what it needs. Pointers are Vulkan types: VkInstanceCreateInfo*, VkInstance*, VkPhysicalDevice,
// VkDeviceCreateInfo*, VkDevice*; results are VkResult. The device owns what the callbacks make (destroy_device destroys them).
struct VulkanCreateHooks
{
    void* user = nullptr;
    int32 (*create_instance)(void* user, const void* instance_create_info, void* instance_out) = nullptr;
    // The GPU the host requires, or null to let create_device choose. It must still have every capability create_device needs.
    void* (*select_physical_device)(void* user, void* instance) = nullptr;
    int32 (*create_device)(void* user, void* physical_device, const void* device_create_info, void* device_out) = nullptr;
};

// Win32 windowed device lifecycle, swapchain format/extent changes, acquire, and presentation stay on the window's message-pump thread.
// Metal calls may use a render thread; synchronize CAMetalLayer access with native UI/layer changes.
// The window/layer must outlive the device. Other calls follow the object-level threading contract below.
struct DeviceDesc
{
    void* window = nullptr; // HWND on Windows; CAMetalLayer* on macOS/iOS. Null creates a headless device.
    Format swapchain_format = Format::undefined;
    ColorSpace swapchain_color_space = ColorSpace::srgb;
    uint32 desired_swapchain_image_count = 2; // Vulkan: 1..8 presentation contexts. Metal clamps to 2..3 drawables.
    bool steam_overlay_compatibility = false; // macOS only: allow drawable texture views and use classic Metal presentation.
    // Counts are capped to each family's capacity. A nonzero request requires that kind of queue to be available.
    uint32 desired_queue_count = 1; // General graphics + compute queues; must be nonzero.
    uint32 desired_compute_queue_count = 0;
    uint32 desired_copy_queue_count = 0;
    uint32 timestamp_query_count = 256; // Per command buffer; zero disables timestamps.
    const VulkanCreateHooks* vulkan_create_hooks = nullptr; // Vulkan only; null creates the instance and device directly.
};

// The native objects of a Vulkan device, for interop (OpenXR's XrGraphicsBindingVulkan2KHR). Null on Metal.
struct VulkanHandles
{
    void* instance = nullptr;        // VkInstance
    void* physical_device = nullptr; // VkPhysicalDevice
    void* device = nullptr;          // VkDevice
    uint32 queue_family_index = 0;
    uint32 queue_index = 0;          // Within its family.
};

struct DeviceInit
{
    Device* device = nullptr;
    Error error = Error::none;
};

struct SwapchainFrame
{
    RenderView* render_view = nullptr;
    uint32x2 extent = {};
};

struct TextureDesc
{
    TextureType type = TextureType::two_d;
    uint32x3 extent = {.x = 1, .y = 1, .z = 1};
    uint32 mip_levels = 1;
    uint32 layer_count = 1; // Vulkan array layers; cube faces are individual layers.
    Format format = Format::rgba8_unorm;
    // 1, 2, 4 or 8, within DeviceCaps::sample_counts (integer_sample_counts for integer formats).
    // Multisample textures are 2D or 2D arrays with one mip level, no storage usage, and no copies to or from memory.
    uint8 sample_count = 1;
    bool mutable_format = false; // Allow format-compatible descriptor views, but could lose DCC.
    TextureUsage usage = TextureUsage::sampled;
};

struct RenderViewDesc
{
    uint32 mip_level = 0;
    uint32 slice = 0; // Physical array slice; cube faces are individual slices.
    uint32 layer_count = 1; // Consecutive slices from slice. More than one creates an array view for multiview rendering.
    // Undefined inherits the texture format. Another format needs a mutable_format texture and a compatible format, such as
    // rgba8_unorm on an rgba8_srgb texture: writes then store the shader's values without sRGB encoding.
    Format format = Format::undefined;
};

struct TextureDescriptorDesc
{
    Format format = Format::undefined; // Undefined inherits the texture format.
    TextureAspect aspect = TextureAspect::automatic; // Automatic selects color, or depth before stencil.
    uint32 base_mip = 0;
    uint32 mip_count = 0; // Zero selects every remaining mip level.
    uint32 base_layer = 0; // Vulkan array layer; cube faces are individual layers.
    uint32 layer_count = 0; // Vulkan array layers; zero selects every remaining layer.
};

struct TextureCopyDesc
{
    uint32 mip_level = 0;
    uint32 base_slice = 0; // Physical array slice; cube faces are individual slices.
    uint32 slice_count = 0; // Zero selects every remaining physical slice.
    uint32x3 offset = {};
    uint32x3 extent = {}; // Zero components select the remaining mip extent.
    uint64 row_pitch_bytes = 0;   // Zero is tightly packed.
    uint64 slice_pitch_bytes = 0; // Zero is tightly packed.
};

struct SamplerDesc
{
    Filter min_filter = Filter::linear;
    Filter mag_filter = Filter::linear;
    Filter mip_filter = Filter::linear;
    AddressMode address_u = AddressMode::repeat;
    AddressMode address_v = AddressMode::repeat;
    AddressMode address_w = AddressMode::repeat;
    bool anisotropic = false; // Uses the API's fixed 4x profile.
    uint8 max_anisotropy = 0; // 2 to 16: anisotropic filtering at this level (clamped to DeviceCaps::max_sampler_anisotropy), whatever
                              // anisotropic says; 0 or 1 keeps anisotropic's profile.
    bool compare_enabled = false;
    CompareOp compare = CompareOp::less_equal;
};

struct BlendComponentState
{
    BlendFactor source = BlendFactor::one;
    BlendFactor destination = BlendFactor::zero;
    BlendOp operation = BlendOp::add;
};

struct BlendState
{
    bool enabled = false;
    BlendComponentState color = {};
    BlendComponentState alpha = {};
};

struct ColorTargetDesc
{
    Format format = Format::undefined;
    BlendState blend = {};
    uint8 write_mask = 0xf;
};

struct RasterizationState
{
    CullMode cull = CullMode::none;
    float depth_bias_constant = 0.0f;
    float depth_bias_clamp = 0.0f;
    float depth_bias_slope = 0.0f;
};

struct Viewport
{
    float x = 0.0f;
    float y = 0.0f;
    float width = 1.0f;
    float height = 1.0f;
    float min_depth = 0.0f;
    float max_depth = 1.0f;
};

struct Scissor
{
    int32 x = 0;
    int32 y = 0;
    uint32 width = 1;
    uint32 height = 1;
};

struct StencilFaceState
{
    CompareOp compare = CompareOp::always;
    StencilOp fail = StencilOp::keep;
    StencilOp pass = StencilOp::keep;
    StencilOp depth_fail = StencilOp::keep;
    uint8 reference = 0;
};

struct DepthStencilState
{
    bool depth_test = false;
    bool depth_write = false;
    CompareOp depth_compare = CompareOp::less_equal;
    bool stencil_test = false;
    uint8 stencil_read_mask = 0xff;
    uint8 stencil_write_mask = 0xff;
    StencilFaceState front = {};
    StencilFaceState back = {};
};

struct ShaderStage
{
    // Precompiled SPIR-V on Vulkan (4-byte-aligned words), metallib on Metal. Empty omits an optional stage.
    ByteSpan code = {};
    const char* entry_point = "main";
    // Metal compute/task/mesh dimensions must match the compiled numthreads. Vulkan uses the SPIR-V metadata.
    uint32x3 threadgroup_size = {.x = 1, .y = 1, .z = 1};
};

struct GraphicsPSODesc
{
    ShaderStage vertex = {};
    ShaderStage fragment = {}; // Empty omits the fragment stage, for depth-only rasterization.
    Span<const ColorTargetDesc> color_targets = {};
    Format depth_format = Format::undefined;
    Format stencil_format = Format::undefined;
    RasterizationState rasterization = {};
    uint32 sample_count = 1; // Must match the attachment textures' sample_count.
    bool alpha_to_coverage = false; // Fragment SV_Target0 alpha masks sample coverage.
    uint32 view_mask = 0; // Must match RenderingDesc::view_mask.
};

struct MeshPSODesc
{
    ShaderStage task = {}; // Empty launches mesh workgroups directly; otherwise draws launch task workgroups.
    ShaderStage mesh = {};
    ShaderStage fragment = {}; // Empty omits the fragment stage, for depth-only rasterization.
    Span<const ColorTargetDesc> color_targets = {};
    Format depth_format = Format::undefined;
    Format stencil_format = Format::undefined;
    RasterizationState rasterization = {};
    uint32 sample_count = 1; // Must match the attachment textures' sample_count.
    bool alpha_to_coverage = false; // Fragment SV_Target0 alpha masks sample coverage.
    uint32 view_mask = 0; // Must match RenderingDesc::view_mask; views are limited by DeviceCaps::max_mesh_multiview_view_count.
};

struct ClearColor
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 1.0f;
};

struct ColorAttachment
{
    RenderView* render_view = nullptr;
    LoadOp load = LoadOp::load;
    StoreOp store = StoreOp::store;
    ClearColor clear = {};
};

struct DepthAttachment
{
    RenderView* render_view = nullptr;
    LoadOp load = LoadOp::load;
    StoreOp store = StoreOp::store;
    float clear = 1.0f;
};

struct StencilAttachment
{
    RenderView* render_view = nullptr;
    LoadOp load = LoadOp::load;
    StoreOp store = StoreOp::store;
    uint8 clear = 0;
};

struct RenderingDesc
{
    Span<const ColorAttachment> colors = {};
    DepthAttachment depth = {};
    StencilAttachment stencil = {};
    // Multiview: bit i renders view i (SV_ViewID) into layer i of every attachment's render view. Zero renders one layer.
    // The highest set bit must be below DeviceCaps::max_multiview_view_count and each render view's layer_count.
    uint32 view_mask = 0;
};

enum class RenderingFlags : uint32
{
    none = 0,
    suspending = 2,
    resuming = 4,
};

constexpr RenderingFlags operator|(RenderingFlags lhs, RenderingFlags rhs) noexcept
{
    return static_cast<RenderingFlags>(static_cast<uint32>(lhs) | static_cast<uint32>(rhs));
}

// Rendering and raster PSOs accept at most eight color attachments. A render pass needs at least one attachment to infer its area.
// All resource destruction is immediate. Destroy resources only when no recorded or executing GPU frame uses them.
// The optional NoGraphicsAPIUtility DeleteQueue can defer destruction until a submitted frame completes.
// Wait for all submitted frames to drain before destroying the device.
// Distinct resource creation/destruction, immutable queries, and timeline waits may run concurrently on one device.
// Resource lifetime changes must be synchronized with every CPU/GPU use of that resource.
// Concurrent descriptor updates require disjoint destinations, and copy source slots must not be modified by another update.
// Each (device, queue_index) and command pool (including recording its buffers) is externally synchronized; different queues/pools may run concurrently.
// Device idle/destruction requires exclusive access. Destroy command pools before their device. There are no internal queue or pool locks.
// Initialization and resource creation failures return an Error, nullptr, or an empty aggregate; details go to the optional error callback.
[[nodiscard]] DeviceInit create_device(const DeviceDesc& desc = {}) noexcept;
void destroy_device(Device* device) noexcept;
[[nodiscard]] const DeviceCaps& get_device_caps(const Device* device) noexcept;
// Vulkan: the device's native objects and the queue_index queue's family and index; empty on Metal.
[[nodiscard]] VulkanHandles get_vulkan_handles(const Device* device, uint32 queue_index = 0) noexcept;
[[nodiscard]] bool supports_texture_format(const Device* device, Format format, TextureUsage usage) noexcept;
[[nodiscard]] uint32x2 get_drawable_extent(Device* device) noexcept;
// Call after wait_idle, outside an acquired frame. Unsupported pairs leave the current mode unchanged.
// A zero-size drawable defers swapchain recreation until acquire sees a nonzero extent.
[[nodiscard]] Error set_swapchain_format(Device* device, Format format, ColorSpace color_space) noexcept;

[[nodiscard]] TimelineSemaphore* create_timeline_semaphore(Device* device, uint64 initial_value = 0) noexcept;
void destroy_timeline_semaphore(TimelineSemaphore* semaphore) noexcept;
[[nodiscard]] uint64 timeline_completed_value(const TimelineSemaphore* semaphore) noexcept;
void wait_timeline(TimelinePoint point) noexcept;
void wait_idle(Device* device) noexcept;

// Acquire outside a render pass. Later buffers in the same presentation submission may also access the returned image.
// submit_and_present prepares the image for presentation after all submitted buffers.
// Empty while the drawable extent is zero. A nonempty acquire must be submitted with submit_and_present on queue zero.
[[nodiscard]] SwapchainFrame acquire(CommandBuffer* commands) noexcept;
void submit_and_present(Device* device, const SubmitDesc& desc) noexcept;

// Every non-null returned pointer is 16-byte aligned. GPU heaps are raw blocks for application-side suballocation.
// Metal and Vulkan without device-address commands support at most 64 live GPU heaps per device.
[[nodiscard]] GpuHeap create_gpu_heap(Device* device, uint64 byte_count, MemoryType memory = MemoryType::cpu_visible) noexcept;
void destroy_gpu_heap(const GpuHeap& heap) noexcept;

template<typename T>
[[nodiscard]] constexpr GpuRange gpu_range(GpuCpuRange<T> range) noexcept
{
    return {.gpu = range.gpu, .size = range.size};
}

[[nodiscard]] constexpr GpuRange gpu_range(const GpuHeap& heap) noexcept
{
    return gpu_range(heap.range);
}

// Texture heaps use one device-selected GPU-only memory type and must outlive every placed texture.
// Placements must satisfy get_texture_size_align(). Overlapping textures require disjoint periods of GPU use and reinitialization after aliased writes.
// DeviceCaps::texture_heap_alignment can be used as a common allocator element size, avoiding per-placement leading alignment padding.
[[nodiscard]] TextureHeap create_texture_heap(Device* device, uint64 byte_count) noexcept;
void destroy_texture_heap(const TextureHeap& heap) noexcept;
[[nodiscard]] SizeAlign get_texture_size_align(Device* device, const TextureDesc& desc) noexcept;
// Creates an uninitialized texture without recording GPU commands. Views and descriptors can be created before initialize_texture().
[[nodiscard]] Texture* create_texture(Device* device, const TextureDesc& desc, const TextureHeap& heap, uint64 offset) noexcept;
void destroy_texture(Texture* texture) noexcept;
// Wraps an image made by someone else without taking it: destroy_texture leaves the image. desc describes the image as it was made
// (type, extent, layers, format, samples, mutable_format, usage). Vulkan: a VkImage (an OpenXR swapchain image); the library keeps
// every texture in one layout, begin_external_access moves the image into it before its first use in a command buffer and
// end_external_access hands it back in the layout an OpenXR runtime expects. Metal: an id<MTLTexture> of the device's own MTLDevice
// (another library's render target on the system's default device), retained and made resident until destroy_texture; its pixel
// format and extent must match desc; the access calls do nothing. Ordering with the other library's queue is the caller's (no
// shared event: wait for its work before recording a use, and let it write again only after the use completed).
[[nodiscard]] Texture* create_external_texture(Device* device, const TextureDesc& desc, void* native_image) noexcept;
[[nodiscard]] RenderView* create_render_view(Texture* texture, const RenderViewDesc& desc = {}) noexcept;
void destroy_render_view(RenderView* render_view) noexcept;
// Descriptor slots form application-owned namespaces. Writes and copies require indices within capacity;
// do not modify slots until all commands using their previous contents have completed. Copies may overlap.
[[nodiscard]] TextureDescriptorHeap* create_texture_descriptor_heap(Device* device, uint32 capacity) noexcept;
void destroy_texture_descriptor_heap(TextureDescriptorHeap* heap) noexcept;
void write_texture_descriptor(TextureDescriptorHeap* heap, uint32 index, const Texture* texture, TextureDescriptorType type,
                              const TextureDescriptorDesc& desc = {}) noexcept;
void copy_texture_descriptors(const TextureDescriptorHeap* source, uint32 source_index, TextureDescriptorHeap* destination,
                              uint32 destination_index, uint32 count) noexcept;
[[nodiscard]] SamplerDescriptorHeap* create_sampler_descriptor_heap(Device* device, uint32 capacity) noexcept;
void destroy_sampler_descriptor_heap(SamplerDescriptorHeap* heap) noexcept;
void write_sampler_descriptor(SamplerDescriptorHeap* heap, uint32 index, const SamplerDesc& desc = {}) noexcept;
void copy_sampler_descriptors(const SamplerDescriptorHeap* source, uint32 source_index, SamplerDescriptorHeap* destination,
                              uint32 destination_index, uint32 count) noexcept;

[[nodiscard]] PSO* create_graphics_pso(Device* device, const GraphicsPSODesc& desc) noexcept;
[[nodiscard]] PSO* create_mesh_pso(Device* device, const MeshPSODesc& desc) noexcept;
[[nodiscard]] PSO* create_compute_pso(Device* device, const ShaderStage& compute) noexcept;
void destroy_pso(PSO* pso) noexcept;

// Pools retain command storage until destruction. Reset only after every submitted buffer from this pool completes; unsubmitted buffers are discarded.
// Reset invalidates all previously returned CommandBuffer handles. Use one pool per worker and in-flight frame for independent recording/reuse.
// Buffers from a pool must be submitted to the selected queue's family.
[[nodiscard]] CommandPool* create_command_pool(Device* device, uint32 queue_index = 0) noexcept;
void destroy_command_pool(CommandPool* pool) noexcept;
void reset_command_pool(CommandPool* pool) noexcept;
[[nodiscard]] CommandBuffer* begin_commands(CommandPool* pool) noexcept;
void end_commands(CommandBuffer* commands) noexcept;
// Submit any ended subset exactly once before pool reset. Order completion semaphore signal values across queues.
// queue_index must be less than DeviceCaps::queue_count; omitted selects queue zero.
void submit(Device* device, const SubmitDesc& desc, uint32 queue_index = 0) noexcept;

void set_texture_descriptor_heap(CommandBuffer* commands, TextureDescriptorHeap* heap) noexcept;
void set_sampler_descriptor_heap(CommandBuffer* commands, SamplerDescriptorHeap* heap) noexcept;

// Initialize whole-texture metadata outside a render pass before first GPU use and before reuse after writes through an alias.
// Texture aliasing uses overlapping placements in one TextureHeap; no creation flag is required.
// Synchronize prior storage accesses and make their writes available with barriers or timeline waits first; these do not initialize metadata.
// Discards contents; clear or overwrite before reading. Orders initialization before subsequent texture accesses on this queue.
void initialize_texture(CommandBuffer* commands, Texture* texture) noexcept;
// An external texture (create_external_texture) at the start of its use in a command buffer: it discards the contents (the image
// may be in any layout, such as an OpenXR swapchain image after xrWaitSwapchainImage) and orders the transition after every
// earlier command of the queue.
void begin_external_access(CommandBuffer* commands, Texture* texture) noexcept;
// Its end: after every earlier command of the queue, the image goes to the color attachment layout (the depth-stencil attachment
// layout for a depth format), the layout OpenXR requires at xrReleaseSwapchainImage. Record nothing on the texture after it.
void end_external_access(CommandBuffer* commands, Texture* texture) noexcept;

// Command GPU ranges must lie within one GPU heap.
void copy_memory(CommandBuffer* commands, GpuRange source, GpuRange destination) noexcept;
// Depth/stencil copies require a general queue. Copy-only queues also require DeviceCaps::copy_texture_granularity alignment.
void copy_memory_to_texture(CommandBuffer* commands, GpuRange source, Texture* destination, const TextureCopyDesc& copy = {}) noexcept;
void copy_texture_to_memory(CommandBuffer* commands, Texture* source, GpuRange destination, const TextureCopyDesc& copy = {}) noexcept;

void barrier(CommandBuffer* commands, Stage before, Access before_access, Stage after, Access after_access) noexcept;

// Up to DeviceDesc::timestamp_query_count markers per command buffer. stage must map to a single GPU pipeline stage.
// Ignored, leaving the destination unchanged, when timestamps are disabled or the queue lacks profiling support.
// Destinations are ordinary CPU memory and must remain valid and distinct until read_timestamps retrieves the results.
void write_timestamp(CommandBuffer* commands, uint64* cpu_destination, Stage stage = Stage::all_commands) noexcept;
// Retrieve unread timestamps from submitted buffers after every submission from this pool completes. Does not wait.
// Call before resetting the pool; reset discards unread results. Unsubmitted buffers are ignored.
void read_timestamps(CommandPool* pool) noexcept;

// The host clock of CalibratedTimestamps::host: QueryPerformanceCounter counts (Windows) or CLOCK_MONOTONIC nanoseconds
enum class HostTimeDomain : uint8
{
    none,
    query_performance_counter,
    clock_monotonic,
    clock_monotonic_raw, // macOS: nanoseconds of CLOCK_MONOTONIC_RAW (it counts sleep, unlike the Mach absolute time)
};

// A GPU timestamp (the ticks write_timestamp gives) and the host clock, sampled together (VK_KHR_calibrated_timestamps)
struct CalibratedTimestamps
{
    uint64 gpu = 0;
    uint64 host = 0;
    uint64 max_deviation_ns = 0; // the sampling interval of the two values, as the driver reports it
    HostTimeDomain host_domain = HostTimeDomain::none;
};

// False, leaving out empty, when the device has no calibrated timestamps (Vulkan without the extension).
// Tools align GPU timestamps to the CPU timeline with it: host time = host + (ticks - gpu) * timestamp_period_ns.
[[nodiscard]] bool get_calibrated_timestamps(Device* device, CalibratedTimestamps& out) noexcept;

// A named region of commands for GPU tools (VK_EXT_debug_utils labels: Nsight, RenderDoc). No effect without the extension.
// Pairs nest and must close in the same command buffer.
void begin_debug_label(CommandBuffer* commands, const char* name) noexcept;
void end_debug_label(CommandBuffer* commands) noexcept;

// Every render pass starts with no graphics PSO. Bind one inside the pass before drawing, including every resumed segment.
// Render-pass boundaries invalidate the current PSO selection; rebind a compute PSO before dispatching after a render pass.
// Each segment needs matching attachments, load/store operations, and clear values, and its own begin/end_render_pass pair.
// Submit the complete suspend/resume chain in order in one batch. No action or synchronization commands may occur between segments.
// The first/last segments apply the requested load/store operations.
// Command-buffer bindings are not inherited.
void begin_render_pass(CommandBuffer* commands, const RenderingDesc& desc, RenderingFlags flags = RenderingFlags::none) noexcept;
void end_render_pass(CommandBuffer* commands) noexcept;

// begin_render_pass resets a full render-area viewport and scissor and disables depth/stencil; these commands override those defaults until the next pass
void set_viewport(CommandBuffer* commands, const Viewport& viewport) noexcept;
void set_scissor(CommandBuffer* commands, const Scissor& scissor) noexcept;
void set_depth_stencil(CommandBuffer* commands, const DepthStencilState& state) noexcept;

// Bind graphics/mesh PSOs inside an active render pass; bind compute PSOs outside render passes.
void bind_pso(CommandBuffer* commands, const PSO* pso) noexcept;

// root uses the shared constant-buffer layout and a 16-byte-aligned GPU address, or nullptr for rootless shaders.
// Retain its storage through GPU completion.
// Finish CPU writes before submission; wait for prior GPU users before overwriting. Synchronize GPU writes before consuming the root.
// Storage is application-owned; these commands do not copy root bytes or allocate memory.
void draw(CommandBuffer* commands, const void* root, uint32 vertex_count, uint32 instance_count = 1, uint32 first_vertex = 0,
          uint32 first_instance = 0) noexcept;
void draw_indexed(CommandBuffer* commands, const void* root, GpuRange indices, IndexType type, uint32 index_count, uint32 instance_count = 1,
                  uint32 first_index = 0, int32 vertex_offset = 0, uint32 first_instance = 0) noexcept;
void draw_indirect(CommandBuffer* commands, const void* root, GpuRange arguments, uint32 draw_count = 1, uint32 stride = 0) noexcept;
void draw_indexed_indirect(CommandBuffer* commands, const void* root, GpuRange indices, IndexType type, GpuRange arguments, uint32 draw_count = 1,
                           uint32 stride = 0) noexcept;
void dispatch(CommandBuffer* commands, const void* root, uint32x3 group_count) noexcept;
void dispatch_indirect(CommandBuffer* commands, const void* root, GpuRange arguments) noexcept;
void draw_meshlets(CommandBuffer* commands, const void* root, uint32x3 group_count) noexcept;
// Requires DeviceCaps::indirect_mesh_draw. Direct task/mesh draws remain available on every supported device.
void draw_meshlets_indirect(CommandBuffer* commands, const void* root, GpuRange arguments, uint32 draw_count = 1, uint32 stride = 0) noexcept;

// Ray tracing: acceleration structures for inline ray queries (DeviceCaps::ray_query; Vulkan, and Metal 4 in compute).
// The structure lives in storage of its own, made by create_acceleration_structure; the shader receives its 64-bit address
// (get_acceleration_structure_address: Vulkan's device address, Metal's MTLResourceID) in its root data, like any GPU pointer, and
// makes the handle from it (shader.slang: gpu_acceleration_structure_from_id; SPIR-V OpConvertUToAccelerationStructureKHR).
// Metal reads a top level's instances at record time (build_acceleration_structure): put them in CPU-visible memory, written by the
// CPU before the call (GPU-written instances would need a conversion kernel on Metal: not done).
// Build inputs and scratch are ordinary GPU heap memory: the heaps of a ray_query device can hold them.
enum class AccelerationStructureType : uint8
{
    bottom_level,
    top_level,
};

// Bottom level: axis-aligned boxes, six floats each (min x, y, z, max x, y, z) every stride bytes (8-byte aligned).
// A ray query sees each box as a procedural primitive candidate; its index is CandidatePrimitiveIndex().
struct AccelerationStructureAabbs
{
    GpuRange aabbs = {};
    uint32 count = 0;
    uint32 stride = 24;
    bool opaque = true; // Opaque boxes skip the any-hit-like candidate step of non-opaque ones (the shader still decides the hit).
};

// Top level instance, VkAccelerationStructureInstanceKHR's layout (64 bytes): a row-major 3x4 object-to-world transform,
// a 24-bit custom index (InstanceID()) and an 8-bit mask tested against the ray query's mask, a 24-bit hit group offset and
// 8 flags (unused by ray queries: zero), and the bottom level's address.
struct AccelerationStructureInstance
{
    float transform[3][4] = {{1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 0.0f}};
    uint32 custom_index_and_mask = 0xFF000000u; // index | mask << 24
    uint32 offset_and_flags = 0;
    uint64 acceleration_structure = 0;
};

struct AccelerationStructureDesc
{
    AccelerationStructureType type = AccelerationStructureType::bottom_level;
    AccelerationStructureAabbs aabbs = {}; // bottom_level
    GpuRange instances = {};               // top_level: AccelerationStructureInstance records, 16-byte aligned
    uint32 instance_count = 0;
    bool fast_build = false;               // Prefer build speed over trace speed (a structure rebuilt every frame).
};

struct AccelerationStructureSizes
{
    uint64 storage = 0; // create_acceleration_structure's size
    uint64 scratch = 0; // build_acceleration_structure's scratch, GPU heap memory (gpu_only), aligned to scratch_alignment
    uint64 scratch_alignment = 0;
};

// The sizes for a build with these counts (the GPU ranges may be empty): a structure sized for a count builds any smaller count.
[[nodiscard]] AccelerationStructureSizes get_acceleration_structure_sizes(Device* device, const AccelerationStructureDesc& desc) noexcept;
[[nodiscard]] AccelerationStructure* create_acceleration_structure(Device* device, AccelerationStructureType type, uint64 size) noexcept;
void destroy_acceleration_structure(AccelerationStructure* structure) noexcept;
[[nodiscard]] uint64 get_acceleration_structure_address(const AccelerationStructure* structure) noexcept;
// Records a full build outside render passes. Inputs and scratch must be written before (barrier to acceleration_structure_build);
// a ray query reads the result after a barrier from acceleration_structure_build / acceleration_structure_write.
void build_acceleration_structure(CommandBuffer* commands, AccelerationStructure* destination, const AccelerationStructureDesc& desc,
                                  GpuRange scratch) noexcept;

} // namespace gpu
