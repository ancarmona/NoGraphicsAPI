#pragma once

#define GPU_ROOT_SELECT(_1, _2, _3, macro, ...) macro
#define GPU_ROOT(...) GPU_ROOT_SELECT(__VA_ARGS__, GPU_ROOT_IMPL, GPU_ROOT_DEFAULT)(__VA_ARGS__)

#ifdef NOGRAPHICSAPI_METAL

#define GPU_ROOT_DEFAULT(type, name) GPU_ROOT_IMPL(type, name, 16)
#define GPU_ROOT_IMPL(type, name, alignment) \
    [[vk::binding(0)]] ConstantBuffer<type> name##_storage : register(b0); \
    static type name = gpu_root_aligned_storage(name##_storage, alignment)
#define GPU_ADDRESS(value) __getAddress(value)

[ForceInline]
ConstantBuffer<T> gpu_root_aligned_storage<T>(ConstantBuffer<T> storage, constexpr int alignment)
{
    static_assert(sizeof(T) % 16 == 0, "Root structs require a 16-byte stride and native vector alignment; pointed-to data uses C layout");
    __intrinsic_asm "(__builtin_assume((reinterpret_cast<ulong>($0) & ($1 - 1)) == 0), $0)";
}

[[vk::binding(1)]] ConstantBuffer<uint64_t> gpu_texture_pool_base : register(b1);

// Metal requires a struct around resource elements in an argument buffer; each sampler still occupies exactly eight bytes.
__target_intrinsic(metal, "GPUSampler")
struct GPUSampler { uint64_t value; };
[[vk::binding(2)]] StructuredBuffer<GPUSampler> gpu_sampler_heap : register(t2);

T gpu_sampler_value<T : IOpaqueDescriptor>(GPUSampler value)
{
    __requirePrelude(R"(struct GPUSampler { sampler value; }; static_assert(sizeof(GPUSampler) == 8);)");
    __intrinsic_asm "($0.value)";
}

T gpu_texture_from_id<T : IOpaqueDescriptor>(uint64_t resource_id)
{
    __requirePrelude(R"(template<typename T> T gpu_texture_from_id(ulong id) { return *reinterpret_cast<thread T*>(&id); })");
    __intrinsic_asm "gpu_texture_from_id<$TR>($0)";
}

[ForceInline]
T gpu_texture<T : IOpaqueDescriptor>(uint index)
{
    return gpu_texture_from_id<T>(gpu_texture_pool_base + uint64_t(index));
}

[ForceInline]
T gpu_sampler<T : IOpaqueDescriptor = SamplerState>(uint index)
{
    return gpu_sampler_value<T>(gpu_sampler_heap[index]);
}

#else

#define GPU_ROOT_DEFAULT(type, name) GPU_ROOT_IMPL(type, name, 16)
#define GPU_ROOT_IMPL(type, name, alignment) [[vk::binding(0, 0)]] ConstantBuffer<type> name
#define GPU_ADDRESS(value) (&(value))

[ForceInline]
T gpu_texture<T : IOpaqueDescriptor>(uint index)
{
    return getDescriptorFromHandle(DescriptorHandle<T>(uint2(index, 0)));
}

[ForceInline]
T gpu_sampler<T : IOpaqueDescriptor = SamplerState>(uint index)
{
    return getDescriptorFromHandle(DescriptorHandle<T>(uint2(index, 0)));
}

#endif

// An acceleration structure from get_acceleration_structure_address's value (DeviceCaps::ray_query). Vulkan: the structure's device
// address (OpConvertUToAccelerationStructureKHR). Metal: its MTLResourceID, whose 8 bytes are the handle (as gpu_texture_from_id for
// textures; Slang has no RaytracingAccelerationStructure(address) for Metal)
[ForceInline]
RaytracingAccelerationStructure gpu_acceleration_structure_from_id(uint64_t id)
{
    __target_switch
    {
    case metal:
        __requirePrelude(R"(template<typename T> T gpu_texture_from_id(ulong id) { return *reinterpret_cast<thread T*>(&id); })");
        __intrinsic_asm "gpu_texture_from_id<metal::raytracing::instance_acceleration_structure>($0)";
    default:
        return RaytracingAccelerationStructure(id);
    }
}

// Multisample textures (Vulkan and Metal). Load(int2 pixel, int sample_index) reads one sample; Load(int3(pixel, layer), sample_index) for arrays.
[ForceInline]
Texture2DMS<T> gpu_texture_ms<T : ITexelElement>(uint index)
{
    return gpu_texture<Texture2DMS<T>>(index);
}

[ForceInline]
Texture2DMSArray<T> gpu_texture_ms_array<T : ITexelElement>(uint index)
{
    return gpu_texture<Texture2DMSArray<T>>(index);
}

// One sample of a multisample array: Load(int3(pixel, layer), sample). Slang 2026.20 emits int coordinates for Metal's
// texture2d_ms_array::read, which takes only unsigned ones, so Metal reads through its own call
[ForceInline]
float gpu_load_ms_array(Texture2DMSArray<float> texture, int3 location, int sample_index)
{
    __target_switch
    {
    case metal: __intrinsic_asm "$0.read(uint2($1.xy), uint($1.z), uint($2)).x";
    default: return texture.Load(location, sample_index);
    }
}

[ForceInline]
float4 gpu_load_ms_array(Texture2DMSArray<float4> texture, int3 location, int sample_index)
{
    __target_switch
    {
    case metal: __intrinsic_asm "$0.read(uint2($1.xy), uint($1.z), uint($2))";
    default: return texture.Load(location, sample_index);
    }
}

[ForceInline]
uint gpu_nonuniform_index(uint index)
{
    __target_switch
    {
    case metal: return index;
    default: return NonUniformResourceIndex(index);
    }
}

[ForceInline]
uint gpu_wave_prefix_count_bits(bool value)
{
    __target_switch
    {
    case metal: __intrinsic_asm "simd_prefix_exclusive_sum(uint($0))";
    default: return WavePrefixCountBits(value);
    }
}
