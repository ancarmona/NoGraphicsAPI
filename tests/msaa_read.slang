#include <NoGraphicsAPI/shader.slang>
#include "msaa_read_shared.h"

GPU_ROOT(MsaaReadRoot, root);

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 thread : SV_DispatchThreadID)
{
    if (thread.x >= msaa_test_size || thread.y >= msaa_test_size) return;
    Texture2DMS<float4> color = gpu_texture_ms<float4>(root.color_texture);
    Texture2DMS<uint> id = gpu_texture_ms<uint>(root.id_texture);
    Texture2DMS<uint2> id2 = gpu_texture_ms<uint2>(root.id2_texture);
    Texture2DMS<float> depth = gpu_texture_ms<float>(root.depth_texture);
    for (uint sample_index = 0; sample_index < msaa_test_samples; ++sample_index)
    {
        MsaaSample value;
        value.color = color.Load(int2(thread.xy), sample_index);
        value.id = id.Load(int2(thread.xy), sample_index);
        uint2 pair = id2.Load(int2(thread.xy), sample_index);
        value.id2_x = pair.x;
        value.id2_y = pair.y;
        value.depth = depth.Load(int2(thread.xy), sample_index);
        root.output[(thread.y * msaa_test_size + thread.x) * msaa_test_samples + sample_index] = value;
    }
}
