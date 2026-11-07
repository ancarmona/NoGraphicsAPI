#include <NoGraphicsAPI/shader.slang>
#include "multiview_read_shared.h"

GPU_ROOT(MultiviewReadRoot, root);

[shader("compute")]
[numthreads(8, 8, 1)]
void computeMain(uint3 thread : SV_DispatchThreadID)
{
    if (thread.x >= multiview_test_size || thread.y >= multiview_test_size || thread.z >= multiview_test_views) return;
    Texture2DMSArray<float4> texture = gpu_texture_ms_array<float4>(root.texture);
    for (uint sample_index = 0; sample_index < multiview_test_samples; ++sample_index)
    {
        uint index = ((thread.z * multiview_test_size + thread.y) * multiview_test_size + thread.x) * multiview_test_samples + sample_index;
        root.output[index] = texture.Load(int3(thread), sample_index);
    }
}
