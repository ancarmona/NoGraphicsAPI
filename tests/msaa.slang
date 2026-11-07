#include <NoGraphicsAPI/shader.slang>
#include "msaa_shared.h"

GPU_ROOT(MsaaDrawRoot, root);

static const uint msaa_corners[6] = {0, 1, 2, 2, 1, 3};

[shader("vertex")]
float4 vertexMain(uint vertex_index : SV_VertexID) : SV_Position
{
    uint corner = msaa_corners[vertex_index];
    float right = root.full_screen != 0 ? 1.0f : (float(msaa_test_edge) + 0.5f) / float(msaa_test_size) * 2.0f - 1.0f;
    return float4((corner & 1) != 0 ? right : -1.0f, (corner & 2) != 0 ? 1.0f : -1.0f, 0.25f, 1.0f);
}

struct MsaaTargets
{
    float4 color : SV_Target0;
    uint id : SV_Target1;
    uint2 id2 : SV_Target2;
};

[shader("fragment")]
MsaaTargets fragmentMain()
{
    MsaaTargets output;
    output.color = root.color;
    output.id = msaa_test_id;
    output.id2 = uint2(msaa_test_id2_x, msaa_test_id2_y);
    return output;
}
