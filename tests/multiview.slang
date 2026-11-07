#include <NoGraphicsAPI/shader.slang>
#include "multiview_common.h"

// View 0 covers the left half of layer 0 and view 1 the right half of layer 1; the color also comes from the view index.
static const uint multiview_corners[6] = {0, 1, 2, 2, 1, 3};

float4 multiview_position(uint corner, uint view)
{
    float left = view == 0 ? -1.0f : 0.0f;
    return float4((corner & 1) != 0 ? left + 1.0f : left, (corner & 2) != 0 ? 1.0f : -1.0f, 0.5f, 1.0f);
}

[shader("vertex")]
float4 vertexMain(uint vertex_index : SV_VertexID, uint view : SV_ViewID) : SV_Position
{
    return multiview_position(multiview_corners[vertex_index], view);
}

struct MultiviewVertex
{
    float4 position : SV_Position;
};

// Topology is view-independent; only the position depends on SV_ViewID, as VK_EXT_mesh_shader multiview requires.
[shader("mesh")]
[outputtopology("triangle")]
[numthreads(4, 1, 1)]
void meshMain(uint lane : SV_GroupIndex, uint view : SV_ViewID, out vertices MultiviewVertex vertices[4], out indices uint3 triangles[2])
{
    SetMeshOutputCounts(4, 2);
    MultiviewVertex vertex;
    vertex.position = multiview_position(lane, view);
    vertices[lane] = vertex;
    if (lane < 2) triangles[lane] = lane == 0 ? uint3(0, 1, 2) : uint3(2, 1, 3);
}

[shader("fragment")]
float4 fragmentMain(uint view : SV_ViewID) : SV_Target0
{
    return view == 0 ? float4(1.0f, 0.0f, 0.0f, 1.0f) : float4(0.0f, 1.0f, 0.0f, 1.0f);
}
