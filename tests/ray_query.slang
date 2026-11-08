#include <NoGraphicsAPI/shader.slang>
#include "ray_query_shared.h"

GPU_ROOT(RayQueryRoot, root);

// Each box is a procedural candidate: the shader commits the ray's entry into it (its slab test in object space: the instances are
// translated only, so the object ray is the world ray minus the translation, read from the candidate's object ray origin)
[shader("compute")]
[numthreads(ray_query_thread_count, 1, 1)]
void computeMain(uint3 thread : SV_DispatchThreadID)
{
    if (thread.x >= root.ray_count) return;
    RayQueryRay ray = root.rays[thread.x];
    RaytracingAccelerationStructure tlas = gpu_acceleration_structure_from_id((uint64_t(root.tlas[1]) << 32u) | uint64_t(root.tlas[0]));
    RayDesc desc;
    desc.Origin = float3(ray.origin[0], ray.origin[1], ray.origin[2]);
    desc.Direction = float3(ray.direction[0], ray.direction[1], ray.direction[2]);
    desc.TMin = ray.t_min;
    desc.TMax = ray.t_max;
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(tlas, RAY_FLAG_NONE, ray.mask, desc);
    float best = ray.t_max;
    while (query.Proceed())
    {
        if (query.CandidateType() != CANDIDATE_PROCEDURAL_PRIMITIVE) continue;
        const uint box = query.CandidatePrimitiveIndex() * 6u;
        const float3 lo = float3(root.boxes[box], root.boxes[box + 1u], root.boxes[box + 2u]);
        const float3 hi = float3(root.boxes[box + 3u], root.boxes[box + 4u], root.boxes[box + 5u]);
        const float3 o = query.CandidateObjectRayOrigin();
        const float3 d = query.CandidateObjectRayDirection();
        const float3 a = (lo - o) / d;
        const float3 b = (hi - o) / d;
        const float3 near3 = min(a, b);
        const float3 far3 = max(a, b);
        const float entry = max(max(max(near3.x, near3.y), near3.z), ray.t_min);
        const float exit = min(min(far3.x, far3.y), far3.z);
        if (entry <= exit && entry < best)
        {
            best = entry;
            query.CommitProceduralPrimitiveHit(entry);
        }
    }
    RayQueryHit hit = {};
    if (query.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT)
    {
        hit.status = 1u;
        hit.primitive = query.CommittedPrimitiveIndex();
        hit.instance = query.CommittedInstanceID();
        hit.t = query.CommittedRayT();
    }
    root.hits[thread.x] = hit;
}
