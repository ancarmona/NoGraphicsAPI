#pragma once

#include <NoGraphicsAPIUtility/shader_types.h>

// test_ray_query: rays against a top level of two instances of one bottom level of boxes
struct RayQueryRay
{
    float origin[3];
    float t_min;
    float direction[3];
    float t_max;
    uint32 mask;
    uint32 padding[3];
};

struct RayQueryHit
{
    uint32 status;    // 0 a miss, 1 a box committed
    uint32 primitive; // CommittedPrimitiveIndex
    uint32 instance;  // CommittedInstanceID: the instance's custom index
    float t;
};

struct RayQueryRoot
{
    float* boxes; // the bottom level's boxes: min x, y, z, max x, y, z
    RayQueryRay* rays;
    RayQueryHit* hits;
    uint32 tlas[2]; // get_acceleration_structure_address, low and high words
    uint32 ray_count;
    uint32 root_alignment_padding0[3];
};

static const uint32 ray_query_thread_count = 64;

#if !defined(__SLANG__)
static_assert(sizeof(RayQueryRay) == 48);
static_assert(sizeof(RayQueryHit) == 16);
static_assert(sizeof(RayQueryRoot) == 48);
#endif
