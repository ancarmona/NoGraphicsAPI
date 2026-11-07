#pragma once

#include <NoGraphicsAPIUtility/shader_types.h>

static const uint32 msaa_test_size = 16;
static const uint32 msaa_test_samples = 4;
static const uint32 msaa_test_edge = 8; // The default rectangle ends at x = msaa_test_edge + 0.5.
static const uint32 msaa_test_id = 0x1234u;
static const uint32 msaa_test_id2_x = 0xabcdu;
static const uint32 msaa_test_id2_y = 0x5678u;

struct MsaaSample
{
    float4 color;
    uint32 id;
    uint32 id2_x;
    uint32 id2_y;
    float depth;
};
