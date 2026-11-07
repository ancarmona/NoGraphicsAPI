#pragma once

#include "msaa_common.h"

struct MsaaDrawRoot
{
    float4 color;
    uint32 full_screen;
    uint32 root_alignment_padding0[3];
};

#if !defined(__SLANG__)
static_assert(sizeof(MsaaDrawRoot) == 32);
#endif
