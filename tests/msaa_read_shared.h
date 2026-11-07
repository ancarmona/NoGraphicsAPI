#pragma once

#include "msaa_common.h"

struct MsaaReadRoot
{
    MsaaSample* output;
    uint32 color_texture;
    uint32 id_texture;
    uint32 id2_texture;
    uint32 depth_texture;
    uint32 root_alignment_padding0[2];
};

#if !defined(__SLANG__)
static_assert(sizeof(MsaaSample) == 32 && sizeof(MsaaReadRoot) == 32);
#endif
