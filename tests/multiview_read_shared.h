#pragma once

#include "multiview_common.h"

struct MultiviewReadRoot
{
    float4* output;
    uint32 texture;
    uint32 root_alignment_padding0[1];
};

#if !defined(__SLANG__)
static_assert(sizeof(MultiviewReadRoot) == 16);
#endif
