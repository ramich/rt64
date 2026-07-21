//
// RT64 - WR64 fork
//

#pragma once

#include "shared/rt64_hlsl.h"

#ifdef HLSL_CPU
namespace interop {
#endif
    // Push constants for the WR64 contrast-adaptive sharpen pass.
    struct WR64SharpenCB {
        float2 texSize;
        float strength;
        float padding;
    };
#ifdef HLSL_CPU
};
#endif
