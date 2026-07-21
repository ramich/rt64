//
// RT64 - WR64 fork
//

#pragma once

#include "shared/rt64_hlsl.h"

#ifdef HLSL_CPU
namespace interop {
#endif
    // Push constants for the WR64 motion-blur compose pass (previous presented
    // frame drawn over the current one with constant alpha = blur strength).
    struct WR64MotionBlurCB {
        float2 uvScale;
        float alpha;
        float padding;
    };
#ifdef HLSL_CPU
};
#endif
