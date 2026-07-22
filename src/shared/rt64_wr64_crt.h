//
// RT64 - WR64 fork
//

#pragma once

#include "shared/rt64_hlsl.h"

#ifdef HLSL_CPU
namespace interop {
#endif
    // Push constants for the WR64 CRT (Trinitron-style) present pass.
    // rectMin/rectMax bound the game content rectangle ("tube") in swapchain
    // pixels — curvature/corners map onto it, pixels outside pass through.
    // srcRows = visible source scanlines (198/240), so the scanline pitch
    // tracks the game's real lines instead of output pixels.
    struct WR64CrtCB {
        float2 rectMin;
        float2 rectMax;
        float2 texSize;
        float srcRows;
        float intensity;
    };

    // Push constants for the phosphor-glow prepass (bright-pass tent blur
    // into a quarter-res target the CRT shader adds back as halation).
    struct WR64GlowCB {
        float2 srcSize;
        float threshold;
        float padding;
    };
#ifdef HLSL_CPU
};
#endif
