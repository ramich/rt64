//
// RT64 - WR64 fork
//
// Motion-blur compose (prototype): draws the previous presented frame over the
// current one; the pipeline uses standard alpha blending and the constant
// alpha below sets the accumulation strength (exponential trail).

#include "shared/rt64_wr64_motion_blur.h"

[[vk::push_constant]] ConstantBuffer<WR64MotionBlurCB> gConstants : register(b0);

Texture2D<float4> gInput : register(t1);

float4 PSMain(in float4 pos : SV_Position, in float2 uv : TEXCOORD0) : SV_TARGET {
    uint2 pixelPos = uv.xy * gConstants.uvScale;
    float4 color = gInput.Load(uint3(pixelPos, 0));
    color.a = gConstants.alpha;
    return color;
}
