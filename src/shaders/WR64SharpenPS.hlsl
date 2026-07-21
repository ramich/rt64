//
// RT64 - WR64 fork
//
// Contrast-adaptive sharpen (CAS-lite): 5-tap unsharp mask clamped to the
// local neighborhood min/max so edges crispen without ringing halos.
// Reads a copy of the presented frame and overwrites the swap chain.

#include "shared/rt64_wr64_sharpen.h"

[[vk::push_constant]] ConstantBuffer<WR64SharpenCB> gConstants : register(b0);

Texture2D<float4> gInput : register(t1);

float3 Tap(int2 p) {
    int2 maxP = int2(gConstants.texSize) - 1;
    return gInput.Load(int3(clamp(p, int2(0, 0), maxP), 0)).rgb;
}

float4 PSMain(in float4 pos : SV_Position, in float2 uv : TEXCOORD0) : SV_TARGET {
    int2 p = int2(uv * gConstants.texSize);
    float3 c = Tap(p);
    float3 n = Tap(p + int2(0, -1));
    float3 s = Tap(p + int2(0, 1));
    float3 e = Tap(p + int2(1, 0));
    float3 w = Tap(p + int2(-1, 0));

    float3 mn = min(c, min(min(n, s), min(e, w)));
    float3 mx = max(c, max(max(n, s), max(e, w)));

    float3 sharp = c + (4.0f * c - n - s - e - w) * 0.25f * gConstants.strength;
    return float4(clamp(sharp, mn, mx), 1.0f);
}
