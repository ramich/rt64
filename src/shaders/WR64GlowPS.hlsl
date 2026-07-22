//
// RT64 - WR64 fork
//
// Phosphor-glow prepass for the CRT filter: bright-pass + 9-tap tent blur
// of the presented frame into a quarter-resolution target. The CRT shader
// samples it (bilinearly, so the effective blur is wide) and adds it back
// as halation — the soft light bleed of bright phosphors through the glass.

#include "shared/rt64_wr64_crt.h"

[[vk::push_constant]] ConstantBuffer<WR64GlowCB> gConstants : register(b0);

Texture2D<float4> gInput : register(t1);
SamplerState gSampler : register(s2);

float3 BrightTap(float2 uv, float2 offset) {
    float3 c = gInput.SampleLevel(gSampler, uv + offset, 0).rgb;
    // Soft knee around the threshold: only bright content glows.
    float lum = dot(c, float3(0.299f, 0.587f, 0.114f));
    float knee = saturate((lum - gConstants.threshold) / max(1.0f - gConstants.threshold, 1e-3f));
    return c * knee * knee;
}

float4 PSMain(in float4 pos : SV_Position, in float2 uv : TEXCOORD0) : SV_TARGET {
    // One glow texel covers ~4 source pixels; tent taps at that spacing plus
    // the bilinear fetches give a cheap, wide, stable blur.
    const float2 o = 4.0f / gConstants.srcSize;
    float3 acc = BrightTap(uv, float2(0.0f, 0.0f)) * 4.0f;
    acc += BrightTap(uv, float2( o.x, 0.0f)) * 2.0f;
    acc += BrightTap(uv, float2(-o.x, 0.0f)) * 2.0f;
    acc += BrightTap(uv, float2(0.0f,  o.y)) * 2.0f;
    acc += BrightTap(uv, float2(0.0f, -o.y)) * 2.0f;
    acc += BrightTap(uv, float2( o.x,  o.y));
    acc += BrightTap(uv, float2(-o.x,  o.y));
    acc += BrightTap(uv, float2( o.x, -o.y));
    acc += BrightTap(uv, float2(-o.x, -o.y));
    return float4(acc / 16.0f, 1.0f);
}
