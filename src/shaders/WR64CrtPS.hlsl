//
// RT64 - WR64 fork
//
// CRT filter (Sony Trinitron look): aperture grille (vertical RGB phosphor
// stripes), scanlines locked to the game's visible SOURCE rows, slight
// barrel curvature with rounded corners, mild vignette, and brightness
// compensation for the light the masks absorb. Single pass over a copy of
// the presented frame; everything scales with gConstants.intensity.
//
// The effect maps onto the game content rectangle ("tube", rectMin..rectMax
// in swapchain pixels) — pixels outside pass through untouched, so pillarbox
// bars stay flat black and a real-CRT feel is kept in every Border Area /
// Aspect Ratio combination (2P split-screen = one tube, like the console).
//
// Resolution-adaptive: the grille pitch derives from the tube height per
// source row (dot pitch tracks scanline pitch), staying >= 3 output pixels
// (one RGB triad); below ~3 px per scanline/triad both masks fade out so
// small windows degrade to curvature+vignette instead of moire.

#include "shared/rt64_wr64_crt.h"

[[vk::push_constant]] ConstantBuffer<WR64CrtCB> gConstants : register(b0);

Texture2D<float4> gInput : register(t1);
SamplerState gSampler : register(s2);

float4 PSMain(in float4 pos : SV_Position, in float2 uv : TEXCOORD0) : SV_TARGET {
    const float2 pixel = uv * gConstants.texSize;
    const float2 rectMin = gConstants.rectMin;
    const float2 rectMax = gConstants.rectMax;
    const float2 tubeSize = rectMax - rectMin;
    const float k = gConstants.intensity;

    // Outside the tube (black bars), or degenerate rect: pass through.
    if (tubeSize.x < 8.0f || tubeSize.y < 8.0f ||
        pixel.x < rectMin.x || pixel.x >= rectMax.x ||
        pixel.y < rectMin.y || pixel.y >= rectMax.y) {
        return float4(gInput.SampleLevel(gSampler, uv, 0).rgb, 1.0f);
    }

    // Tube coordinates, centered: c in [-1,1] across the content rect.
    const float2 tubeUV = (pixel - rectMin) / tubeSize;
    float2 c = tubeUV * 2.0f - 1.0f;

    // Barrel distortion: map the (flat) output pixel to the (bulged) source
    // position. Edges/corners sample outside the image and go black, giving
    // the curved-glass silhouette.
    const float curve = 0.045f * k;
    const float r2 = dot(c, c);
    float2 d = c * (1.0f + curve * r2);

    // Rounded corners on the distorted tube: signed distance to a rounded
    // box, smoothed over ~2 output pixels.
    const float cornerR = 0.02f + 0.06f * k;
    const float2 q = abs(d) - (1.0f - cornerR);
    const float cornerDist = length(max(q, 0.0f)) + min(max(q.x, q.y), 0.0f) - cornerR;
    const float edgeW = 2.0f / min(tubeSize.x, tubeSize.y);
    const float tubeMask = 1.0f - smoothstep(-edgeW, edgeW, cornerDist);

    // Sample the source at the distorted position (clamped just inside so
    // the border-color sampler never bleeds).
    const float2 srcTube = d * 0.5f + 0.5f;
    const float2 srcPixel = rectMin + saturate(srcTube) * tubeSize;
    float3 color = gInput.SampleLevel(gSampler, srcPixel / gConstants.texSize, 0).rgb;

    // Resolution adaptivity: output pixels per source scanline drive both
    // the mask scale and the fade-out guard against moire.
    const float pxPerRow = tubeSize.y / max(gConstants.srcRows, 1.0f);
    const float maskFade = smoothstep(2.0f, 3.0f, pxPerRow);

    // Aperture grille: vertical RGB triads. Pitch tracks the scanline pitch
    // (Trinitron dot pitch ~ line pitch) but never drops below one triad =
    // 3 output pixels.
    const float grillePitch = max(3.0f, floor(pxPerRow * 0.75f + 0.5f));
    const float grilleStrength = 0.40f * k * maskFade;
    const float band = floor(frac(pixel.x / grillePitch) * 3.0f);
    float3 grille = float3(band == 0.0f ? 1.0f : 1.0f - grilleStrength,
                           band == 1.0f ? 1.0f : 1.0f - grilleStrength,
                           band == 2.0f ? 1.0f : 1.0f - grilleStrength);
    color *= grille;

    // Scanlines over the SOURCE rows (mapped through the distortion so they
    // curve with the image), raised-cosine profile.
    const float scanStrength = 0.35f * k * maskFade;
    const float lineCoord = srcTube.y * gConstants.srcRows;
    const float scan = 1.0f - scanStrength * (0.5f - 0.5f * cos(6.28318530718f * lineCoord));
    color *= scan;

    // Mild vignette toward the tube corners.
    color *= 1.0f - 0.12f * k * r2 * r2;

    // Brightness compensation: the grille eats grilleStrength on 2 of 3
    // channels, the scanlines eat scanStrength/2 on average.
    const float gain = 1.0f / max((1.0f - grilleStrength * 2.0f / 3.0f) *
                                  (1.0f - scanStrength * 0.5f), 0.55f);
    color = saturate(color * min(gain, 1.35f));

    return float4(color * tubeMask, 1.0f);
}
