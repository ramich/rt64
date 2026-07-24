//
// RT64 - WR64 fork
//
// CRT filter (Sony Trinitron look) + image-based bezel overlay.
//
// CRT part: aperture grille (vertical RGB phosphor stripes), scanlines locked
// to the game's visible SOURCE rows, slight barrel curvature with rounded
// corners, mild vignette, P22 phosphor colour and inline halation. Everything
// scales with gConstants.intensity; at intensity 0 the game passes through
// untouched (only the bezel, if enabled, still frames it).
//
// Bezel part: instead of drawing a procedural plastic frame (which never read
// as a real TV housing), the presented image is shrunk into an inset "glass"
// rect and a pre-authored bezel PNG (gBezel) is composited over the whole
// presented area. The PNG's transparent cutout sits exactly where the glass
// is — the cutout inset MUST match INSET_X / INSET_Y here and in the asset
// generator (scratchpad make_bezel.py). RetroArch/Mega-Bezel principle.
//
// The effect maps onto the game content rectangle ("tube", rectMin..rectMax in
// swapchain pixels) published by the VI renderer; pixels outside pass through,
// so pillarbox bars stay flat black and 2P split-screen = one tube.
//
// Resolution-adaptive: the grille pitch derives from the tube height per source
// row (dot pitch tracks scanline pitch), staying >= 3 output pixels; below ~3px
// per scanline/triad both masks fade out so small windows degrade to
// curvature+vignette instead of moire.

#include "shared/rt64_wr64_crt.h"

[[vk::push_constant]] ConstantBuffer<WR64CrtCB> gConstants : register(b0);

Texture2D<float4> gInput : register(t1);
Texture2D<float4> gBezel : register(t2);   // bezel overlay image (RGBA)
Texture2D<float4> gRefl  : register(t3);   // 1/32 box-downsampled frame (reflection source)
SamplerState gSampler : register(s4);

// Cutout inset of the bezel PNG as a fraction of the presented rect per axis.
// SHARED with the asset generator (scratchpad make_bezel.py). The game is
// shrunk into this inset "glass" so it lines up with the hole in the frame.
static const float INSET_X = 0.026f;
static const float INSET_Y = 0.035f;

// Phosphor halation computed INLINE from the frame (no separate glow render
// target — that second target retained stale memory across window resizes).
// A 2-ring set of wide, bright-passed taps through the linear sampler gives a
// soft bloom around highlights; each tap is bilinear so the effective blur is
// wider than the tap count suggests. Sampled in tube-content UV space.
float3 CrtHalation(float2 uv) {
    const float2 r1 = 5.0f / gConstants.texSize;
    const float2 r2 = 12.0f / gConstants.texSize;
    float3 acc = float3(0.0f, 0.0f, 0.0f);
    // inner ring (4) weight 2, outer ring (8) weight 1
    acc += gInput.SampleLevel(gSampler, uv + float2( r1.x, 0), 0).rgb * 2.0f;
    acc += gInput.SampleLevel(gSampler, uv + float2(-r1.x, 0), 0).rgb * 2.0f;
    acc += gInput.SampleLevel(gSampler, uv + float2(0,  r1.y), 0).rgb * 2.0f;
    acc += gInput.SampleLevel(gSampler, uv + float2(0, -r1.y), 0).rgb * 2.0f;
    acc += gInput.SampleLevel(gSampler, uv + float2( r2.x,  r2.y), 0).rgb;
    acc += gInput.SampleLevel(gSampler, uv + float2(-r2.x,  r2.y), 0).rgb;
    acc += gInput.SampleLevel(gSampler, uv + float2( r2.x, -r2.y), 0).rgb;
    acc += gInput.SampleLevel(gSampler, uv + float2(-r2.x, -r2.y), 0).rgb;
    acc += gInput.SampleLevel(gSampler, uv + float2( r2.x, 0), 0).rgb;
    acc += gInput.SampleLevel(gSampler, uv + float2(-r2.x, 0), 0).rgb;
    acc += gInput.SampleLevel(gSampler, uv + float2(0,  r2.y), 0).rgb;
    acc += gInput.SampleLevel(gSampler, uv + float2(0, -r2.y), 0).rgb;
    acc /= 16.0f;
    // Soft knee: only bright content blooms.
    const float lum = dot(acc, float3(0.299f, 0.587f, 0.114f));
    const float knee = saturate((lum - 0.60f) / 0.40f);
    return acc * knee * knee;
}

// Extra-smooth reflection colour: 3x3 box on the already-1/32 downsampled frame
// (cheap — tiny texture). Kills the residual per-edge variation so the bezel
// reflection reads as a near-uniform soft glow per side, not tracking content.
float3 SampleRefl(float2 uv) {
    const float2 s = 44.0f / gConstants.texSize;   // ~1.4 texels of the 1/32 image
    float3 a = float3(0.0f, 0.0f, 0.0f);
    [unroll] for (int i = -1; i <= 1; ++i) {
        [unroll] for (int j = -1; j <= 1; ++j) {
            a += gRefl.SampleLevel(gSampler, uv + float2(i, j) * s, 0).rgb;
        }
    }
    return a / 9.0f;
}

float4 PSMain(in float4 pos : SV_Position, in float2 uv : TEXCOORD0) : SV_TARGET {
    const float2 pixel = uv * gConstants.texSize;
    const float2 fullMin = gConstants.rectMin;
    const float2 fullMax = gConstants.rectMax;
    const float2 fullSize = fullMax - fullMin;
    const float k = gConstants.intensity;

    // Degenerate rect: pass through.
    if (fullSize.x < 8.0f || fullSize.y < 8.0f) {
        return float4(gInput.SampleLevel(gSampler, uv, 0).rgb, 1.0f);
    }

    // When the bezel is on, shrink the picture into an inset "glass" rect so it
    // lands in the PNG's cutout; the reclaimed margin is where the frame image
    // sits. The inset is a FIXED fraction of the presented size per axis
    // (matches the asset), so it is stable across window size/aspect and is
    // independent of the CRT intensity slider.
    const bool bezelOn = gConstants.bezel > 0.5f;
    const float2 insetPx = bezelOn ? float2(INSET_X, INSET_Y) * fullSize : float2(0.0f, 0.0f);
    const float2 rectMin = fullMin + insetPx;
    const float2 rectMax = fullMax - insetPx;
    const float2 tubeSize = max(rectMax - rectMin, float2(8.0f, 8.0f));

    // Tube coordinates, centered: c in [-1,1] across the glass rect.
    const float2 tubeUV = (pixel - rectMin) / tubeSize;
    float2 c = tubeUV * 2.0f - 1.0f;

    // Barrel distortion (only with CRT intensity): map the flat output pixel to
    // the bulged source position. At k=0 this is the identity, so the bezel-only
    // mode is a clean linear shrink into the glass.
    const float curve = 0.045f * k;
    const float r2 = dot(c, c);
    // Normalised barrel: divide by (1+curve) so the edge midpoints stay put
    // (|c|=1 -> d==c). Only the shape bows; the tube — and thus the bezel frame
    // riding on it — keeps its size instead of the frame fattening as curvature
    // rises.
    float2 d = c * (1.0f + curve * r2) / (1.0f + curve);

    // Rounded-corner silhouette for the glass (keeps the CRT-only mode's rounded
    // tube; when the bezel is on, the PNG frame covers this region anyway). Small
    // radius => boxy corners. P is the pixel offset from the glass centre.
    const float2 halfPix = tubeSize * 0.5f;
    const float2 P = d * halfPix;
    const float rTight = 0.006f * min(halfPix.x, halfPix.y);
    const float2 qg = abs(P) - (halfPix - rTight);
    const float cornerDist = length(max(qg, 0.0f)) + min(max(qg.x, qg.y), 0.0f) - rTight;
    const float tubeMask = 1.0f - smoothstep(-2.0f, 2.0f, cornerDist);

    // Scale-to-fit: the WHOLE presented image maps into the (inset) glass, so
    // nothing is cropped. Sample the source at the distorted position.
    const float2 srcTube = d * 0.5f + 0.5f;
    const float2 srcPixel = fullMin + saturate(srcTube) * fullSize;
    // Clamp half a texel inside the content rect so the border sampler never
    // bleeds black at the edges/corners (that black bleed, poking through the
    // rounded cutout, was the "square" in the corners).
    const float2 srcUV = clamp(srcPixel, fullMin + 0.5f, fullMax - 0.5f) / gConstants.texSize;
    float3 color = gInput.SampleLevel(gSampler, srcUV, 0).rgb;

    // P22 phosphor color: mild channel crosstalk toward the CRT phosphor
    // primaries plus a touch of CRT gamma. Scaled with intensity.
    const float3 p22 = float3(
        dot(color, float3(0.955f, 0.040f, 0.005f)),
        dot(color, float3(0.030f, 0.960f, 0.010f)),
        dot(color, float3(0.015f, 0.030f, 0.955f)));
    color = lerp(color, p22, k);
    color = pow(saturate(color), lerp(1.0f, 1.10f, k));

    // Resolution adaptivity: output pixels per source scanline drive both the
    // mask scale and the fade-out guard against moire.
    const float pxPerRow = tubeSize.y / max(gConstants.srcRows, 1.0f);
    const float maskFade = smoothstep(2.0f, 3.0f, pxPerRow);

    // Aperture grille: vertical RGB triads. Pitch tracks the scanline pitch but
    // never drops below one triad = 3 output pixels.
    const float grillePitch = max(3.0f, floor(pxPerRow * 0.75f + 0.5f));
    const float grilleStrength = 0.40f * k * maskFade;
    const float band = floor(frac(pixel.x / grillePitch) * 3.0f);
    float3 grille = float3(band == 0.0f ? 1.0f : 1.0f - grilleStrength,
                           band == 1.0f ? 1.0f : 1.0f - grilleStrength,
                           band == 2.0f ? 1.0f : 1.0f - grilleStrength);
    color *= grille;

    // Scanlines over the SOURCE rows (curve with the image), raised-cosine.
    const float scanStrength = 0.35f * k * maskFade;
    const float lineCoord = srcTube.y * gConstants.srcRows;
    const float scan = 1.0f - scanStrength * (0.5f - 0.5f * cos(6.28318530718f * lineCoord));
    color *= scan;

    // Mild vignette toward the tube corners, capped.
    color *= 1.0f - min(0.12f * k * r2 * r2, 0.22f);

    // Phosphor glow / halation on top of the masked image.
    const float3 glow = CrtHalation(srcUV);
    color += glow * (0.45f * k);

    // Brightness compensation for the light the masks absorb.
    const float gain = 1.0f / max((1.0f - grilleStrength * 2.0f / 3.0f) *
                                  (1.0f - scanStrength * 0.5f), 0.55f);
    color = saturate(color * min(gain, 1.35f));

    // Glass silhouette. Bezel OFF: keep the rounded CRT tube. Bezel ON: the game
    // fills the inset rect (rectangle) and the PNG's rounded cutout frames it;
    // the game's sharp corner sits UNDER the frame and the UV clamp above keeps
    // it from bleeding black — so no rounded game mask is needed.
    float3 glass = bezelOn ? color : (color * tubeMask);

    // Bezel overlay: the PNG maps across the FULL presented rect. Src-over the
    // frame image on top of the game; the transparent cutout lets the glass
    // show, the opaque plastic covers the margin.
    float3 composited = glass;
    if (bezelOn) {
        // Sample the bezel at the SAME barrel-curved position as the game (via d)
        // so the frame's opening follows the CRT curvature exactly. With the CRT
        // filter off, curve is 0 -> d == c -> warpedPixel == pixel -> the frame
        // stays flat/straight (the approved bezel-only look).
        const float2 warpedPixel = rectMin + (d * 0.5f + 0.5f) * tubeSize;
        const float2 bezelUV = (warpedPixel - fullMin) / fullSize;
        const float4 b = gBezel.SampleLevel(gSampler, saturate(bezelUV), 0);

        // Screen reflection on the inner rim, in the SAME warped space so it
        // rides the curved frame. Soft blurred wash, tapered at the corners.
        const float2 edgeP = clamp(warpedPixel, rectMin, rectMax);
        // How far this frame pixel sits outside the glass on each axis (the
        // clamped/"outside" axis is the one perpendicular to the nearest edge).
        const float ox = max(max(rectMin.x - warpedPixel.x, warpedPixel.x - rectMax.x), 0.0f);
        const float oy = max(max(rectMin.y - warpedPixel.y, warpedPixel.y - rectMax.y), 0.0f);
        const float2 rTube = (edgeP - rectMin) / tubeSize;
        // Sample from the OUTER ~10% band of the game — inset ONLY the
        // perpendicular (outside) axis toward the content; keep the tangential
        // axis at the true position. Insetting BOTH pulled sideways content (a
        // menu-panel edge) onto frame sitting over black, which glowed wrongly.
        const float2 outMask = float2(ox > 0.0f ? 1.0f : 0.0f, oy > 0.0f ? 1.0f : 0.0f);
        const float2 rTubeInset = lerp(rTube, 0.06f + 0.88f * rTube, outMask);
        const float2 rUV = (fullMin + rTubeInset * fullSize) / gConstants.texSize;
        // Reflection colour from the 1/32 box-downsampled frame, 3x3-smoothed —
        // a near-uniform soft glow per side, no content structure/"mirror".
        const float3 edgeCol = SampleRefl(rUV);
        // Gate from the SAME smooth source so it has no per-pixel notches: a
        // dark/near-black region (pillarbox bar, curved-out edge) casts no glow.
        const float lit = smoothstep(0.14f, 0.40f, dot(edgeCol, float3(0.299f, 0.587f, 0.114f)));
        const float distPx = length(warpedPixel - edgeP);
        const float bandPx = max(min(INSET_X * fullSize.x, INSET_Y * fullSize.y), 1.0f);
        // reflFall uses the EUCLIDEAN distance to the glass rect, so it is
        // rounded at the corners. NO min(ox,oy) corner suppression — that had
        // SQUARE contours and drew a visible dark square ("Viereck") in every
        // corner. The black-luma gate below already kills reflection where the
        // screen is dark/curved-out, so no extra corner handling is needed.
        const float reflFall = 1.0f - smoothstep(0.0f, bandPx * 0.6f, distPx);
        // Purely ADDITIVE glow (no darkening of the plastic) scaled by how lit
        // the nearest screen band is — black screen => frame untouched.
        const float3 plastic = b.rgb + edgeCol * (reflFall * lit * 0.5f);

        composited = lerp(glass, plastic, b.a);
    }

    // Outside the presented rect (black bars / raw frame): pass the original
    // pixels through untouched.
    const bool insideRect = (pixel.x >= fullMin.x && pixel.x < fullMax.x &&
                             pixel.y >= fullMin.y && pixel.y < fullMax.y);
    const float3 result = insideRect ? composited : gInput.SampleLevel(gSampler, uv, 0).rgb;
    return float4(result, 1.0f);
}
