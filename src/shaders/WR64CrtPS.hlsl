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

float4 PSMain(in float4 pos : SV_Position, in float2 uv : TEXCOORD0) : SV_TARGET {
    const float2 pixel = uv * gConstants.texSize;
    const float2 rectMin = gConstants.rectMin;
    const float2 rectMax = gConstants.rectMax;
    const float2 tubeSize = rectMax - rectMin;
    const float k = gConstants.intensity;

    // Degenerate rect: pass through.
    if (tubeSize.x < 8.0f || tubeSize.y < 8.0f) {
        return float4(gInput.SampleLevel(gSampler, uv, 0).rgb, 1.0f);
    }

    // Tube coordinates, centered: c in [-1,1] across the content rect.
    // Pixels outside the rect keep computing in this space (|c| > 1) — they
    // fall into the bezel band or the untouched surroundings below.
    const float2 tubeUV = (pixel - rectMin) / tubeSize;
    float2 c = tubeUV * 2.0f - 1.0f;

    // Barrel distortion: map the (flat) output pixel to the (bulged) source
    // position. Edges/corners sample outside the image and go black, giving
    // the curved-glass silhouette.
    const float curve = 0.045f * k;
    const float r2 = dot(c, c);
    float2 d = c * (1.0f + curve * r2);

    // Frame opening: a rounded box in the DISTORTED image space, scaled to
    // pixels. Its straight edges lie exactly on the image boundary (|d| = 1)
    // and the corner arcs cut INTO the image — the picture sits UNDER the
    // bezel like a real tube under a real frame, so no black moat can open
    // between the (barrel-shrunk) image and the frame. Pixel scaling keeps
    // the arcs circular and the frame width uniform on every side (the
    // normalized space is anisotropic in wide windows — computing there made
    // corners look like flat 45-degree chamfers). Radius is FIXED (not
    // scaled by the CRT intensity).
    const float2 halfPix = tubeSize * 0.5f;
    const float minHalf = min(halfPix.x, halfPix.y);
    const float cornerR = 0.10f * minHalf;
    const float2 qd = (abs(d) - 1.0f) * halfPix + cornerR;   // pixel dist to the content edge, radius-shifted
    const float cornerDist = length(max(qd, 0.0f)) + min(max(qd.x, qd.y), 0.0f) - cornerR;
    const float tubeMask = 1.0f - smoothstep(-2.0f, 2.0f, cornerDist);
    const float2 q = qd;   // used by the bezel shading below

    // Sample the source at the distorted position (clamped just inside so
    // the border-color sampler never bleeds).
    const float2 srcTube = d * 0.5f + 0.5f;
    const float2 srcPixel = rectMin + saturate(srcTube) * tubeSize;
    const float2 srcUV = srcPixel / gConstants.texSize;
    float3 color = gInput.SampleLevel(gSampler, srcUV, 0).rgb;

    // P22 phosphor color: mild channel crosstalk toward the CRT phosphor
    // primaries plus a touch of CRT gamma — the warm, slightly denser look
    // of the tube. Scaled with intensity like everything else.
    const float3 p22 = float3(
        dot(color, float3(0.955f, 0.040f, 0.005f)),
        dot(color, float3(0.030f, 0.960f, 0.010f)),
        dot(color, float3(0.015f, 0.030f, 0.955f)));
    color = lerp(color, p22, k);
    color = pow(saturate(color), lerp(1.0f, 1.10f, k));

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

    // Mild vignette toward the tube corners, capped so the extreme corners
    // (r2 reaches 2.0 there) don't sink into a dark pit.
    color *= 1.0f - min(0.12f * k * r2 * r2, 0.22f);

    // Phosphor glow / halation: add the inline bright-pass bloom on top of
    // the masked image (emitted light scatters in the glass, so it sits over
    // the grille/scanline pattern and softens it around highlights). Sampled
    // at the distorted position so the glow curves with the image.
    const float3 glow = CrtHalation(srcUV);
    color += glow * (0.45f * k);

    // Brightness compensation: the grille eats grilleStrength on 2 of 3
    // channels, the scanlines eat scanStrength/2 on average.
    const float gain = 1.0f / max((1.0f - grilleStrength * 2.0f / 3.0f) *
                                  (1.0f - scanStrength * 0.5f), 0.55f);
    color = saturate(color * min(gain, 1.35f));

    // Bezel: a plastic frame with visual depth around the glass edge. The
    // screen's (bright-passed, blurred) glow reflects into it, mirrored at
    // the tube edge and falling off outward — like light spilling onto the
    // cabinet of a real TV. Rendered wherever there is room around the tube
    // (pillarbox bars, the rounded-corner voids); farther out the frame
    // fades back to the untouched surroundings.
    // The bezel is an independent toggle with FIXED strength — it does not
    // fade with the CRT intensity slider (gConstants.bezel is 0 or 1).
    // Width in pixels (isotropic): same frame thickness on every side.
    const float bezelW = 0.12f * minHalf;
    float3 bezel = float3(0.0f, 0.0f, 0.0f);
    float bezelMask = 0.0f;
    if (gConstants.bezel > 0.5f && cornerDist > 0.0f && cornerDist < bezelW) {
        // Diffuse reflection: sample the image at the NEAREST point on the
        // tube edge (clamp d into the tube). This is continuous all the way
        // around the rounded corners — a per-axis mirror seams diagonally
        // where an edge meets a corner. Matte plastic spill, not a mirror.
        const float2 reflTube = clamp(d, -1.0f, 1.0f) * 0.5f + 0.5f;
        const float2 reflUV = (rectMin + reflTube * tubeSize) / gConstants.texSize;
        // Tone-compress the reflection (Reinhard): matte plastic can never
        // reflect anywhere near image brightness. Uncompressed, bright white
        // content (sky/surf) lit the whole band up to image level and visually
        // "filled in" the rounded corner, leaving a square white silhouette —
        // while on mid-tone scenes the same strength was barely visible.
        // Compression caps white at ~0.4 and keeps mid tones present.
        const float3 reflRaw = CrtHalation(reflUV) * 0.90f
                             + gInput.SampleLevel(gSampler, reflUV, 0).rgb * 0.10f;
        const float3 refl = reflRaw / (1.0f + 2.5f * reflRaw) * 1.40f;

        // Depth shading of the recess: the bezel walls face the tube center,
        // lit from above — the top wall falls into shadow, the bottom wall
        // catches the room light, the sides sit in between (like a real TV's
        // recessed screen). A dark gap right at the glass separates tube and
        // frame; the slanted face dims toward its outer edge.
        const float t = cornerDist / bezelW;
        const float2 outward = max(q, 0.0f);
        const float2 dirO = (outward.x + outward.y > 1e-4f)
            ? normalize(outward * sign(d)) : float2(0.0f, (d.y >= 0.0f) ? 1.0f : -1.0f);
        const float lambert = dirO.y;   // +1 below the tube (lit), -1 above (shadow)
        float shade = 0.80f + 0.40f * lambert;
        const float glassGap = smoothstep(0.03f, 0.28f, t);  // dark ring at the glass
        shade *= lerp(0.30f, 1.0f, glassGap);
        shade *= 1.08f - 0.38f * smoothstep(0.30f, 1.0f, t); // slant falloff

        // Reflection starts BEHIND the glass gap and peaks mid-band: without
        // the gap, bright image content continued seamlessly across the
        // curved glass edge into the frame and read as a square white ledge
        // sticking past the rounded corner (user-reported).
        const float falloff = glassGap * pow(saturate(1.0f - t), 1.6f);
        const float rim = (1.0f - smoothstep(0.0f, bezelW * 0.18f, cornerDist)) * 0.06f;
        bezel = float3(0.090f, 0.090f, 0.100f) * shade
              + refl * falloff * (0.55f + 0.45f * shade)
              + rim.xxx;
        // Fade the frame out toward its outer edge. Fixed strength — the
        // bezel does not scale with the CRT intensity.
        bezelMask = 1.0f - smoothstep(bezelW * 0.75f, bezelW, cornerDist);
    }

    // Compose the three zones: tube -> bezel -> untouched surroundings.
    // Inside the content rect the non-tube area must stay black (the
    // curvature pulled the image away from it); outside the rect the
    // original pixels (black bars / raw frame) show through.
    const bool insideRect = (pixel.x >= rectMin.x && pixel.x < rectMax.x &&
                             pixel.y >= rectMin.y && pixel.y < rectMax.y);
    const float3 background = insideRect
        ? float3(0.0f, 0.0f, 0.0f)
        : gInput.SampleLevel(gSampler, uv, 0).rgb;
    const float3 outsideTube = lerp(background, saturate(bezel), bezelMask);
    return float4(color * tubeMask + outsideTube * (1.0f - tubeMask), 1.0f);
}
