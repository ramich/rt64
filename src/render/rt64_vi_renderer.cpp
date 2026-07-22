//
// RT64
//

#include "rt64_vi_renderer.h"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <string>

#include "shared/rt64_hlsl.h"
#include "shared/rt64_video_interface.h"

// WR64 per-scene presentation: when enabled, the final blit to the swap chain
// is scissored to a centered 4:3 region, pillarboxing whatever the game
// rendered outside it. This lets a port keep AspectRatio::Expand permanently
// (stable render-target sizes — switching UserConfiguration at runtime either
// crashes on the framebuffer discard or ghosts stale wide targets) while
// menus still present as 4:3. The swap chain is cleared before the VI blit,
// so the cropped margins are true black.
static std::atomic<bool> wr64PresentCrop43{false};

extern "C" void rt64_wr64_set_present_crop43(int enabled) {
    wr64PresentCrop43.store(enabled != 0, std::memory_order_relaxed);
}

extern "C" int rt64_wr64_get_present_crop43() {
    return wr64PresentCrop43.load(std::memory_order_relaxed) ? 1 : 0;
}

// Crop43 sub-mode, following the port's Border Area setting for 4:3 GAMEPLAY:
//   0 = menu (default): full 4:3 box with the 1.08 overscan zoom (menus).
//   1 = borders: inset the scissor's top/bottom to the game's content rect
//       (rows 20..218 of 240) so the uncleared framebuffer strip above the
//       content and the bottom overscan margin present as the black swapchain
//       clear — i.e. the game's original black borders.
//   2 = content zoom: remap the viewport so the content rect (8,20)-(310,218)
//       fills the whole 4:3 box — overscan crop with scale-up, 4:3 flavor.
static std::atomic<int> wr64Crop43Mode{0};

extern "C" void rt64_wr64_set_crop43_mode(int mode) {
    wr64Crop43Mode.store(mode, std::memory_order_relaxed);
}

// WR64 motion blur (prototype): exponential accumulation at present time. The
// strength is the alpha the PREVIOUS presented frame is drawn with over the
// current one (0 = off, ~0.5 = moderate trail, capped below 1 so it decays).
// Consumed by the present queue after the VI blit, before the UI draw hook
// (so menus/overlays stay sharp).
static std::atomic<float> wr64MotionBlur{0.0f};

extern "C" void rt64_wr64_set_motion_blur(float strength) {
    if (!(strength >= 0.0f)) strength = 0.0f;   // also catches NaN
    if (strength > 0.95f) strength = 0.95f;
    wr64MotionBlur.store(strength, std::memory_order_relaxed);
}

extern "C" float rt64_wr64_get_motion_blur() {
    return wr64MotionBlur.load(std::memory_order_relaxed);
}

// WR64 sharpen (contrast-adaptive): strength of the present-time unsharp
// pass, 0 = off (pass skipped entirely).
static std::atomic<float> wr64Sharpen{0.0f};

extern "C" void rt64_wr64_set_sharpen(float strength) {
    if (!(strength >= 0.0f)) strength = 0.0f;   // also catches NaN
    if (strength > 1.0f) strength = 1.0f;
    wr64Sharpen.store(strength, std::memory_order_relaxed);
}

extern "C" float rt64_wr64_get_sharpen() {
    return wr64Sharpen.load(std::memory_order_relaxed);
}

// WR64 CRT filter (Trinitron-style): intensity of the present-time CRT pass
// (aperture grille + scanlines + slight curvature + rounded corners +
// vignette), 0 = off (pass skipped entirely). Applied AFTER sharpen/motion
// blur and before the UI draw hook — the mask sits "on the glass", the
// launcher/overlays stay clean.
static std::atomic<float> wr64Crt{0.0f};

extern "C" void rt64_wr64_set_crt(float strength) {
    if (!(strength >= 0.0f)) strength = 0.0f;   // also catches NaN
    if (strength > 1.0f) strength = 1.0f;
    wr64Crt.store(strength, std::memory_order_relaxed);
}

extern "C" float rt64_wr64_get_crt() {
    return wr64Crt.load(std::memory_order_relaxed);
}

// Whether the CRT filter contributes a subtle phosphor-persistence trail
// (via the motion-blur accumulation pass). Default on; a diagnostic/opt-out
// (WR64_CRT_PERSIST=0) since the accumulation buffer can ghost across
// fullscreen<->window layout changes.
static std::atomic<int> wr64CrtPersist{1};

extern "C" void rt64_wr64_set_crt_persist(int enabled) {
    wr64CrtPersist.store(enabled, std::memory_order_relaxed);
}

extern "C" int rt64_wr64_get_crt_persist() {
    return wr64CrtPersist.load(std::memory_order_relaxed);
}

// WR64 game content rectangle ("tube") in swapchain pixel space, published by
// VIRenderer::render() every present for the CRT pass: the final VI scissor
// after crop43 pillarbox / frame-side bars / band blackout — i.e. exactly the
// region the game image occupies. The CRT curvature/corners map onto this
// rect, not the whole window (black bars stay flat black). Also publishes the
// number of visible SOURCE rows so the scanline pitch tracks the game's real
// scanlines rather than output pixels.
static std::atomic<int32_t> wr64ContentX0{0};
static std::atomic<int32_t> wr64ContentY0{0};
static std::atomic<int32_t> wr64ContentX1{0};
static std::atomic<int32_t> wr64ContentY1{0};
static std::atomic<float>   wr64ContentSrcRows{240.0f};

extern "C" void rt64_wr64_get_content_rect(int32_t* x0, int32_t* y0, int32_t* x1, int32_t* y1) {
    *x0 = wr64ContentX0.load(std::memory_order_relaxed);
    *y0 = wr64ContentY0.load(std::memory_order_relaxed);
    *x1 = wr64ContentX1.load(std::memory_order_relaxed);
    *y1 = wr64ContentY1.load(std::memory_order_relaxed);
}

extern "C" float rt64_wr64_get_content_src_rows() {
    return wr64ContentSrcRows.load(std::memory_order_relaxed);
}

// WR64 screenshot request: the port hands a destination path; the present
// queue consumes it on the next present (copies the final swap-chain image,
// UI included, into a readback buffer and writes a PNG off-thread).
static std::mutex wr64ShotMutex;
static std::string wr64ShotPath;

extern "C" void rt64_wr64_request_screenshot(const char* path) {
    std::lock_guard<std::mutex> lk(wr64ShotMutex);
    wr64ShotPath = (path != nullptr && path[0] != '\0') ? path : "screenshot.png";
}

bool rt64_wr64_take_screenshot_path(std::string& out) {
    std::lock_guard<std::mutex> lk(wr64ShotMutex);
    if (wr64ShotPath.empty()) {
        return false;
    }
    out.swap(wr64ShotPath);
    wr64ShotPath.clear();
    return true;
}

// WR64 2P split-screen band blackout: the two half-viewports are inset by the
// game's ~5% top/bottom border AND separated by a small mid gutter, and shared
// full-frame passes (start gate, countdown, water) spill into all three border
// bands (above the top half, between the halves, below the bottom half). The
// present blits ONLY the two play bands [a0,a1] and [b0,b1] (fractions of the
// scissor height), leaving the top/mid/bottom bands as the pre-cleared black
// swap chain. Persistent state (like the crop flag) so EVERY present --
// including interpolated frames above the 20 Hz game rate -- applies it, with
// no flicker; the port sets it each game frame and clears it otherwise.
// Default (a0=0,a1=1,b1<=b0) = a single full-height band = ordinary present.
static std::atomic<float> wr64SplitA0{0.0f};
static std::atomic<float> wr64SplitA1{1.0f};
static std::atomic<float> wr64SplitB0{0.0f};
static std::atomic<float> wr64SplitB1{0.0f};

// WR64 vertex-position interpolation for CPU-animated meshes; consumed by
// rt64_workload.cpp when building the default transform group and by
// rt64_game_frame.cpp when computing per-vertex velocities. The value is a
// MAX-VERTEX-COUNT threshold: 0 = off; N > 0 = interpolate only transforms
// with <= N vertices. Small drifting quads (WR64 clouds/sprites, 4-14 verts)
// interpolate correctly; the camera-anchored wave-mesh chunks (350-870 verts)
// must not — their vertices are not persistent world points, so interpolating
// them warps the wave animation (user-verified regression).
static std::atomic<int> wr64VertexInterp{0};

extern "C" void rt64_wr64_set_vertex_interp(int maxVerts) {
    wr64VertexInterp.store(maxVerts, std::memory_order_relaxed);
}

extern "C" int rt64_wr64_get_vertex_interp() {
    return wr64VertexInterp.load(std::memory_order_relaxed);
}

extern "C" int rt64_wr64_vertex_interp_limit_hit(uint32_t vertexCount) {
    const int limit = wr64VertexInterp.load(std::memory_order_relaxed);
    return (limit > 0) && (vertexCount > uint32_t(limit));
}

// WR64 overscan crop (GLideN64-style, per-edge, final present stage): map the
// content sub-rect of the VI image onto the full previous display area — crop
// WITH scale-up, so the TV-overscan junk rows (WR64 content rect is
// (8,20)-(310,218) of 320x240) disappear and the game fills the window
// instead of leaving letterbox bands. Fractions of the VI image per edge;
// persistent (set by the port per game frame, interpolation-safe). The
// vertical scale-up stretches the image, so the port pairs this with a
// vertical-FOV compensation in the projection processor (see
// rt64_wr64_get_overscan_yscale) to keep the 3D world's proportions and
// coverage identical; 2D/HUD gets the authentic TV framing.
static std::atomic<float> wr64OverscanL{0.0f};
static std::atomic<float> wr64OverscanR{0.0f};
static std::atomic<float> wr64OverscanT{0.0f};
static std::atomic<float> wr64OverscanB{0.0f};

extern "C" void rt64_wr64_set_overscan(float l, float r, float t, float b) {
    wr64OverscanL.store(l, std::memory_order_relaxed);
    wr64OverscanR.store(r, std::memory_order_relaxed);
    wr64OverscanT.store(t, std::memory_order_relaxed);
    wr64OverscanB.store(b, std::memory_order_relaxed);
}

// Render compensations for the overscan scale-up: (1 - insets), or 1 when
// overscan is inactive. The projection processor multiplies the projection's
// Y (and X) column by these so the world renders pre-compressed by exactly
// the display's stretch in each axis.
extern "C" float rt64_wr64_get_overscan_yscale() {
    const float t = wr64OverscanT.load(std::memory_order_relaxed);
    const float b = wr64OverscanB.load(std::memory_order_relaxed);
    const float remain = 1.0f - t - b;
    return (remain > 0.0f && remain < 1.0f) ? remain : 1.0f;
}

extern "C" float rt64_wr64_get_overscan_xscale() {
    const float l = wr64OverscanL.load(std::memory_order_relaxed);
    const float r = wr64OverscanR.load(std::memory_order_relaxed);
    const float remain = 1.0f - l - r;
    return (remain > 0.0f && remain < 1.0f) ? remain : 1.0f;
}

// EXPERIMENTAL: rigid-translation interpolation for large meshes (the wave
// grid). Off by default — the wave motion still doesn't read right in game
// (user-tested); exposed as a launcher option while it bakes.
static std::atomic<int> wr64VertexInterpRigid{0};

extern "C" void rt64_wr64_set_vertex_interp_rigid(int enabled) {
    wr64VertexInterpRigid.store(enabled, std::memory_order_relaxed);
}

extern "C" int rt64_wr64_get_vertex_interp_rigid() {
    return wr64VertexInterpRigid.load(std::memory_order_relaxed);
}

// Thin divider between the two remapped halves, as a fraction of the window
// height (split across the middle).
static constexpr float wr64SplitDivider = 0.008f;

// Whether the split halves are REMAPPED to fill the window (overscan-style) or
// presented with plain scissor bands (classic gutters). Follows the port's
// Overscan Crop setting.
static std::atomic<int> wr64SplitRemap{0};

extern "C" void rt64_wr64_set_split_remap(int enabled) {
    wr64SplitRemap.store(enabled, std::memory_order_relaxed);
}

// WR64 Border Area = Original in Expand (widescreen): black side bars over
// the outer edges of the widened world. The extreme margins of the world
// expansion show artifacts (blurry seams, stale bands), so the bars are
// deliberately WIDER than the original border columns — the port passes the
// fraction of the presented width to black out per side (0 = off).
static std::atomic<float> wr64FrameSideFrac{0.0f};

extern "C" void rt64_wr64_set_frame_sides(float frac) {
    wr64FrameSideFrac.store(frac, std::memory_order_relaxed);
}

// Presented-frame counter (one per VI present, i.e. including interpolated
// frames) for the port's FPS readout — the game-frame counter only ticks at
// the game's native ~20 Hz.
static std::atomic<uint32_t> wr64PresentCount{0};

extern "C" uint32_t rt64_wr64_consume_present_count() {
    return wr64PresentCount.exchange(0, std::memory_order_relaxed);
}

extern "C" void rt64_wr64_set_split_bands(float a0, float a1, float b0, float b1) {
    wr64SplitA0.store(a0, std::memory_order_relaxed);
    wr64SplitA1.store(a1, std::memory_order_relaxed);
    wr64SplitB0.store(b0, std::memory_order_relaxed);
    wr64SplitB1.store(b1, std::memory_order_relaxed);
}

// Per-half vertical render compensation for the split remap: source band
// height / destination band height (the display stretches each half by the
// inverse). 1.0 when the split remap is inactive.
extern "C" float rt64_wr64_split_yscale(int isTopHalf) {
    if (wr64SplitRemap.load(std::memory_order_relaxed) == 0) {
        return 1.0f;
    }
    const float b0 = wr64SplitB0.load(std::memory_order_relaxed);
    const float b1 = wr64SplitB1.load(std::memory_order_relaxed);
    if (b1 <= b0) {
        return 1.0f;
    }
    const float a0 = wr64SplitA0.load(std::memory_order_relaxed);
    const float a1 = wr64SplitA1.load(std::memory_order_relaxed);
    const float srcH = isTopHalf ? (a1 - a0) : (b1 - b0);
    const float dstH = 0.5f - wr64SplitDivider * 0.5f;
    return (srcH > 0.0f && dstH > 0.0f) ? (srcH / dstH) : 1.0f;
}

// WR64 wide-world mode: while enabled (gameplay frames with border removal
// active), perspective projections whose scissor covers the framebuffer-pair
// scissor take the wide-viewport path even when the game's camera-bob
// viewport translation would fail the usual intersection test (see
// rt64_framebuffer_renderer.cpp).
static std::atomic<bool> wr64WideWorld{false};

extern "C" void rt64_wr64_set_wide_world(int enabled) {
    wr64WideWorld.store(enabled != 0, std::memory_order_relaxed);
}

extern "C" int rt64_wr64_get_wide_world() {
    return wr64WideWorld.load(std::memory_order_relaxed) ? 1 : 0;
}

namespace RT64 {
    // VIRenderer

    VIRenderer::VIRenderer() { }

    VIRenderer::~VIRenderer() { }

    inline hlslpp::float2 computeHDSize(hlslpp::float2 sdSize, hlslpp::float2 resolutionScale, uint32_t downsamplingScale) {
        return (sdSize * resolutionScale) / float(downsamplingScale);
    }

    inline hlslpp::float2 fromSDtoHD(hlslpp::float2 coordinate, hlslpp::float2 sdSize, hlslpp::float2 hdSize) {
        const hlslpp::float2 relativeScale = hdSize / sdSize;
        return coordinate * relativeScale;
    }

    inline hlslpp::float2 fromHDtoWindow(hlslpp::float2 coordinate, hlslpp::float2 hdSize, hlslpp::float2 windowSize) {
        const hlslpp::float2 hdCenter = hdSize / 2;
        const hlslpp::float2 windowCenter = windowSize / 2;
        const hlslpp::float2 relativeCoordinate = { coordinate.x - hdCenter.x, coordinate.y - hdCenter.y };

        // Window is wider than virtual HD TV, we do pillarboxing.
        float relativeScale;
        if ((windowSize.x / windowSize.y) > (hdSize.x / hdSize.y)) {
            relativeScale = windowSize.y / hdSize.y;
        }
        // Window is taller than virtual HD TV, we do letterboxing.
        else {
            relativeScale = windowSize.x / hdSize.x;
        }

        return windowCenter + relativeCoordinate * relativeScale;
    }

    void VIRenderer::render(const RenderParams &p) {
        const ShaderRecord *shader = nullptr;
        const RenderSampler *sampler = nullptr;
        switch (p.filtering) {
        case UserConfiguration::Filtering::Nearest:
            shader = &p.shaderLibrary->videoInterfaceNearest;
            sampler = p.shaderLibrary->samplerLibrary.nearest.borderBorder.get();
            break;
        case UserConfiguration::Filtering::AntiAliasedPixelScaling:
            shader = &p.shaderLibrary->videoInterfacePixel;
            sampler = p.shaderLibrary->samplerLibrary.linear.borderBorder.get();
            break;
        case UserConfiguration::Filtering::Linear:
        default:
            shader = &p.shaderLibrary->videoInterfaceLinear;
            sampler = p.shaderLibrary->samplerLibrary.linear.borderBorder.get();
            break;
        }

        if ((descriptorSet == nullptr) || (descriptorSetSampler != sampler)) {
            descriptorSet = std::make_unique<VideoInterfaceDescriptorSet>(sampler, p.device);
            descriptorSetSampler = sampler;
        }

        descriptorSet->setTexture(descriptorSet->gInput, p.texture, RenderTextureLayout::SHADER_READ);

        wr64PresentCount.fetch_add(1, std::memory_order_relaxed);

        RenderViewport viewport;
        RenderRect scissor;
        getViewportAndScissor(p.swapChain, *p.vi, p.resolutionScale, p.downsamplingScale, p.removeBlackBorders, viewport, scissor);
        p.commandList->setViewports(viewport);

        interop::VideoInterfaceCB pushConstants;
        pushConstants.videoResolution = computeHDSize(hlslpp::float2(p.vi->fbSize()), p.resolutionScale, p.downsamplingScale);
        pushConstants.textureResolution = { float(p.textureWidth), float(p.textureHeight) };
        pushConstants.gamma = p.vi->gamma();

        p.commandList->setPipeline(shader->pipeline.get());
        p.commandList->setGraphicsPipelineLayout(shader->pipelineLayout.get());
        p.commandList->setGraphicsDescriptorSet(descriptorSet->get(), 0);
        p.commandList->setGraphicsPushConstants(0, &pushConstants);
        p.commandList->setVertexBuffers(0, nullptr, 0, nullptr);

        // WR64 2P split-screen: with two bands set, each half's content band is
        // REMAPPED to fill its half of the window (overscan-style crop with
        // scale-up: no top/bottom border gutters, just a thin divider). The
        // projection processor compensates the per-half vertical stretch (see
        // rt64_wr64_split_yscale) so the world keeps its proportions. With a
        // single band, plain scissor cropping as before. Default single [0,1]
        // band = ordinary present.
        const float a0 = wr64SplitA0.load(std::memory_order_relaxed);
        const float a1 = wr64SplitA1.load(std::memory_order_relaxed);
        const float b0 = wr64SplitB0.load(std::memory_order_relaxed);
        const float b1 = wr64SplitB1.load(std::memory_order_relaxed);
        const int32_t sTop = scissor.top;
        const int32_t sH = scissor.bottom - scissor.top;
        {
            // Original-borders side bars in Expand (see the setter above):
            // inset the present scissor left/right by the configured fraction
            // of the visible width; the excluded margins present as the black
            // swapchain clear. The top/bottom border bands arrive separately
            // via the split bands below.
            const float sideFrac = wr64FrameSideFrac.load(std::memory_order_relaxed);
            if (sideFrac > 0.0f) {
                const int32_t inset = int32_t(lround(float(scissor.right - scissor.left) * sideFrac));
                scissor.left += inset;
                scissor.right -= inset;
            }
        }
        {
            // Publish the content rectangle + visible source rows for the CRT
            // pass (see the getters above). Base = the final scissor (crop43 /
            // side bars already applied); a partial single band (Border Area =
            // Original bars in Expand) narrows it vertically. 2P split keeps
            // the full scissor — both halves form ONE tube, like a real CRT.
            RenderRect content = scissor;
            float srcRows = 240.0f;
            if (b1 > b0) {
                srcRows = 240.0f;
            } else if (a1 - a0 < 0.999f) {
                content.top = std::max(scissor.top, sTop + int32_t(lround(a0 * sH)));
                content.bottom = std::min(scissor.bottom, sTop + int32_t(lround(a1 * sH)));
                srcRows = 240.0f * (a1 - a0);
            } else if (wr64PresentCrop43.load(std::memory_order_relaxed) &&
                       wr64Crop43Mode.load(std::memory_order_relaxed) != 0) {
                // 4:3 borders-inset / content-zoom show the content rows only.
                srcRows = 198.0f;
            } else {
                const float ovT = wr64OverscanT.load(std::memory_order_relaxed);
                const float ovB = wr64OverscanB.load(std::memory_order_relaxed);
                const float remain = 1.0f - ovT - ovB;
                if (remain > 0.0f && remain < 1.0f) {
                    srcRows = 240.0f * remain;
                }
            }
            wr64ContentX0.store(content.left, std::memory_order_relaxed);
            wr64ContentY0.store(content.top, std::memory_order_relaxed);
            wr64ContentX1.store(content.right, std::memory_order_relaxed);
            wr64ContentY1.store(content.bottom, std::memory_order_relaxed);
            wr64ContentSrcRows.store(srcRows, std::memory_order_relaxed);
        }
        if (b1 > b0 && wr64SplitRemap.load(std::memory_order_relaxed) != 0) {
            // Two halves: source band [srcF0..srcF1] of the image maps onto
            // destination band [dstF0..dstF1] of the scissor area.
            auto drawHalf = [&](float srcF0, float srcF1, float dstF0, float dstF1) {
                const float srcH = srcF1 - srcF0;
                if (srcH <= 0.0f) return;
                const float dstY0 = float(sTop) + dstF0 * float(sH);
                const float dstY1 = float(sTop) + dstF1 * float(sH);
                RenderViewport vp = viewport;
                vp.height = (dstY1 - dstY0) / srcH;
                vp.y = dstY0 - srcF0 * vp.height;
                RenderRect band = scissor;
                band.top = std::max(scissor.top, int32_t(lround(dstY0)));
                band.bottom = std::min(scissor.bottom, int32_t(lround(dstY1)));
                if (band.bottom <= band.top) return;
                p.commandList->setViewports(vp);
                p.commandList->setScissors(band);
                p.commandList->drawInstanced(3, 1, 0, 0);
            };
            drawHalf(a0, a1, 0.0f, 0.5f - wr64SplitDivider * 0.5f);
            drawHalf(b0, b1, 0.5f + wr64SplitDivider * 0.5f, 1.0f);
        }
        else {
            auto blitBand = [&](float f0, float f1) {
                RenderRect band = scissor;
                band.top = std::max(scissor.top, sTop + int32_t(lround(f0 * sH)));
                band.bottom = std::min(scissor.bottom, sTop + int32_t(lround(f1 * sH)));
                if (band.bottom <= band.top) return;
                p.commandList->setScissors(band);
                p.commandList->drawInstanced(3, 1, 0, 0);
            };
            blitBand(a0, a1);
            if (b1 > b0) {
                blitBand(b0, b1);
            }
        }
    }

    void VIRenderer::getViewportAndScissor(const RenderSwapChain *swapChain, const VI &vi, hlslpp::float2 resolutionScale, uint32_t downsamplingScale, bool removeBlackBorders, RenderViewport &viewport, RenderRect &scissor) {
        // We define three different coordinate spaces to work with to translate the VI parameters into the Window.
        //
        // VideoSD: This corresponds to the SD TV Scanline space, which is what the VI natively works on.
        // 
        // VideoHD: This corresponds to what the imaginary "HD" TV would be if it supported the amount of scanlines
        // desired by the resolution scale (divided by the downsampling scale) and a wider aspect ratio.
        // 
        // Window: The native coordinate space of the device's render target.
        //
        // The provided buffer doesn't necessarily have the same dimensions that the VI will sample to display it 
        // on the screen. To work around that, the viewport the buffer will be drawn in will be expanded so only
        // the region of interest is rendered. A scissor will cut it off correctly according to the coordinates
        // specified by the VI.
        const hlslpp::float2 sdSize = removeBlackBorders ? hlslpp::float2(vi.fbSize()) : hlslpp::float2(320.0f, 240.0f);
        const hlslpp::float2 hdSize = computeHDSize(sdSize, resolutionScale, downsamplingScale);
        const hlslpp::float2 windowSize = { float(swapChain->getWidth()), float(swapChain->getHeight()) };

        // Query the VI for the current rendering area.
        hlslpp::float4 viViewRect = vi.viewRectangle() * sdSize.xyxy;
        hlslpp::float4 viCropRect = vi.cropRectangle() * sdSize.xyxy;

        // Scale all the rectangles to the space of the Window.
        hlslpp::float2 topLeftViewport = fromSDtoHD({ float(viViewRect.x), float(viViewRect.y) }, sdSize, hdSize);
        hlslpp::float2 bottomRightViewport = fromSDtoHD({ float(viViewRect.x + viViewRect.z), float(viViewRect.y + viViewRect.w) }, sdSize, hdSize);
        topLeftViewport = fromHDtoWindow(topLeftViewport, hdSize, windowSize);
        bottomRightViewport = fromHDtoWindow(bottomRightViewport, hdSize, windowSize);

        hlslpp::float2 topLeftScissor = fromSDtoHD({ float(viCropRect.x), float(viCropRect.y) }, sdSize, hdSize);
        hlslpp::float2 bottomRightScissor = fromSDtoHD({ float(viCropRect.x + viCropRect.z), float(viCropRect.y + viCropRect.w) }, sdSize, hdSize);
        topLeftScissor = fromHDtoWindow(topLeftScissor, hdSize, windowSize);
        bottomRightScissor = fromHDtoWindow(bottomRightScissor, hdSize, windowSize);

        viewport = RenderViewport(topLeftViewport.x, topLeftViewport.y, bottomRightViewport.x - topLeftViewport.x, bottomRightViewport.y - topLeftViewport.y);
        scissor = RenderRect(lround(topLeftScissor.x), lround(topLeftScissor.y), lround(bottomRightScissor.x), lround(bottomRightScissor.y));

        if (wr64PresentCrop43.load(std::memory_order_relaxed)) {
            const float scissorHeight = float(scissor.bottom - scissor.top);
            const float centerX = (float(scissor.left) + float(scissor.right)) * 0.5f;
            const float halfWidth = scissorHeight * (4.0f / 3.0f) * 0.5f;
            scissor.left = std::max(scissor.left, int32_t(lround(centerX - halfWidth)));
            scissor.right = std::min(scissor.right, int32_t(lround(centerX + halfWidth)));

            const int crop43Mode = wr64Crop43Mode.load(std::memory_order_relaxed);
            if (crop43Mode == 1) {
                // 4:3 GAMEPLAY, borders mode: inset the scissor to the game's
                // content rect (8,20)-(310,218) on ALL four sides so the
                // border band presents as the black swapchain clear — the
                // game's original black borders. The band is NOT black in the
                // framebuffer: it is uncleared memory a real TV's overscan
                // hid (visibly flickering in-race if displayed), so the whole
                // band must be scissored out, sides included. No overscan
                // zoom here (that is menu-only).
                const int32_t top0 = scissor.top;
                const int32_t left0 = scissor.left;
                const float boxW = float(scissor.right - scissor.left);
                scissor.top = top0 + int32_t(lround(scissorHeight * (20.0f / 240.0f)));
                scissor.bottom = top0 + int32_t(lround(scissorHeight * (218.0f / 240.0f)));
                scissor.left = left0 + int32_t(lround(boxW * (8.0f / 320.0f)));
                scissor.right = left0 + int32_t(lround(boxW * (310.0f / 320.0f)));
            } else if (crop43Mode == 2) {
                // 4:3 GAMEPLAY, overscan-crop mode: remap the viewport so the
                // SD content rect (8,20)-(310,218) fills the whole 4:3 box.
                // Within the box, SD x spans 0..320 and y spans 0..240; the
                // affine remap p -> boxStart + (p - inset0)/contentSize * boxSize
                // applied to the viewport crops the border margins with
                // scale-up (no black bars, no junk strip), like the widescreen
                // overscan crop but confined to the pillarboxed 4:3 area.
                const float boxL = float(scissor.left);
                const float boxT = float(scissor.top);
                const float boxW = float(scissor.right - scissor.left);
                const float boxH = float(scissor.bottom - scissor.top);
                const float sx = 320.0f / 302.0f;  // content cols 8..310
                const float sy = 240.0f / 198.0f;  // content rows 20..218
                const float vx = boxL + (viewport.x - boxL) * sx - (8.0f / 302.0f) * boxW;
                const float vy = boxT + (viewport.y - boxT) * sy - (20.0f / 198.0f) * boxH;
                viewport = RenderViewport(vx, vy, viewport.width * sx, viewport.height * sy);
            } else {
                // Overscan zoom: WR64's menus assume a TV's overscan hides the
                // frame edges, but our exact 4:3 crop exposes parked-world
                // content sitting in that overscan margin (e.g. the stray craft
                // at the right edge of the 2P watercraft-select). Enlarge the
                // blit ~8% about its center so the overscan band falls outside
                // the scissor. Menu panels live well inside the overscan-safe
                // area, so this doesn't clip real content.
                const float overscan = 1.08f;
                const float vcx = viewport.x + viewport.width * 0.5f;
                const float vcy = viewport.y + viewport.height * 0.5f;
                const float nw = viewport.width * overscan;
                const float nh = viewport.height * overscan;
                viewport = RenderViewport(vcx - nw * 0.5f, vcy - nh * 0.5f, nw, nh);

                // Vertical clamp: the zoom fully pushes the SIDE border band
                // out of the box (cols 8/310 land outside at 1.08x), but only
                // ~4% of the ~8-9% TALL border band — the rest of the
                // uncleared rows (garbage that flickers in some menus, e.g.
                // player select) would still show. Scissor to the zoomed
                // positions of the content rows 20..218 so the remainder
                // presents as black without cropping any real menu content.
                auto zoomedRow = [&](float row) {
                    return 0.5f + (row / 240.0f - 0.5f) * overscan;
                };
                const int32_t boxTop = scissor.top;
                scissor.top = boxTop + int32_t(lround(scissorHeight * zoomedRow(20.0f)));
                scissor.bottom = boxTop + int32_t(lround(scissorHeight * zoomedRow(218.0f)));
            }
        }

        // WR64 gameplay overscan crop (see the setters above): remap the
        // viewport per axis so the inset content sub-rect of the VI image maps
        // onto the area the full image used to occupy — crop with scale-up.
        // The port only sets these during (non-menu, non-split) gameplay.
        {
            const float fl = wr64OverscanL.load(std::memory_order_relaxed);
            const float fr = wr64OverscanR.load(std::memory_order_relaxed);
            const float ft = wr64OverscanT.load(std::memory_order_relaxed);
            const float fb = wr64OverscanB.load(std::memory_order_relaxed);
            const float remainX = 1.0f - fl - fr;
            const float remainY = 1.0f - ft - fb;
            if ((remainX > 0.0f && remainX < 1.0f) || (remainY > 0.0f && remainY < 1.0f)) {
                float vx = viewport.x, vy = viewport.y;
                float vw = viewport.width, vh = viewport.height;
                if (remainX > 0.0f && remainX < 1.0f) {
                    const float nw2 = vw / remainX;
                    vx = vx - fl * nw2;
                    vw = nw2;
                }
                if (remainY > 0.0f && remainY < 1.0f) {
                    const float nh2 = vh / remainY;
                    vy = vy - ft * nh2;
                    vh = nh2;
                }
                viewport = RenderViewport(vx, vy, vw, vh);
            }
        }
        // NOTE: the 2P split-screen band blackout is applied in render() (it
        // needs two separate blits with a black mid gap), not here.
    }
};