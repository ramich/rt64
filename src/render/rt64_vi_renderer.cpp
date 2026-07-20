//
// RT64
//

#include "rt64_vi_renderer.h"

#include <algorithm>
#include <atomic>

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

extern "C" void rt64_wr64_set_split_bands(float a0, float a1, float b0, float b1) {
    wr64SplitA0.store(a0, std::memory_order_relaxed);
    wr64SplitA1.store(a1, std::memory_order_relaxed);
    wr64SplitB0.store(b0, std::memory_order_relaxed);
    wr64SplitB1.store(b1, std::memory_order_relaxed);
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

        // WR64 2P split-screen: blit only the two play bands, leaving the
        // top/mid/bottom border gutters black (see the setter above). Default
        // state is a single full band [0,1] = ordinary present.
        const float a0 = wr64SplitA0.load(std::memory_order_relaxed);
        const float a1 = wr64SplitA1.load(std::memory_order_relaxed);
        const float b0 = wr64SplitB0.load(std::memory_order_relaxed);
        const float b1 = wr64SplitB1.load(std::memory_order_relaxed);
        const int32_t sTop = scissor.top;
        const int32_t sH = scissor.bottom - scissor.top;
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
        }
        // NOTE: the 2P split-screen band blackout is applied in render() (it
        // needs two separate blits with a black mid gap), not here.
    }
};