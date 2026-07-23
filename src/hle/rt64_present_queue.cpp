//
// RT64
//

#include "rt64_present_queue.h"

#include <algorithm>
#include <thread>
#include <vector>

#include "common/rt64_thread.h"
#include "rhi/rt64_render_hooks.h"
#include "shared/rt64_wr64_motion_blur.h"
#include "shared/rt64_wr64_sharpen.h"
#include "shared/rt64_wr64_crt.h"

#include "rt64_workload_queue.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

// WR64 fork hooks (rt64_vi_renderer.cpp): motion-blur/sharpen strengths and
// the pending-screenshot path for the present-time passes below.
extern "C" float rt64_wr64_get_motion_blur();
extern "C" float rt64_wr64_get_sharpen();
extern "C" float rt64_wr64_get_crt();
extern "C" int rt64_wr64_get_crt_persist();
extern "C" int rt64_wr64_get_crt_bezel();
extern "C" void rt64_wr64_get_content_rect(int32_t* x0, int32_t* y0, int32_t* x1, int32_t* y1);
extern "C" float rt64_wr64_get_content_src_rows();
bool rt64_wr64_take_screenshot_path(std::string& out);

namespace RT64 {
    // PresentQueue

    PresentQueue::PresentQueue() {
        reset();
    }

    PresentQueue::~PresentQueue() {
        presentThreadRunning = false;
        cursorCondition.notify_all();

        if (presentThread != nullptr) {
            presentThread->join();
            delete presentThread;
        }

        presentIdCondition.notify_all();
    }

    void PresentQueue::reset() {
        threadCursor = 0;
        writeCursor = 0;
        barrierCursor = 0;
        presentId = 0;
    }

    void PresentQueue::advanceToNextPresent() {
        int nextWriteCursor = (writeCursor + 1) % presents.size();

        // Stall the thread until the barrier is lifted if we're trying to write on a present being used by the GPU.
        bool waitForBarrier;
        do {
            const std::scoped_lock lock(cursorMutex);
            waitForBarrier = (nextWriteCursor == barrierCursor);
        } while (waitForBarrier);

        // Modify the cursor and notify anything waiting on the queue.
        {
            const std::scoped_lock lock(cursorMutex);
            writeCursor = nextWriteCursor;
        }

        cursorCondition.notify_all();
    }

    void PresentQueue::repeatLastPresent() {
        {
            const std::scoped_lock lock(cursorMutex);
            threadCursor = previousWriteCursor();
        }

        cursorCondition.notify_all();
    }

    uint32_t PresentQueue::previousWriteCursor() const {
        if (writeCursor > 0) {
            return writeCursor - 1;
        }
        else {
            return uint32_t(presents.size()) - 1;
        }
    }

    void PresentQueue::waitForIdle() {
        std::unique_lock<std::mutex> threadLock(threadMutex);
    }

    void PresentQueue::waitForPresentId(uint64_t waitId) {
        std::unique_lock<std::mutex> presentLock(presentIdMutex);
        presentIdCondition.wait(presentLock, [&]() {
            return (waitId <= presentId) || !presentThreadRunning;
        });
    }

    void PresentQueue::setup(const External &ext) {
        this->ext = ext;

        viRenderer = std::make_unique<VIRenderer>();

        presentThreadRunning = true;
        presentThread = new std::thread(&PresentQueue::threadLoop, this);
    }

    void PresentQueue::threadPresent(const Present &present, bool &swapChainValid) {
        FramebufferManager &fbManager = ext.sharedResources->framebufferManager;
        RenderTargetManager &targetManager = ext.sharedResources->renderTargetManager;
        const bool usingMSAA = (targetManager.multisampling.sampleCount > 1);
        hlslpp::float2 resolutionScale;
        EnhancementConfiguration::Presentation::Mode presentationMode;
        bool removeBlackBorders;
        UserConfiguration::RefreshRate refreshRate;
        UserConfiguration::Filtering filtering;
        uint32_t viOriginalRate;
        uint32_t targetRate;
        {
            std::scoped_lock<std::mutex> configurationLock(ext.sharedResources->configurationMutex);
            resolutionScale = ext.sharedResources->resolutionScale;
            presentationMode = ext.sharedResources->enhancementConfig.presentation.mode;
            removeBlackBorders = ext.sharedResources->enhancementConfig.presentation.removeBlackBorders;
            refreshRate = ext.sharedResources->userConfig.refreshRate;
            filtering = ext.sharedResources->userConfig.filtering;
            viOriginalRate = ext.sharedResources->viOriginalRate;
            targetRate = ext.sharedResources->targetRate;
        }

        RenderTarget *colorTarget = nullptr;
        int32_t framesToPresent = 1;
        bool lockedWorkloadMutex = false;
        InterpolatedFrameCounters &frameCounters = ext.sharedResources->interpolatedFrames[ext.sharedResources->interpolatedFramesIndex];

        // TODO: There's a possible race condition interactions that can happen while the workload
        // queue is rendering extra frames and the present event is processed while it's generating
        // interpolated frames. When the framebuffer manager or the render target manager maps are
        // modified while the present queue is retrieving the framebuffer or the target. These can
        // likely be solved by locking the access to the managers during modification.
        
        // Perform any external write operations indicated by the event.
        if (!present.fbOperations.empty()) {
            const std::scoped_lock lock(screenFbChangePoolMutex);
            {
                RenderWorkerExecution workerExecution(ext.presentGraphicsWorker);
                fbManager.performOperations(ext.presentGraphicsWorker, &screenFbChangePool, nullptr, ext.shaderLibrary, nullptr,
                    present.fbOperations, targetManager, resolutionScale, 0, 0, nullptr);
            }
        }

        // Present the VI specified by the event.
        // Attempt to find the matching framebuffer for the VI based on the origin address.
        // If that fails, we look at the shared storage.
        if (present.screenVI.visible()) {
            Framebuffer *viFb = nullptr;
            if (!viewRDRAM) {
                viFb = fbManager.find(present.screenVI.fbAddress());
            }

            Framebuffer *presentFb = viFb;
            
            // Show the framebuffer the debugger has requested instead.
            if (present.debuggerFramebuffer.view) {
                Framebuffer *candidateFb = fbManager.find(present.debuggerFramebuffer.address);
                if (candidateFb != nullptr) {
                    presentFb = candidateFb;
                }
            }
            
            if ((presentFb != nullptr) && (viFb != nullptr)) {
                for (uint32_t colorAddress : ext.sharedResources->colorImageAddressVector) {
                    Framebuffer *colorFb = fbManager.find(colorAddress);
                    if (colorFb == nullptr) {
                        continue;
                    }

                    // Always default to interpolation being disabled for all modified framebuffers.
                    colorFb->interpolationEnabled = false;
                    
                    // When the skip buffering option is on, we check the video history to find if any of the framebuffers that
                    // were drawn in this frame have been previously used for presentation. This is ignored when the debugger
                    // has forced viewing a particular framebuffer.
                    if (!present.debuggerFramebuffer.view && (presentationMode == EnhancementConfiguration::Presentation::Mode::SkipBuffering)) {
                        for (size_t h = 0; h < viHistory.history.size(); h++) {
                            const VIHistory::Present &entry = viHistory.history[h];
                            if ((colorFb->addressStart == entry.vi.fbAddress()) && (colorFb->width == entry.fbWidth) && (colorFb->siz == entry.vi.fbSiz()) && entry.vi.compatibleWith(present.screenVI)) {
                                presentFb = colorFb;
                                break;
                            }
                        }
                    }

                    // Present early (or games that behave like it) will make it so that the presented image is a color image
                    // that the workload modified. We run a basic check to see if that holds true to indicate it was presented
                    // so interpolation is possible.
                    if (colorFb == presentFb) {
                        presentFb->interpolationEnabled = true;
                        break;
                    }
                }

                if (presentFb->interpolationEnabled) {
                    framesToPresent = frameCounters.count;
                }
                else {
                    lockedWorkloadMutex = true;
                    ext.sharedResources->workloadMutex.lock();
                }

                RenderTargetKey colorTargetKey(presentFb->addressStart, presentFb->width, presentFb->siz, Framebuffer::Type::Color);
                colorTarget = &targetManager.get(colorTargetKey, true);
                if (!colorTarget->isEmpty()) {
                    // If a depth framebuffer is about to be shown, convert it to color.
                    if (presentFb->isLastWriteDifferent(Framebuffer::Type::Color)) {
                        RenderTargetKey otherColorTargetKey(presentFb->addressStart, presentFb->width, presentFb->siz, presentFb->lastWriteType);
                        RenderTarget &otherColorTarget = targetManager.get(otherColorTargetKey, true);
                        if (!otherColorTarget.isEmpty()) {
                            const FixedRect &r = presentFb->lastWriteRect;
                            RenderWorkerExecution workerExecution(ext.presentGraphicsWorker);
                            colorTarget->copyFromTarget(ext.presentGraphicsWorker, &otherColorTarget, r.left(false), r.top(false), r.width(false, true), r.height(false, true), ext.shaderLibrary);
                        }
                    }
                }
                else {
                    colorTarget = nullptr;
                }

                if (!present.paused && (viHistory.top().vi != present.screenVI)) {
                    viHistory.pushVI(present.screenVI, viFb->width);
                }
            }
            else {
                uint32_t fbAddress = present.screenVI.fbAddress();

                // Use a scratch framebuffer to upload the RAM to the render target.
                hlslpp::uint2 fbSize = present.screenVI.fbSize();
                scratchFb.addressStart = fbAddress;
                scratchFb.width = fbSize.x;
                scratchFb.height = fbSize.y;
                scratchFb.siz = present.screenVI.fbSiz();

                lockedWorkloadMutex = true;
                ext.sharedResources->workloadMutex.lock();

                RenderTargetKey colorTargetKey(fbAddress, scratchFb.width, scratchFb.siz, Framebuffer::Type::Color);
                colorTarget = &targetManager.get(colorTargetKey, true);
                colorTarget->resize(ext.presentGraphicsWorker, scratchFb.width, scratchFb.height);
                colorTarget->resolutionScale = { 1.0f, 1.0f };
                colorTarget->downsampleMultiplier = 1;

                scratchFb.nativeTarget.resetBufferHistory();

                {
                    RenderWorkerExecution workerExecution(ext.presentGraphicsWorker);
                    colorTarget->clearColorTarget(ext.presentGraphicsWorker);
                    FramebufferChange *colorFbChange = scratchFb.readChangeFromBytes(ext.presentGraphicsWorker, scratchFbChangePool, Framebuffer::Type::Color,
                        G_IM_FMT_RGBA, present.storage.data(), 0, scratchFb.height, ext.shaderLibrary);

                    if (colorFbChange != nullptr) {
                        colorTarget->copyFromChanges(ext.presentGraphicsWorker, *colorFbChange, scratchFb.width, scratchFb.height, 0, ext.shaderLibrary);
                    }
                }

                scratchFbChangePool.reset();

                if (!present.paused && (viHistory.top().vi != present.screenVI)) {
                    viHistory.pushVI(present.screenVI, fbSize.x);
                }
            }
        }

        // Create the framebuffers if necessary.
        if (swapChainFramebuffers.empty()) {
            uint32_t textureCount = ext.swapChain->getTextureCount();
            swapChainFramebuffers.resize(textureCount);
            for (uint32_t i = 0; i < textureCount; i++) {
                const RenderTexture *swapChainTexture = ext.swapChain->getTexture(i);
                swapChainFramebuffers[i] = ext.device->createFramebuffer(RenderFramebufferDesc(&swapChainTexture, 1));
            }

            // WR64: the swap chain was (re)created — drop ALL present-effect
            // resources and history. The lazy (re)allocation below keys on
            // SIZES only; across a fullscreen<->window transition sizes can
            // transiently match while the descriptor sets still reference
            // old texture objects (deleted with deferred destruction, so they
            // keep serving a coherent STALE frame — seen as permanent image
            // retention with the CRT filter's glow composite).
            wr64PrevFrame = nullptr;
            wr64PrevFrameDescSet = nullptr;
            wr64PrevFrameWidth = 0;
            wr64PrevFrameHeight = 0;
            wr64PrevFrameValid = false;
            wr64Scratch = nullptr;
            wr64ScratchDescSet = nullptr;
            wr64ScratchWidth = 0;
            wr64ScratchHeight = 0;
            wr64CrtDescSet = nullptr;
            wr64CrtDescWidth = 0;
            wr64CrtDescHeight = 0;
            wr64CrtDescBoundScratch = nullptr;
            wr64SkipEffectsOnce = true;
        }
        
        for (int32_t i = 0; i < framesToPresent; i++) {
            uint32_t frameCountersNextPresented = 0;
            if ((framesToPresent > 1) && (usingMSAA || (i > 0))) {
                // Stall until the interpolated color target is available.
                const uint32_t targetIndex = usingMSAA ? i : (i - 1);
                std::unique_lock<std::mutex> interpolatedLock(ext.sharedResources->interpolatedMutex);
                ext.sharedResources->interpolatedCondition.wait(interpolatedLock, [&]() {
                    return (frameCounters.available > targetIndex) || ((frameCounters.available == targetIndex) && frameCounters.skipped);
                });

                // Do not present any more frames after this one after reaching the last available frame if the workload was skipped.
                if ((frameCounters.available == targetIndex) && frameCounters.skipped) {
                    framesToPresent = std::min(int(frameCounters.available), i + 1);
                    frameCountersNextPresented = frameCounters.count;
                }
                else {
                    frameCountersNextPresented = frameCounters.presented + 1;
                }

                if (i < framesToPresent) {
                    uint32_t targetIndex = usingMSAA ? i : (i - 1);
                    colorTarget = ext.sharedResources->interpolatedColorTargets[targetIndex].get();
                }
                else {
                    colorTarget = nullptr;
                }
            }
            else if (framesToPresent == 1) {
                frameCountersNextPresented = frameCounters.count;
            }

            uint32_t swapChainIndex = 0;
            const bool presentFrame = (i < framesToPresent) && swapChainValid;
            if (presentFrame) {
                swapChainValid = ext.swapChain->acquireTexture(acquiredSemaphore.get(), &swapChainIndex);
            }

            if (presentFrame && swapChainValid) {
                // Draw the framebuffer with the VI renderer.
                RenderTexture *swapChainTexture = ext.swapChain->getTexture(swapChainIndex);
                RenderFramebuffer *swapChainFramebuffer = swapChainFramebuffers[swapChainIndex].get();
                RenderCommandList *commandList = ext.presentGraphicsWorker->commandList.get();
                commandList->begin();
                commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(swapChainTexture, RenderTextureLayout::COLOR_WRITE));
                
                VIRenderer::RenderParams renderParams;
                if (colorTarget != nullptr) {
                    renderParams.device = ext.device;
                    renderParams.commandList = commandList;
                    renderParams.swapChain = ext.swapChain;
                    renderParams.shaderLibrary = ext.shaderLibrary;
                    renderParams.textureFormat = colorTarget->format;
                    renderParams.resolutionScale = colorTarget->resolutionScale;
                    renderParams.downsamplingScale = 1;
                    renderParams.filtering = filtering;
                    renderParams.vi = &present.screenVI;
                    renderParams.removeBlackBorders = removeBlackBorders;

                    const bool useDownsampling = (colorTarget->downsampleMultiplier > 1);
                    if (useDownsampling) {
                        colorTarget->downsampleTarget(ext.presentGraphicsWorker, ext.shaderLibrary);
                        renderParams.texture = colorTarget->downsampledTexture.get();
                        renderParams.textureWidth = colorTarget->width / colorTarget->downsampleMultiplier;
                        renderParams.textureHeight = colorTarget->height / colorTarget->downsampleMultiplier;
                        renderParams.downsamplingScale = colorTarget->downsampleMultiplier;
                    }
                    else {
                        colorTarget->resolveTarget(ext.presentGraphicsWorker, ext.shaderLibrary);
                        renderParams.texture = colorTarget->getResolvedTexture();
                        renderParams.textureWidth = colorTarget->width;
                        renderParams.textureHeight = colorTarget->height;
                    }
                }
                
                commandList->setFramebuffer(swapChainFramebuffer);
                commandList->clearColor();

                if (renderParams.texture != nullptr) {
                    commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(renderParams.texture, RenderTextureLayout::SHADER_READ));
                    viRenderer->render(renderParams);
                }

                // WR64 sharpen (CAS-lite): copy the freshly rendered frame to a
                // scratch texture and draw it back through the sharpen shader.
                // Runs before the motion blur (which then accumulates the
                // sharpened image) and before the UI draw hook (UI unaffected).
                {
                    const float sharpenK = rt64_wr64_get_sharpen();
                    const ShaderRecord &sharpenShader = ext.shaderLibrary->wr64Sharpen;
                    if (sharpenK > 0.0f && !wr64SkipEffectsOnce && renderParams.texture != nullptr && sharpenShader.pipeline != nullptr) {
                        const uint32_t scWidth = ext.swapChain->getWidth();
                        const uint32_t scHeight = ext.swapChain->getHeight();
                        if (wr64Scratch == nullptr || wr64ScratchWidth != scWidth || wr64ScratchHeight != scHeight) {
                            wr64Scratch = ext.device->createTexture(RenderTextureDesc::Texture2D(scWidth, scHeight, 1, RenderFormat::B8G8R8A8_UNORM));
                            wr64ScratchDescSet = std::make_unique<TextureCopyDescriptorSet>(ext.device);
                            wr64ScratchDescSet->setTexture(wr64ScratchDescSet->gInput, wr64Scratch.get(), RenderTextureLayout::SHADER_READ);
                            wr64ScratchWidth = scWidth;
                            wr64ScratchHeight = scHeight;
                        }

                        commandList->setFramebuffer(nullptr);
                        commandList->barriers(RenderBarrierStage::COPY, RenderTextureBarrier(swapChainTexture, RenderTextureLayout::COPY_SOURCE));
                        commandList->barriers(RenderBarrierStage::COPY, RenderTextureBarrier(wr64Scratch.get(), RenderTextureLayout::COPY_DEST));
                        commandList->copyTexture(wr64Scratch.get(), swapChainTexture);
                        commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(swapChainTexture, RenderTextureLayout::COLOR_WRITE));
                        commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(wr64Scratch.get(), RenderTextureLayout::SHADER_READ));
                        commandList->setFramebuffer(swapChainFramebuffer);

                        commandList->setPipeline(sharpenShader.pipeline.get());
                        commandList->setGraphicsPipelineLayout(sharpenShader.pipelineLayout.get());
                        commandList->setGraphicsDescriptorSet(wr64ScratchDescSet->get(), 0);

                        interop::WR64SharpenCB sharpenCB;
                        sharpenCB.texSize.x = float(scWidth);
                        sharpenCB.texSize.y = float(scHeight);
                        sharpenCB.strength = sharpenK;
                        sharpenCB.padding = 0.0f;
                        commandList->setGraphicsPushConstants(0, &sharpenCB);

                        const RenderViewport sharpenViewport(0.0f, 0.0f, float(scWidth), float(scHeight));
                        const RenderRect sharpenScissor(0, 0, int32_t(scWidth), int32_t(scHeight));
                        commandList->setViewports(sharpenViewport);
                        commandList->setScissors(sharpenScissor);
                        commandList->setVertexBuffers(0, nullptr, 0, nullptr);
                        commandList->drawInstanced(3, 1, 0, 0);
                    }
                }

                // WR64 motion blur (prototype): exponential accumulation at
                // present time. Draw the PREVIOUS presented frame over the
                // current one with constant alpha = strength, then capture the
                // blended result for the next present. Runs before the UI draw
                // hook so menus/overlays stay sharp.
                {
                    // Phosphor persistence: with the CRT filter active, a very
                    // small accumulation trail approximates the tube's phosphor
                    // decay (highlights linger for a few frames). Reuses the
                    // motion-blur machinery; an explicit Motion Blur setting
                    // still wins if it is stronger.
                    const float crtPersist = rt64_wr64_get_crt_persist() ? (rt64_wr64_get_crt() * 0.12f) : 0.0f;
                    const float blurK = std::max(rt64_wr64_get_motion_blur(), crtPersist);
                    const ShaderRecord &blurShader = ext.shaderLibrary->wr64MotionBlur;
                    if (blurK > 0.0f && !wr64SkipEffectsOnce && renderParams.texture != nullptr && blurShader.pipeline != nullptr) {
                        const uint32_t scWidth = ext.swapChain->getWidth();
                        const uint32_t scHeight = ext.swapChain->getHeight();
                        if (wr64PrevFrame == nullptr || wr64PrevFrameWidth != scWidth || wr64PrevFrameHeight != scHeight) {
                            wr64PrevFrame = ext.device->createTexture(RenderTextureDesc::Texture2D(scWidth, scHeight, 1, RenderFormat::B8G8R8A8_UNORM));
                            wr64PrevFrameDescSet = std::make_unique<TextureCopyDescriptorSet>(ext.device);
                            wr64PrevFrameDescSet->setTexture(wr64PrevFrameDescSet->gInput, wr64PrevFrame.get(), RenderTextureLayout::SHADER_READ);
                            wr64PrevFrameWidth = scWidth;
                            wr64PrevFrameHeight = scHeight;
                            wr64PrevFrameValid = false;
                        }

                        // Drop history when the game-image layout changed since
                        // the stored frame (fullscreen<->window, aspect/crop/
                        // border-mode switch). Blending a differently-placed
                        // previous frame would ghost/stretch (user-reported on
                        // the FS<->window toggle).
                        int32_t curX0 = 0, curY0 = 0, curX1 = 0, curY1 = 0;
                        rt64_wr64_get_content_rect(&curX0, &curY0, &curX1, &curY1);
                        if (curX0 != wr64PrevContentX0 || curY0 != wr64PrevContentY0 ||
                            curX1 != wr64PrevContentX1 || curY1 != wr64PrevContentY1) {
                            wr64PrevFrameValid = false;
                        }

                        if (wr64PrevFrameValid) {
                            commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(wr64PrevFrame.get(), RenderTextureLayout::SHADER_READ));
                            commandList->setPipeline(blurShader.pipeline.get());
                            commandList->setGraphicsPipelineLayout(blurShader.pipelineLayout.get());
                            commandList->setGraphicsDescriptorSet(wr64PrevFrameDescSet->get(), 0);

                            interop::WR64MotionBlurCB blurCB;
                            blurCB.uvScale.x = float(scWidth);
                            blurCB.uvScale.y = float(scHeight);
                            blurCB.alpha = blurK;
                            blurCB.padding = 0.0f;
                            commandList->setGraphicsPushConstants(0, &blurCB);

                            const RenderViewport blurViewport(0.0f, 0.0f, float(scWidth), float(scHeight));
                            const RenderRect blurScissor(0, 0, int32_t(scWidth), int32_t(scHeight));
                            commandList->setViewports(blurViewport);
                            commandList->setScissors(blurScissor);
                            commandList->setVertexBuffers(0, nullptr, 0, nullptr);
                            commandList->drawInstanced(3, 1, 0, 0);
                        }

                        // Capture the blended result for the next present.
                        commandList->setFramebuffer(nullptr);
                        commandList->barriers(RenderBarrierStage::COPY, RenderTextureBarrier(swapChainTexture, RenderTextureLayout::COPY_SOURCE));
                        commandList->barriers(RenderBarrierStage::COPY, RenderTextureBarrier(wr64PrevFrame.get(), RenderTextureLayout::COPY_DEST));
                        commandList->copyTexture(wr64PrevFrame.get(), swapChainTexture);
                        commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(swapChainTexture, RenderTextureLayout::COLOR_WRITE));
                        commandList->setFramebuffer(swapChainFramebuffer);
                        wr64PrevFrameValid = true;
                        wr64PrevContentX0 = curX0; wr64PrevContentY0 = curY0;
                        wr64PrevContentX1 = curX1; wr64PrevContentY1 = curY1;
                    }
                    else {
                        // Off (or nothing rendered): drop history so re-enabling
                        // starts from a fresh frame.
                        wr64PrevFrameValid = false;
                    }
                }

                // WR64 CRT filter (Trinitron look): LAST image pass — copy the
                // (possibly sharpened/blurred) frame to the scratch texture and
                // draw it back through the CRT shader. Running after the motion
                // blur keeps the phosphor mask off the accumulation history
                // ("the mask sits on the glass"); running before the UI draw
                // hook keeps the launcher/overlays clean. The shader confines
                // curvature/masks to the game content rect published by the VI
                // renderer (rt64_wr64_get_content_rect) — black bars stay flat.
                {
                    const float crtK = rt64_wr64_get_crt();
                    const bool crtBezel = rt64_wr64_get_crt_bezel() != 0;
                    const ShaderRecord &crtShader = ext.shaderLibrary->wr64Crt;
                    // Run the pass when the CRT look is on OR just the bezel is
                    // enabled (the bezel frames the image independently of the
                    // intensity slider).
                    if ((crtK > 0.0f || crtBezel) && !wr64SkipEffectsOnce && renderParams.texture != nullptr && crtShader.pipeline != nullptr) {
                        const uint32_t scWidth = ext.swapChain->getWidth();
                        const uint32_t scHeight = ext.swapChain->getHeight();
                        if (wr64Scratch == nullptr || wr64ScratchWidth != scWidth || wr64ScratchHeight != scHeight) {
                            wr64Scratch = ext.device->createTexture(RenderTextureDesc::Texture2D(scWidth, scHeight, 1, RenderFormat::B8G8R8A8_UNORM));
                            wr64ScratchDescSet = std::make_unique<TextureCopyDescriptorSet>(ext.device);
                            wr64ScratchDescSet->setTexture(wr64ScratchDescSet->gInput, wr64Scratch.get(), RenderTextureLayout::SHADER_READ);
                            wr64ScratchWidth = scWidth;
                            wr64ScratchHeight = scHeight;
                        }
                        if (wr64CrtDescSet == nullptr || wr64CrtDescWidth != scWidth || wr64CrtDescHeight != scHeight ||
                            wr64CrtDescBoundScratch != wr64Scratch.get()) {
                            wr64CrtDescSet = std::make_unique<VideoInterfaceDescriptorSet>(
                                ext.shaderLibrary->samplerLibrary.linear.borderBorder.get(), ext.device);
                            wr64CrtDescSet->setTexture(wr64CrtDescSet->gInput, wr64Scratch.get(), RenderTextureLayout::SHADER_READ);
                            wr64CrtDescBoundScratch = wr64Scratch.get();
                            wr64CrtDescWidth = scWidth;
                            wr64CrtDescHeight = scHeight;
                        }

                        commandList->setFramebuffer(nullptr);
                        commandList->barriers(RenderBarrierStage::COPY, RenderTextureBarrier(swapChainTexture, RenderTextureLayout::COPY_SOURCE));
                        commandList->barriers(RenderBarrierStage::COPY, RenderTextureBarrier(wr64Scratch.get(), RenderTextureLayout::COPY_DEST));
                        commandList->copyTexture(wr64Scratch.get(), swapChainTexture);
                        commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(swapChainTexture, RenderTextureLayout::COLOR_WRITE));
                        commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(wr64Scratch.get(), RenderTextureLayout::SHADER_READ));
                        commandList->setFramebuffer(swapChainFramebuffer);

                        commandList->setPipeline(crtShader.pipeline.get());
                        commandList->setGraphicsPipelineLayout(crtShader.pipelineLayout.get());
                        commandList->setGraphicsDescriptorSet(wr64CrtDescSet->get(), 0);

                        int32_t cx0 = 0, cy0 = 0, cx1 = 0, cy1 = 0;
                        rt64_wr64_get_content_rect(&cx0, &cy0, &cx1, &cy1);
                        interop::WR64CrtCB crtCB;
                        crtCB.rectMin.x = float(cx0);
                        crtCB.rectMin.y = float(cy0);
                        crtCB.rectMax.x = float(cx1);
                        crtCB.rectMax.y = float(cy1);
                        crtCB.texSize.x = float(scWidth);
                        crtCB.texSize.y = float(scHeight);
                        crtCB.srcRows = rt64_wr64_get_content_src_rows();
                        crtCB.intensity = crtK;
                        crtCB.bezel = crtBezel ? 1.0f : 0.0f;
                        crtCB.pad0 = crtCB.pad1 = crtCB.pad2 = 0.0f;
                        commandList->setGraphicsPushConstants(0, &crtCB);

                        const RenderViewport crtViewport(0.0f, 0.0f, float(scWidth), float(scHeight));
                        const RenderRect crtScissor(0, 0, int32_t(scWidth), int32_t(scHeight));
                        commandList->setViewports(crtViewport);
                        commandList->setScissors(crtScissor);
                        commandList->setVertexBuffers(0, nullptr, 0, nullptr);
                        commandList->drawInstanced(3, 1, 0, 0);
                    }
                }

                // Effects sat this present out if the swapchain was just
                // rebuilt (see wr64SkipEffectsOnce); they resume next present.
                wr64SkipEffectsOnce = false;

                RenderHookDraw *drawHook = GetRenderHookDraw();
                if (drawHook != nullptr) {
                    drawHook(commandList, swapChainFramebuffer);
                }

                // WR64 screenshot: copy the FINAL frame (game + UI) into a
                // readback buffer; converted + written to PNG after the fence
                // below. Requested via rt64_wr64_request_screenshot (F12).
                std::string wr64ShotPathTaken;
                uint32_t wr64ShotW = 0, wr64ShotH = 0, wr64ShotPitch = 0;
                const bool wr64TakeShot = rt64_wr64_take_screenshot_path(wr64ShotPathTaken);

                {
                    const std::scoped_lock lock(inspectorMutex);
                    if (inspector != nullptr) {
                        inspector->draw(commandList);
                    }

                    if (wr64TakeShot) {
                        wr64ShotW = ext.swapChain->getWidth();
                        wr64ShotH = ext.swapChain->getHeight();
                        wr64ShotPitch = (wr64ShotW * 4 + 255) & ~255u;   // 256-byte aligned rows
                        fprintf(stderr, "[WR64] screenshot: capturing %ux%u -> %s\n", wr64ShotW, wr64ShotH, wr64ShotPathTaken.c_str());
                        wr64ShotBuffer = ext.device->createBuffer(RenderBufferDesc::ReadbackBuffer(uint64_t(wr64ShotPitch) * wr64ShotH));
                        if (wr64ShotBuffer == nullptr) {
                            fprintf(stderr, "[WR64] screenshot: readback buffer creation FAILED\n");
                        }
                        commandList->setFramebuffer(nullptr);
                        commandList->barriers(RenderBarrierStage::COPY, RenderTextureBarrier(swapChainTexture, RenderTextureLayout::COPY_SOURCE));
                        RenderTextureCopyLocation shotDst = RenderTextureCopyLocation::PlacedFootprint(wr64ShotBuffer.get(), RenderFormat::B8G8R8A8_UNORM, wr64ShotW, wr64ShotH, 1, wr64ShotPitch / 4);
                        // plume D3D12 quirk: copyTextureRegion calls
                        // setSamplePositions(dst.texture), which null-derefs for
                        // buffer destinations. The field is otherwise ignored for
                        // placed footprints, so hand it a harmless non-MSAA
                        // texture instead of patching the plume submodule.
                        shotDst.texture = swapChainTexture;
                        commandList->copyTextureRegion(shotDst, RenderTextureCopyLocation::Subresource(swapChainTexture));
                    }

                    commandList->barriers(RenderBarrierStage::NONE, RenderTextureBarrier(swapChainTexture, RenderTextureLayout::PRESENT));
                    commandList->end();
                    const RenderCommandList *commandList = ext.presentGraphicsWorker->commandList.get();
                    RenderCommandSemaphore *waitSemaphore = acquiredSemaphore.get();
                    RenderCommandSemaphore *signalSemaphore = drawSemaphores[swapChainIndex].get();
                    ext.presentGraphicsWorker->commandQueue->executeCommandLists(&commandList, 1, &waitSemaphore, 1, &signalSemaphore, 1, ext.presentGraphicsWorker->commandFence.get());
                    ext.presentGraphicsWorker->wait();
                }

                if (wr64TakeShot && wr64ShotBuffer != nullptr) {
                    const uint8_t *src = reinterpret_cast<const uint8_t *>(wr64ShotBuffer->map());
                    if (src == nullptr) {
                        fprintf(stderr, "[WR64] screenshot: readback map FAILED\n");
                    }
                    if (src != nullptr) {
                        // BGRA (swap chain) -> RGBA, dropping the row padding.
                        std::vector<uint8_t> rgba(size_t(wr64ShotW) * wr64ShotH * 4);
                        for (uint32_t y = 0; y < wr64ShotH; y++) {
                            const uint8_t *row = src + size_t(y) * wr64ShotPitch;
                            uint8_t *dst = rgba.data() + size_t(y) * wr64ShotW * 4;
                            for (uint32_t x = 0; x < wr64ShotW; x++) {
                                dst[x * 4 + 0] = row[x * 4 + 2];
                                dst[x * 4 + 1] = row[x * 4 + 1];
                                dst[x * 4 + 2] = row[x * 4 + 0];
                                dst[x * 4 + 3] = 0xFF;
                            }
                        }
                        wr64ShotBuffer->unmap();

                        std::thread([path = std::move(wr64ShotPathTaken), w = wr64ShotW, h = wr64ShotH, data = std::move(rgba)]() {
                            if (stbi_write_png(path.c_str(), int(w), int(h), 4, data.data(), int(w) * 4) != 0) {
                                fprintf(stderr, "[WR64] screenshot saved: %s\n", path.c_str());
                            } else {
                                fprintf(stderr, "[WR64] screenshot FAILED to write: %s\n", path.c_str());
                            }
                        }).detach();
                    }
                    wr64ShotBuffer.reset();
                }
            }

            if (lockedWorkloadMutex) {
                ext.sharedResources->workloadMutex.unlock();
                lockedWorkloadMutex = false;
            }
            
            if (frameCountersNextPresented > 0) {
                {
                    std::unique_lock<std::mutex> interpolatedLock(ext.sharedResources->interpolatedMutex);
                    frameCounters.presented = frameCountersNextPresented;
                }

                ext.sharedResources->interpolatedCondition.notify_all();
            }

            // As soon as we're done with the first render target, we notify the workload queue it can proceed.
            if (i == 0) {
                notifyPresentId(present);
            }

            if (presentFrame && swapChainValid) {
                // Wait until the approximate time the next present should be at the current intended rate.
                if ((presentTimestamp != Timestamp()) && (targetRate > 0) && (targetRate > viOriginalRate)) {
                    Timer::preciseSleepUntil(presentTimestamp + std::chrono::nanoseconds(1'000'000'000 / targetRate));
                }

                if (presentWaitEnabled) {
                    ext.swapChain->wait();
                }

                RenderCommandSemaphore *waitSemaphore = drawSemaphores[swapChainIndex].get();
                presentTimestamp = Timer::current();
                swapChainValid = ext.swapChain->present(swapChainIndex, &waitSemaphore, 1);
                presentProfiler.logAndRestart();
            }
        }
    }

    void PresentQueue::skipInterpolation() {
        {
            std::unique_lock<std::mutex> interpolatedLock(ext.sharedResources->interpolatedMutex);
            InterpolatedFrameCounters &frameCounters = ext.sharedResources->interpolatedFrames[ext.sharedResources->interpolatedFramesIndex];
            frameCounters.presented = frameCounters.count;
        }

        ext.sharedResources->interpolatedCondition.notify_all();
    }

    void PresentQueue::notifyPresentId(const Present &present) {
        {
            std::scoped_lock<std::mutex> cursorLock(presentIdMutex);
            presentId = present.presentId;
        }

        presentIdCondition.notify_all();
    }
    
    void PresentQueue::threadAdvanceBarrier() {
        std::scoped_lock<std::mutex> cursorLock(cursorMutex);
        barrierCursor = (barrierCursor + 1) % presents.size();
    }

    void PresentQueue::threadLoop() {
        Thread::setCurrentThreadName("RT64 Present");

        // Create the semaphore the acquire method will use.
        acquiredSemaphore = ext.device->createCommandSemaphore();

        // Create as many semaphores to signal as textures there are.
        while (drawSemaphores.size() < ext.swapChain->getTextureCount()) {
            drawSemaphores.emplace_back(ext.device->createCommandSemaphore());
        }

        // Since the swap chain might not need a resize right away, detect present wait.
        presentWaitEnabled = ext.device->getCapabilities().presentWait;

        int processCursor = -1;
        bool skipPresent = false;
        uint32_t displayTimingRate = UINT32_MAX;
        const bool displayTiming = ext.device->getCapabilities().displayTiming;
        bool swapChainValid = !ext.swapChain->needsResize();
        while (presentThreadRunning) {
            {
                std::unique_lock<std::mutex> cursorLock(cursorMutex);
                cursorCondition.wait(cursorLock, [&]() {
                    return (writeCursor != threadCursor) || !presentThreadRunning;
                });

                if (presentThreadRunning) {
                    processCursor = threadCursor;
                    threadCursor = (threadCursor + 1) % presents.size();
                    skipPresent = (writeCursor != threadCursor);
                }
            }

            if (processCursor >= 0) {
                std::unique_lock<std::mutex> threadLock(threadMutex);
                const bool needsResize = ext.swapChain->needsResize() || !swapChainValid;
                if (needsResize) {
                    ext.presentGraphicsWorker->commandList->begin();
                    ext.presentGraphicsWorker->commandList->end();
                    ext.presentGraphicsWorker->execute();
                    ext.presentGraphicsWorker->wait();
                    swapChainValid = ext.swapChain->resize();
                    swapChainFramebuffers.clear();

                    if (swapChainValid) {
                        ext.sharedResources->setSwapChainSize(ext.swapChain->getWidth(), ext.swapChain->getHeight());
                        
                        // Texture count could've changed after resize, so new semaphores are needed.
                        while (drawSemaphores.size() < ext.swapChain->getTextureCount()) {
                            drawSemaphores.emplace_back(ext.device->createCommandSemaphore());
                        }
                    }
                }

                if (needsResize || ext.appWindow->detectWindowMoved()) {
                    ext.appWindow->detectRefreshRate();
                    ext.sharedResources->setSwapChainRate(std::min(ext.appWindow->getRefreshRate(), displayTimingRate));
                }

                if (displayTiming) {
                    uint32_t newDisplayTimingRate = ext.swapChain->getRefreshRate();
                    if (newDisplayTimingRate == 0) {
                        newDisplayTimingRate = UINT32_MAX;
                    }

                    if (newDisplayTimingRate != displayTimingRate) {
                        ext.sharedResources->setSwapChainRate(std::min(ext.appWindow->getRefreshRate(), newDisplayTimingRate));
                        displayTimingRate = newDisplayTimingRate;
                    }
                }

                skipPresent = skipPresent || ext.swapChain->isEmpty();

                Present &present = presents[processCursor];
                ext.workloadQueue->waitForWorkloadId(present.workloadId);

                if (!presentThreadRunning) {
                    continue;
                }

                if (skipPresent) {
                    skipInterpolation();
                    notifyPresentId(present);
                }
                else {
                    threadPresent(present, swapChainValid);
                }

                if (!present.paused) {
                    if (!present.fbOperations.empty()) {
                        const std::scoped_lock lock(screenFbChangePoolMutex);
                        screenFbChangePool.release(present.fbOperations.front().writeChanges.id);
                        present.fbOperations.clear();
                    }

                    threadAdvanceBarrier();
                }

                processCursor = -1;
            }
        }

        // Transition the active swap chain render target out of the present state to avoid live references to the resource.
        uint32_t swapChainIndex = 0;
        if (!ext.swapChain->isEmpty() && ext.swapChain->acquireTexture(acquiredSemaphore.get(), &swapChainIndex)) {
            RenderTexture *swapChainTexture = ext.swapChain->getTexture(swapChainIndex);
            ext.presentGraphicsWorker->commandList->begin();
            ext.presentGraphicsWorker->commandList->barriers(RenderBarrierStage::NONE, RenderTextureBarrier(swapChainTexture, RenderTextureLayout::COLOR_WRITE));
            ext.presentGraphicsWorker->commandList->end();

            const RenderCommandList *commandList = ext.presentGraphicsWorker->commandList.get();
            RenderCommandSemaphore *waitSemaphore = acquiredSemaphore.get();
            ext.presentGraphicsWorker->commandQueue->executeCommandLists(&commandList, 1, &waitSemaphore, 1, nullptr, 0, ext.presentGraphicsWorker->commandFence.get());
            ext.presentGraphicsWorker->wait();
        }
    }
};
