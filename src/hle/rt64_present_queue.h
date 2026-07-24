//
// RT64
//

#pragma once

#include "common/rt64_profiling_timer.h"
#include "gui/rt64_inspector.h"
#include "render/rt64_texture.h"
#include "render/rt64_vi_renderer.h"

#include "rt64_application_window.h"
#include "rt64_present.h"
#include "rt64_shared_queue_resources.h"

#define PRESENT_QUEUE_SIZE 4

namespace RT64 {
    struct WorkloadQueue;

    struct PresentQueue {
        struct External {
            ApplicationWindow *appWindow = nullptr;
            RenderDevice *device = nullptr;
            RenderSwapChain *swapChain = nullptr;
            RenderWorker *presentGraphicsWorker = nullptr;
            WorkloadQueue *workloadQueue = nullptr;
            SharedQueueResources *sharedResources = nullptr;
            const ShaderLibrary *shaderLibrary = nullptr;
            UserConfiguration::GraphicsAPI createdGraphicsAPI = UserConfiguration::GraphicsAPI::OptionCount;
        };

        External ext;
        std::array<Present, PRESENT_QUEUE_SIZE> presents;
        int threadCursor;
        int writeCursor;
        int barrierCursor;
        std::mutex cursorMutex;
        std::condition_variable cursorCondition;
        uint64_t presentId;
        std::mutex presentIdMutex;
        std::condition_variable presentIdCondition;
        std::thread *presentThread = nullptr;
        std::mutex threadMutex;
        std::atomic<bool> presentThreadRunning = false;
        std::recursive_mutex inspectorMutex;
        std::mutex screenFbChangePoolMutex;
        Framebuffer scratchFb;
        FramebufferChangePool scratchFbChangePool;
        FramebufferChangePool screenFbChangePool;
        std::atomic<bool> viewRDRAM = false;
        std::vector<std::unique_ptr<RenderFramebuffer>> swapChainFramebuffers;
        // WR64 fork: previous presented frame for the motion-blur prototype.
        std::unique_ptr<RenderTexture> wr64PrevFrame;
        std::unique_ptr<TextureCopyDescriptorSet> wr64PrevFrameDescSet;
        uint32_t wr64PrevFrameWidth = 0;
        uint32_t wr64PrevFrameHeight = 0;
        bool wr64PrevFrameValid = false;
        // Content rect of the frame captured into wr64PrevFrame; if the layout
        // changes (fullscreen<->window, aspect/crop switch) the stored frame no
        // longer matches and blending it would ghost, so history is dropped.
        int32_t wr64PrevContentX0 = 0, wr64PrevContentY0 = 0, wr64PrevContentX1 = 0, wr64PrevContentY1 = 0;
        // WR64 fork: scratch copy of the current frame for the sharpen pass
        // (reused by the CRT pass, which runs after it).
        std::unique_ptr<RenderTexture> wr64Scratch;
        std::unique_ptr<TextureCopyDescriptorSet> wr64ScratchDescSet;
        uint32_t wr64ScratchWidth = 0;
        uint32_t wr64ScratchHeight = 0;
        // WR64 fork: sampler-equipped descriptor set for the CRT pass (the
        // curvature + inline halation need linear sampling; the
        // TextureCopyDescriptorSet used by sharpen has no sampler). The
        // phosphor glow is now computed inline in the shader — no separate
        // glow render target (that second target retained stale memory across
        // window resizes and produced permanent garbage bands).
        std::unique_ptr<WR64CrtDescriptorSet> wr64CrtDescSet;
        uint32_t wr64CrtDescWidth = 0;
        uint32_t wr64CrtDescHeight = 0;
        // WR64 fork: CRT bezel overlay image (assets/crt_bezel.png). Decoded
        // once via TextureCache::loadTextureFromBytes and kept resident — it is
        // window-size independent, so it must NOT be dropped on a swapchain
        // rebuild (only wr64Scratch is). Bound to the CRT descriptor's gBezel
        // slot; if the file is missing the bezel silently stays off.
        std::unique_ptr<Texture> wr64BezelTex;
        std::unique_ptr<RenderBuffer> wr64BezelUpload;
        bool wr64BezelLoadTried = false;
        // WR64 fork: heavily downsampled (1/16) copy of the frame, box-filtered
        // via the shared boxFilter compute pipeline, sampled as the bezel's
        // diffuse screen reflection (a per-pixel grid blur couldn't kill large-
        // feature "mirror" detail; a real downsample does).
        std::unique_ptr<RenderTexture> wr64ReflSmall;
        std::unique_ptr<BoxFilterDescriptorSet> wr64ReflDescSet;
        uint32_t wr64ReflWidth = 0;
        uint32_t wr64ReflHeight = 0;
        // Object identity of the scratch texture the CRT descriptor set was
        // bound to — size-only tracking can miss a reallocation (the scratch
        // is also lazily recreated by the sharpen pass).
        const RenderTexture *wr64CrtDescBoundScratch = nullptr;
        // Set when the swapchain was (re)created for this present: the effect
        // passes sit the frame out (plain present). During a drag-resize the
        // swapchain is rebuilt almost every present and freshly allocated
        // textures hold recycled VRAM — compositing them artifacts visibly.
        bool wr64SkipEffectsOnce = false;
        // WR64 fork: pending screenshot readback (created on request).
        std::unique_ptr<RenderBuffer> wr64ShotBuffer;
        std::unique_ptr<RenderCommandSemaphore> acquiredSemaphore;
        std::vector<std::unique_ptr<RenderCommandSemaphore>> drawSemaphores;
        std::unique_ptr<VIRenderer> viRenderer;
        std::unique_ptr<Inspector> inspector;
        ProfilingTimer presentProfiler = ProfilingTimer(120);
        Timestamp presentTimestamp;
        VIHistory viHistory;
        bool presentWaitEnabled = false;

        PresentQueue();
        ~PresentQueue();
        void reset();
        void advanceToNextPresent();
        void repeatLastPresent();
        uint32_t previousWriteCursor() const;
        void waitForIdle();
        void waitForPresentId(uint64_t waitId);
        void setup(const External &ext);
        void threadPresent(const Present &present, bool &swapChainValid);
        void skipInterpolation();
        void notifyPresentId(const Present &present);
        void threadAdvanceBarrier();
        void threadLoop();
    };
};