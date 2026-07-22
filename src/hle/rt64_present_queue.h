//
// RT64
//

#pragma once

#include "common/rt64_profiling_timer.h"
#include "gui/rt64_inspector.h"
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
        // curvature needs linear sampling; TextureCopyDescriptorSet has none)
        // plus the quarter-res phosphor-glow target its prepass renders into.
        std::unique_ptr<WR64CrtDescriptorSet> wr64CrtDescSet;
        uint32_t wr64CrtDescWidth = 0;
        uint32_t wr64CrtDescHeight = 0;
        std::unique_ptr<RenderTexture> wr64Glow;
        std::unique_ptr<RenderFramebuffer> wr64GlowFb;
        std::unique_ptr<VideoInterfaceDescriptorSet> wr64GlowDescSet;
        uint32_t wr64GlowWidth = 0;
        uint32_t wr64GlowHeight = 0;
        // Object identity of the textures the CRT/glow descriptor sets were
        // bound to — size-only tracking can miss a reallocation (the scratch
        // is also lazily recreated by the sharpen pass) and leave a set
        // referencing a stale texture.
        const RenderTexture *wr64GlowDescBoundScratch = nullptr;
        const RenderTexture *wr64CrtDescBoundScratch = nullptr;
        const RenderTexture *wr64CrtDescBoundGlow = nullptr;
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