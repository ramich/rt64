//
// RT64
//

#include "rt64_projection_processor.h"

#include "common/rt64_math.h"
#include "hle/rt64_workload_queue.h"

// WR64: present-crop flag — set while a 4:3 menu is being pillarboxed. The
// split-screen FOV correction below must NOT run for menus (they are cropped),
// only for live gameplay (2P race, uncropped), or it distorts menu/2D layout.
extern "C" int rt64_wr64_get_present_crop43();

namespace RT64 {
    inline void adjustProjectionMatrix(interop::float4x4 &matrix, const float aspectRatioScale) {
        matrix[0][0] *= aspectRatioScale;
        matrix[1][0] *= aspectRatioScale;
        matrix[2][0] *= aspectRatioScale;
        matrix[3][0] *= aspectRatioScale;
    }
    
    // ProjectionProcessor

    ProjectionProcessor::ProjectionProcessor() { }

    ProjectionProcessor::~ProjectionProcessor() {
        bufferUploader.reset(nullptr);
    }

    void ProjectionProcessor::setup(RenderWorker *worker) {
        bufferUploader = std::make_unique<BufferUploader>(worker->device);
    }

    void ProjectionProcessor::process(const ProcessParams &p) {
        for (uint32_t w : p.curFrame->workloads) {
            Workload &workload = p.workloadQueue->workloads[w];
            DrawData &drawData = workload.drawData;

            // Copy the data.
            drawData.modViewTransforms = drawData.viewTransforms;
            drawData.modProjTransforms = drawData.projTransforms;
            drawData.modViewProjTransforms = drawData.viewProjTransforms;
            drawData.prevViewTransforms = drawData.viewTransforms;
            drawData.prevProjTransforms = drawData.projTransforms;
            drawData.prevViewProjTransforms = drawData.viewProjTransforms;
        }

        for (size_t s = 0; s < p.curFrame->perspectiveScenes.size(); s++) {
            processScene(p, p.curFrame->perspectiveScenes[s], s);
        }

        for (size_t s = 0; s < p.curFrame->orthographicScenes.size(); s++) {
            processScene(p, p.curFrame->orthographicScenes[s], s);
        }
    }

    void ProjectionProcessor::processScene(const ProcessParams &p, const GameScene &scene, size_t sceneIndex) {
        for (size_t i = 0; i < scene.projections.size(); i++) {
            const GameIndices::Projection &sceneProj = scene.projections[i];
            Workload &workload = p.workloadQueue->workloads[sceneProj.workloadIndex];
            DrawData &drawData = workload.drawData;
            const FramebufferPair &fbPair = workload.fbPairs[sceneProj.fbPairIndex];
            const Projection &proj = fbPair.projections[sceneProj.projectionIndex];
            const uint16_t viewportOrigin = drawData.viewportOrigins[proj.transformsIndex];
            assert(proj.transformsIndex > 0);

            // Skip projections that didn't actually draw anything.
            if (proj.scissorRect.isNull()) {
                continue;
            }

            // Check the current mapping for the projection.
            const interop::float4x4 *prevProjMatrix = nullptr;
            const interop::float4x4 *prevViewMatrix = nullptr;
            const RigidBody *rigidBody = nullptr;
            const GameFrameMap::WorkloadMap &workloadMap = p.curFrame->frameMap.workloads[sceneProj.workloadIndex];
            if ((p.prevFrame != nullptr) && workloadMap.mapped && !workload.debuggerCamera.enabled) {
                const GameFrameMap::ViewProjectionMap &viewProjMap = workloadMap.viewProjections[proj.transformsIndex];
                if (viewProjMap.mapped) {
                    const Workload &prevWorkload = p.workloadQueue->workloads[workloadMap.prevWorkloadIndex];
                    prevViewMatrix = &prevWorkload.drawData.viewTransforms[viewProjMap.prevTransformIndex];
                    prevProjMatrix = &prevWorkload.drawData.projTransforms[viewProjMap.prevTransformIndex];
                    rigidBody = &viewProjMap.rigidBody;
                }
            }

            const uint32_t curProjGroupIndex = workload.drawData.viewProjTransformGroups[proj.transformsIndex];
            const TransformGroup &curProjGroup = workload.drawData.transformGroups[curProjGroupIndex];
            bool adjustAspectRatio = (curProjGroup.aspectMode == G_EX_ASPECT_ADJUST);
            if (curProjGroup.aspectMode == G_EX_ASPECT_AUTO) {
                FixedRect intersectionRect = proj.scissorRect;
                if (proj.usesViewport()) {
                    const interop::RSPViewport &viewport = drawData.rspViewports[proj.transformsIndex];
                    const int16_t *viewportClipRatios = &drawData.viewportClipRatios[proj.transformsIndex * 4];
                    intersectionRect = intersectionRect.intersection(viewport.rect(viewportClipRatios));
                }

                if (!intersectionRect.isEmpty()) {
                    bool coversWholeWidth = (intersectionRect.ulx <= fbPair.scissorRect.ulx) && (intersectionRect.lrx >= fbPair.scissorRect.lrx);
                    bool horizontalRatio = (intersectionRect.width(true, true) > intersectionRect.height(true, true));
                    adjustAspectRatio = (viewportOrigin == G_EX_ORIGIN_NONE) && coversWholeWidth && horizontalRatio;
                }
            }
 
            // Env override paired with the rect-aspect default in rt64_state:
            // when the port requests stretched 2D, orthographic scenes skip the
            // aspect compensation (perspective scenes keep it — that is what
            // produces the widescreen expansion).
            static const bool stretchOrtho = []() {
                const char *v = getenv("RT64_RECT_ASPECT_DEFAULT");
                return v != nullptr && strcmp(v, "stretch") == 0;
            }();
            if (stretchOrtho && (proj.type == Projection::Type::Orthographic)) {
                adjustAspectRatio = false;
            }

            // WR64: while the present is cropped to 4:3 (menu presentation),
            // perspective projections must NOT be FOV-widened. The framebuffer
            // renderer forces everything onto the squeezed path while cropped
            // (screenScale.x = 1/aspectRatioScale places content at its 4:3
            // positions), so a widened matrix gets squeezed TWICE — the
            // watercraft-select preview craft rendered 1/ars too narrow
            // (measured 0.56x at a 2.4:1 window, exactly 1/1.8). With the
            // matrix untouched, the single squeeze is the correct placement.
            if ((proj.type == Projection::Type::Perspective) &&
                (rt64_wr64_get_present_crop43() != 0)) {
                adjustAspectRatio = false;
            }

            float projRatioScale = adjustAspectRatio ? (1.0f / p.aspectRatioScale) : 1.0f;
            // WR64 2P split-screen: each half is a half-height, full-width
            // perspective viewport inset by the game's ~32px border, so the
            // stock coversWholeWidth gate above rejects it and it gets NO
            // horizontal-FOV widening -> the 4:3 world is stretched to fill the
            // wide half (fat riders). Give it the same widening a 1P full-frame
            // view gets (see below). Gated tightly so it only fires for live
            // gameplay (never menus, which are pillarboxed) and only for a
            // genuine stacked split half: near-full-width, height in a tight
            // band around HALF the frame, and flush to top or starting near
            // mid-frame (not a centered banner). Full-height 1P views never
            // match and are unaffected. Kept in sync with
            // rt64_framebuffer_renderer.cpp.
            if ((proj.type == Projection::Type::Perspective) && !proj.scissorRect.isNull() &&
                (rt64_wr64_get_present_crop43() == 0)) {
                const int32_t wr64PairW = fbPair.scissorRect.lrx - fbPair.scissorRect.ulx;
                const int32_t wr64PairH = fbPair.scissorRect.lry - fbPair.scissorRect.uly;
                const int32_t wr64ProjH = proj.scissorRect.lry - proj.scissorRect.uly;
                const int32_t wr64ProjTop = proj.scissorRect.uly - fbPair.scissorRect.uly;
                const int32_t wr64Tol = wr64PairW / 12;
                const bool wr64NearFullWidth =
                    (proj.scissorRect.ulx <= fbPair.scissorRect.ulx + wr64Tol) &&
                    (proj.scissorRect.lrx >= fbPair.scissorRect.lrx - wr64Tol);
                const bool wr64HalfHeight = (wr64PairH > 0) && (wr64ProjH > 0) &&
                    (wr64ProjH * 100 >= wr64PairH * 38) && (wr64ProjH * 100 <= wr64PairH * 55);
                const bool wr64TopOrMid = (wr64ProjTop <= wr64PairH / 10) ||   // flush to top
                    (wr64ProjTop * 100 >= wr64PairH * 40 && wr64ProjTop * 100 <= wr64PairH * 55); // starts ~mid
                if (wr64NearFullWidth && wr64HalfHeight && wr64TopOrMid) {
                    // Give the split half the SAME horizontal-FOV widening as a
                    // 1P full-frame view. (The stock coversWholeWidth gate denies
                    // it because the half is border-inset, leaving it un-widened
                    // and stretched fat.) No extra half-boost: the game already
                    // compensates its split-screen vertical FOV, so an extra
                    // projH/pairH factor over-widens (thin/stretched riders,
                    // verified via an empirical sweep 2026-07-20).
                    projRatioScale = 1.0f / p.aspectRatioScale;
                }
            }
            interop::float4x4 &viewMatrix = drawData.modViewTransforms[proj.transformsIndex];
            interop::float4x4 &projMatrix = drawData.modProjTransforms[proj.transformsIndex];
            interop::float4x4 &viewProjMatrix = drawData.modViewProjTransforms[proj.transformsIndex];
            viewMatrix = drawData.viewTransforms[proj.transformsIndex];
            projMatrix = drawData.projTransforms[proj.transformsIndex];
            viewProjMatrix = drawData.viewProjTransforms[proj.transformsIndex];

            // Debugger camera.
            if (workload.debuggerCamera.enabled && (proj.type == Projection::Type::Perspective) && (workload.debuggerCamera.sceneIndex == sceneIndex)) {
                viewMatrix = workload.debuggerCamera.viewMatrix;
                projMatrix = workload.debuggerCamera.projMatrix;
            }

            adjustProjectionMatrix(projMatrix, projRatioScale);

            interop::float4x4 &prevViewTransform = drawData.prevViewTransforms[proj.transformsIndex];
            interop::float4x4 &prevProjTransform = drawData.prevProjTransforms[proj.transformsIndex];
            if ((prevProjMatrix != nullptr) && (prevViewMatrix != nullptr) && (rigidBody != nullptr)) {
                const interop::float4x4 curViewTransform = viewMatrix;
                const interop::float4x4 curProjTransform = projMatrix;
                interop::float4x4 adjustedPrevProj = *prevProjMatrix;
                adjustProjectionMatrix(adjustedPrevProj, projRatioScale);
                viewMatrix = rigidBody->lerp(p.curFrameWeight, *prevViewMatrix, curViewTransform, true);
                prevViewTransform = rigidBody->lerp(p.prevFrameWeight, *prevViewMatrix, curViewTransform, true);

                // We only interpolate the projection if the view matrix has been interpolated.
                const bool interpolateProjection = rigidBody->lerpTranslation || rigidBody->lerpRotation;
                if (interpolateProjection) {
                    projMatrix = lerpMatrix(adjustedPrevProj, curProjTransform, p.curFrameWeight);
                    prevProjTransform = lerpMatrix(adjustedPrevProj, curProjTransform, p.prevFrameWeight);
                }
                else {
                    projMatrix = curProjTransform;
                    prevProjTransform = curProjTransform;
                }
            }
            else {
                prevViewTransform = viewMatrix;
                prevProjTransform = projMatrix;
            }

            viewProjMatrix = hlslpp::mul(viewMatrix, projMatrix);

            interop::float4x4 &prevViewProjTransform = drawData.prevViewProjTransforms[proj.transformsIndex];
            prevViewProjTransform = hlslpp::mul(prevViewTransform, prevProjTransform);
        }
    }

    void ProjectionProcessor::upload(const ProcessParams &p) {
        uploads.clear();

        for (uint32_t w : p.curFrame->workloads) {
            Workload &workload = p.workloadQueue->workloads[w];
            const DrawData &drawData = workload.drawData;
            DrawBuffers &drawBuffers = workload.drawBuffers;
            std::pair<size_t, size_t> uploadRange = { 0, drawData.viewProjTransforms.size() };
            uploads.emplace_back(BufferUploader::Upload{ drawData.modViewProjTransforms.data(), uploadRange, sizeof(interop::float4x4), RenderBufferFlag::STORAGE, { }, &drawBuffers.viewProjTransformsBuffer });
        }

        bufferUploader->submit(p.worker, uploads);
    }
};