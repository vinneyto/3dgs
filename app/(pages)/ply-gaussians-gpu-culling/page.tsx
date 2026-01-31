"use client";

import { usePlyPackedRust } from "@/app/_shared/hooks/usePlyPackedRust";
import { IndirectCulledSplatScene } from "@/app/_shared/scenes/IndirectCulledSplatScene";
import { WebGPUCanvasFrame } from "@/app/_shared/webgpu/WebGPUCanvasFrame";

const PLY_URL = "/cactus_splat3_30kSteps_142k_splats.ply";

export default function PlyGaussiansGpuCullingPage() {
  const { status, data } = usePlyPackedRust(PLY_URL);

  return (
    <div className="page">
      <div className="pageHeader">
        <h1>PLY gaussians: GPU culling + prefix-sum + indirect draw</h1>
        <p className="muted">
          Loads <code>{PLY_URL}</code> into storage buffers, runs frustum culling
          on the GPU, compacts visible indices via prefix-sum, sorts visible
          splats by depth, and renders via <code>drawIndexedIndirect</code>{" "}
          (instanceCount computed on GPU).
        </p>
        <div className="muted">
          File: <code>app/(pages)/ply-gaussians-gpu-culling/page.tsx</code>
        </div>
        <div className="muted">
          Status: <code>{status}</code>
        </div>
      </div>

      {data ? (
        <WebGPUCanvasFrame
          camera={{ position: [4, 3, 4], fov: 50, near: 0.1, far: 100 }}
          gl={{ antialias: false }}
        >
          <IndirectCulledSplatScene data={data} controlsGroup="GPU Culling" />
        </WebGPUCanvasFrame>
      ) : null}
    </div>
  );
}

