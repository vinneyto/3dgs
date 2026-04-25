"use client";

import { usePlyPackedRust } from "@/app/_shared/hooks/usePlyPackedRust";
import { NewSplatScene } from "@/app/_shared/scenes/NewSplatScene";
import { WebGPUCanvasFrame } from "@/app/_shared/webgpu/WebGPUCanvasFrame";

const PLY_URL = "/cactus_splat3_30kSteps_142k_splats.ply";

export default function PlyGaussiansPage() {
  const { status, data } = usePlyPackedRust(PLY_URL);

  return (
    <div className="page">
      <div className="pageHeader">
        <h1>PLY gaussians (WIP)</h1>
        <p className="muted">
          Setup-only demo: loads PLY into storage buffers and computes depth
          keys + radix-sorted indices. Rendering will be added step-by-step
          next.
        </p>
        <div className="muted">
          File: <code>app/(pages)/ply-gaussians/page.tsx</code>
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
          <NewSplatScene data={data} controlsGroup="PLY gaussians" />
        </WebGPUCanvasFrame>
      ) : null}
    </div>
  );
}
