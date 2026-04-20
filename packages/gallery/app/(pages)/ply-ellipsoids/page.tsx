"use client";

import { usePlyPackedRust } from "@/app/_shared/hooks/usePlyPackedRust";
import { SplatScene } from "@/app/_shared/scenes/SplatScene";
import { WebGPUCanvasFrame } from "@/app/_shared/webgpu/WebGPUCanvasFrame";

const PLY_URL = "/cactus_splat3_30kSteps_142k_splats.ply";

export default function PlyEllipsoidsPage() {
  const { status, data } = usePlyPackedRust(PLY_URL);

  return (
    <div className="page">
      <div className="pageHeader">
        <h1>PLY ellipsoids (instanced)</h1>
        <p className="muted">
          Renders ellipsoids from PLY using three separate storage buffers:
          center, covariance, rgba.
        </p>
        <div className="muted">
          File: <code>app/(pages)/ply-ellipsoids/page.tsx</code>
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
          <SplatScene data={data} controlsGroup="PLY ellipsoids" />
        </WebGPUCanvasFrame>
      ) : null}
    </div>
  );
}
