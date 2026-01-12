"use client";

import { usePlyPackedRust } from "@/app/_shared/hooks/usePlyPackedRust";
import { SplatScene } from "@/app/_shared/scenes/SplatScene";
import { WebGPUCanvasFrame } from "@/app/_shared/webgpu/WebGPUCanvasFrame";

const PLY_URL = "/ref_splats_binary.ply";

export default function RefSplatsPage() {
  const { status, data } = usePlyPackedRust(PLY_URL);

  return (
    <div className="page">
      <div className="pageHeader">
        <h1>Ref splats (PLY)</h1>
        <p className="muted">
          Loads <code>{PLY_URL}</code> and renders it via the same instanced
          ellipsoid pipeline.
        </p>
        <div className="muted">
          File: <code>app/(pages)/ref-splats/page.tsx</code>
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
          <SplatScene
            data={data}
            controlsGroup="Ref splats"
            ellipsoidSphereGeometryArgs={[1, 24, 24]}
          />
        </WebGPUCanvasFrame>
      ) : null}
    </div>
  );
}
