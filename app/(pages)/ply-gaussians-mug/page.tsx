"use client";

import { usePlyPackedRust } from "@/app/_shared/hooks/usePlyPackedRust";
import { NewSplatScene } from "@/app/_shared/scenes/NewSplatScene";
import { WebGPUCanvasFrame } from "@/app/_shared/webgpu/WebGPUCanvasFrame";

const PLY_URL = "/mug_01_132cd1ab-cf78-4036-b53e-83770be40e69.ply";

export default function PlyGaussiansMugPage() {
  const { status, data } = usePlyPackedRust(PLY_URL);

  return (
    <div className="page">
      <div className="pageHeader">
        <h1>PLY gaussians (mug)</h1>
        <p className="muted">
          Duplicate of <code>/ply-gaussians</code>, but loading mug splats from
          <code> public{PLY_URL}</code>.
        </p>
        <div className="muted">
          File: <code>app/(pages)/ply-gaussians-mug/page.tsx</code>
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
          <NewSplatScene
            data={data}
            controlsGroup="PLY gaussians (mug)"
            meshScale={[1, 1, 1]}
          />
        </WebGPUCanvasFrame>
      ) : null}
    </div>
  );
}
