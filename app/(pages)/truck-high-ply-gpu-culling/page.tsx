"use client";

import { usePlyPackedRust } from "@/app/_shared/hooks/usePlyPackedRust";
import { IndirectCulledSplatScene } from "@/app/_shared/scenes/IndirectCulledSplatScene";
import { WebGPUCanvasFrame } from "@/app/_shared/webgpu/WebGPUCanvasFrame";

const PLY_URL = "https://3dgs-vinneyto.s3.us-east-1.amazonaws.com/truck_high.ply";

export default function TruckHighPlyGpuCullingPage() {
  const { status, data } = usePlyPackedRust(PLY_URL);

  return (
    <div className="page">
      <div className="pageHeader">
        <h1>Truck PLY (GPU culling)</h1>
        <p className="muted">
          Loads <code>{PLY_URL}</code> from <code>public/</code>, parses it via
          Rust→WASM, then renders splats with GPU frustum culling + prefix-sum
          compaction + (optional) depth sorting, using{" "}
          <code>drawIndexedIndirect</code>.
        </p>
        <div className="muted">
          File: <code>app/(pages)/truck-high-ply-gpu-culling/page.tsx</code>
        </div>
        <div className="muted">
          Status: <code>{status}</code>
        </div>
      </div>

      {data ? (
        <WebGPUCanvasFrame
          camera={{ position: [4, 3, 4], fov: 50, near: 0.1, far: 500 }}
          gl={{ antialias: false }}
        >
          <IndirectCulledSplatScene
            data={data}
            controlsGroup="Truck PLY GPU culling"
            meshScale={[1, -1, -1]}
          />
        </WebGPUCanvasFrame>
      ) : null}
    </div>
  );
}

