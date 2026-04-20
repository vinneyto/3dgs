"use client";

import { usePlyPackedRust } from "@/app/_shared/hooks/usePlyPackedRust";
import { NewSplatScene } from "@/app/_shared/scenes/NewSplatScene";
import { WebGPUCanvasFrame } from "@/app/_shared/webgpu/WebGPUCanvasFrame";

const PLY_URL = "https://3dgs-vinneyto.s3.us-east-1.amazonaws.com/truck_high.ply";

export default function TruckHighPlyPage() {
  const { status, data } = usePlyPackedRust(PLY_URL);

  return (
    <div className="page">
      <div className="pageHeader">
        <h1>Truck PLY (Rust WASM)</h1>
        <p className="muted">
          Loads <code>{PLY_URL}</code> from <code>public/</code>, parses it via
          Rust→WASM, then renders splats using storage buffers + depth sorting.
        </p>
        <div className="muted">
          File: <code>app/(pages)/truck-high-ply/page.tsx</code>
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
          <NewSplatScene
            data={data}
            controlsGroup="Truck PLY"
            meshScale={[1, -1, -1]}
          />
        </WebGPUCanvasFrame>
      ) : null}
    </div>
  );
}

