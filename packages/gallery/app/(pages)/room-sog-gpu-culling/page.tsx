"use client";

import { useSogPackedRust } from "@/app/_shared/hooks/useSogPackedRust";
import { IndirectCulledSplatScene } from "@/app/_shared/scenes/IndirectCulledSplatScene";
import { WebGPUCanvasFrame } from "@/app/_shared/webgpu/WebGPUCanvasFrame";

const SOG_URL = "/room.sog";

export default function RoomSogGpuCullingPage() {
  const { status, data, info } = useSogPackedRust(SOG_URL);

  return (
    <div className="page">
      <div className="pageHeader">
        <h1>Room SOG (GPU culling)</h1>
        <p className="muted">
          Loads <code>{SOG_URL}</code> from <code>public/</code>, parses it via
          Rust→WASM, then renders splats with GPU frustum culling + prefix-sum
          compaction + (optional) depth sorting, using{" "}
          <code>drawIndexedIndirect</code>.
        </p>
        <div className="muted">
          File: <code>app/(pages)/room-sog-gpu-culling/page.tsx</code>
        </div>
        <div className="muted">
          Status: <code>{status}</code>
        </div>
        {info ? (
          <div className="muted">
            Parsed:{" "}
            <code>
              {info.bytesMb.toFixed(2)} MB, {info.rustMs.toFixed(2)} ms,
              shDegree={info.shDegree}, format={info.format}
            </code>
          </div>
        ) : null}
      </div>

      {data ? (
        <WebGPUCanvasFrame
          camera={{ position: [4, 3, 4], fov: 50, near: 0.1, far: 500 }}
          gl={{ antialias: false }}
        >
          <IndirectCulledSplatScene
            data={data}
            controlsGroup="Room SOG GPU culling"
            meshScale={[1, -1, -1]}
          />
        </WebGPUCanvasFrame>
      ) : null}
    </div>
  );
}
