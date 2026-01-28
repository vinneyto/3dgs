"use client";

import { useSogPackedRust } from "@/app/_shared/hooks/useSogPackedRust";
import { NewSplatScene } from "@/app/_shared/scenes/NewSplatScene";
import { WebGPUCanvasFrame } from "@/app/_shared/webgpu/WebGPUCanvasFrame";

const SOG_URL = "/room.sog";

export default function RoomSogPage() {
  const { status, data, info } = useSogPackedRust(SOG_URL);

  return (
    <div className="page">
      <div className="pageHeader">
        <h1>Room SOG (Rust WASM)</h1>
        <p className="muted">
          Loads <code>{SOG_URL}</code> from <code>public/</code>, parses it via
          Rust→WASM, then renders splats using storage buffers + depth sorting.
        </p>
        <div className="muted">
          File: <code>app/(pages)/room-sog/page.tsx</code>
        </div>
        <div className="muted">
          Status: <code>{status}</code>
        </div>
        {info ? (
          <div className="muted">
            Parsed:{" "}
            <code>
              {info.bytesMb.toFixed(2)} MB, {info.rustMs.toFixed(2)} ms,
              shDegree=
              {info.shDegree}, format={info.format}
            </code>
          </div>
        ) : null}
      </div>

      {data ? (
        <WebGPUCanvasFrame
          camera={{ position: [4, 3, 4], fov: 50, near: 0.1, far: 500 }}
          gl={{ antialias: false }}
        >
          <NewSplatScene
            data={data}
            controlsGroup="Room SOG"
            meshScale={[1, -1, -1]}
          />
        </WebGPUCanvasFrame>
      ) : null}
    </div>
  );
}
