import { usePlyPackedRust } from "../hooks/usePlyPackedRust";
import { NewSplatScene } from "../scenes/NewSplatScene";
import { WebGPUCanvasFrame } from "../webgpu/WebGPUCanvasFrame";

const PLY_URL = "/photo.ply";

export function PhotoPlyGaussiansPage() {
  const { status, data } = usePlyPackedRust(PLY_URL);

  return (
    <div className="page">
      <div className="pageHeader">
        <h1>Photo PLY gaussians (WIP)</h1>
        <p className="muted">
          Setup-only demo: loads <code>{PLY_URL}</code> from{" "}
          <code>public/</code> into storage buffers and computes depth keys +
          radix-sorted indices.
        </p>
        <div className="muted">
          File: <code>src/pages/PhotoPlyGaussiansPage.tsx</code>
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
          <NewSplatScene data={data} controlsGroup="Photo PLY gaussians" />
        </WebGPUCanvasFrame>
      ) : null}
    </div>
  );
}
