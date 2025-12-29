import { usePlyPackedRust } from "../hooks/usePlyPackedRust";
import { SplatScene } from "../scenes/SplatScene";
import { WebGPUCanvasFrame } from "../webgpu/WebGPUCanvasFrame";

const PLY_URL = "/ref_splats_binary.ply";

export function RefSplatsPage() {
  const { status, data } = usePlyPackedRust(PLY_URL);

  return (
    <div className="page">
      <div className="pageHeader">
        <h1>Ref splats (PLY)</h1>
        <p className="muted">
          Loads <code>/public/ref_splats.ply</code> and renders it via the same
          instanced ellipsoid / gaussian-quad pipeline.
        </p>
        <div className="muted">
          Status: <code>{status}</code>
        </div>
      </div>

      {data ? (
        <WebGPUCanvasFrame
          camera={{ position: [4, 3, 4], fov: 50, near: 0.1, far: 100 }}
          gl={{ antialias: false }}
        >
          <SplatScene data={data} controlsGroup="Ref splats" />
        </WebGPUCanvasFrame>
      ) : null}
    </div>
  );
}
