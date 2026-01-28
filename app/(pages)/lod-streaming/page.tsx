"use client";

import { WebGPUCanvasFrame } from "@/app/_shared/webgpu/WebGPUCanvasFrame";
import { LodStreamingScene } from "@/app/_shared/scenes/LodStreamingScene";
import { useLodMeta } from "@/app/_shared/hooks/useLodMeta";

const LOD_META_URL = "/lod/lod-meta.json";
const MAX_CAPACITY = 600_000;

export default function LodStreamingPage() {
  const state = useLodMeta(LOD_META_URL);

  return (
    <div className="page">
      <div className="pageHeader">
        <h1>LOD streaming (chunked)</h1>
        <p className="muted">
          Loads SOG chunks on demand based on camera frustum and appends them to
          GPU buffers. This demo never unloads.
        </p>
        <div className="muted">
          Meta: <code>{LOD_META_URL}</code>
        </div>
        <div className="muted">
          Status: <code>{state.status}</code>
          {state.error ? <span> ({state.error})</span> : null}
        </div>
      </div>

      {state.status === "ready" ? (
        <WebGPUCanvasFrame
          camera={{ position: [4, 3, 4], fov: 50, near: 0.1, far: 200 }}
          gl={{ antialias: false }}
        >
          <LodStreamingScene
            meta={state.meta}
            metaJson={state.metaJson}
            metaUrl={LOD_META_URL}
            capacity={MAX_CAPACITY}
          />
        </WebGPUCanvasFrame>
      ) : null}
    </div>
  );
}
