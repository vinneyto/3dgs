"use client";

import { useMemo, useState } from "react";
import { useRustSogsV2Parse } from "@/app/_shared/hooks/useRustSogsV2Parse";

export default function RustWasmSogsV2ParsePage() {
  const [url, setUrl] = useState<string>("");
  const [submittedUrl, setSubmittedUrl] = useState<string>("");

  const state = useRustSogsV2Parse(submittedUrl);

  const summary = useMemo(() => {
    if (state.kind !== "ready") return null;
    const out = state.out;
    return {
      bytesMb: state.bytes.byteLength / 1024 / 1024,
      rustMs: state.rustMs,
      count: out.count,
      format: out.format,
      bboxMin: Array.from(out.bboxMin),
      bboxMax: Array.from(out.bboxMax),
      shDegree: out.shDegree,
    };
  }, [state]);

  return (
    <div className="page">
      <div className="pageHeader">
        <h1>SOGS v2 parsing via Rust WASM</h1>
        <p className="muted">
          This page parses a SOGS v2 file (typically a <code>.sog</code> ZIP
          containing <code>meta.json</code> + WebP images) via Rust→WASM.
        </p>
        <div className="muted">
          File: <code>app/(pages)/rust-sogs-v2/page.tsx</code>
        </div>
      </div>

      <div style={{ display: "grid", gap: 10, maxWidth: 760 }}>
        <div style={{ display: "grid", gap: 6 }}>
          <label>
            URL (served by Next, e.g. <code>/my_scene.sog</code>)
          </label>
          <div style={{ display: "flex", gap: 8 }}>
            <input
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="/scene.sog"
              style={{ flex: 1, padding: "8px 10px" }}
            />
            <button
              onClick={() => setSubmittedUrl(url)}
              style={{ padding: "8px 12px" }}
            >
              Parse
            </button>
          </div>
          <div className="muted">
            Tip: drop a <code>.sog</code> into <code>public/</code> and use{" "}
            <code>/filename.sog</code>.
          </div>
        </div>

        {state.kind === "idle" && <div className="muted">Idle…</div>}
        {state.kind === "loading" && <div>Loading SOGS v2 + WASM…</div>}
        {state.kind === "error" && (
          <div style={{ color: "crimson" }}>
            <div>Error:</div>
            <pre style={{ whiteSpace: "pre-wrap" }}>{state.error}</pre>
          </div>
        )}
        {summary && (
          <div style={{ display: "grid", gap: 10 }}>
            <div>
              <b>File</b>: {summary.bytesMb.toFixed(2)} MB
            </div>
            <div>
              <b>Rust WASM</b>: {summary.rustMs.toFixed(2)} ms, count=
              {summary.count}, format={summary.format}, shDegree={summary.shDegree}
            </div>
            <div>
              <b>BBox</b>: min=[
              {summary.bboxMin.map((v) => v.toFixed(3)).join(", ")}], max=[
              {summary.bboxMax.map((v) => v.toFixed(3)).join(", ")}]
            </div>
            <div className="muted">
              Note: WASM buffers are returned as views over WASM memory (no
              copies).
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

