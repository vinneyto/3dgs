"use client";

import { useMemo } from "react";
import { useRustSogsV2Parse } from "@/app/_shared/hooks/useRustSogsV2Parse";

const SOG_URL = "/room.sog";

export default function RustWasmSogsV2ParsePage() {
  const state = useRustSogsV2Parse(SOG_URL);

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
        <div className="muted">
          Reading: <code>{SOG_URL}</code>
        </div>

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
              {summary.count}, format={summary.format}, shDegree=
              {summary.shDegree}
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

