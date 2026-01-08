import { useEffect, useState } from "react";
import { parseSplatPly } from "../loaders/ply";
import { useRustPlyParse } from "../hooks/useRustPlyParse";

type ParseState =
  | { kind: "idle" }
  | { kind: "loading" }
  | {
      kind: "ready";
      bytes: number;
      rustMs: number;
      tsMs: number;
      rust: {
        count: number;
        format: string;
        bboxMin: number[];
        bboxMax: number[];
      };
      ts: { count: number; format: string };
    }
  | { kind: "error"; error: string };

export function RustWasmPlyParsePage() {
  const [state, setState] = useState<ParseState>({ kind: "idle" });
  const rustParse = useRustPlyParse("/cactus_splat3_30kSteps_142k_splats.ply");
  const { kind: rustKind } = rustParse;

  const {
    out: rustOut,
    bytes: plyBytes,
    rustMs: rustParseMs,
  } = rustParse.kind === "ready"
    ? rustParse
    : { out: null, bytes: null, rustMs: null };

  const { error: rustError } =
    rustParse.kind === "error" ? rustParse : { error: null };

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        if (rustKind === "idle") {
          setState({ kind: "idle" });
          return;
        }
        if (rustKind === "loading") {
          setState({ kind: "loading" });
          return;
        }
        if (rustKind === "error") {
          setState({ kind: "error", error: rustError ?? "Unknown error" });
          return;
        }

        if (!rustOut || !plyBytes || rustParseMs == null) {
          throw new Error("Rust parse is not ready");
        }

        const bytes = plyBytes;
        const out = rustOut;
        const bboxMin = Array.from(out.bboxMin);
        const bboxMax = Array.from(out.bboxMax);

        const t2 = performance.now();
        const ts = parseSplatPly(bytes);
        const t3 = performance.now();

        if (!cancelled) {
          setState({
            kind: "ready",
            bytes: bytes.byteLength,
            rustMs: rustParseMs,
            tsMs: t3 - t2,
            rust: { count: out.count, format: out.format, bboxMin, bboxMax },
            ts: { count: ts.count, format: ts.format },
          });
        }
      } catch (e) {
        const error = e instanceof Error ? e.message : String(e);
        if (!cancelled) setState({ kind: "error", error });
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [rustKind, rustError, rustOut, plyBytes, rustParseMs]);

  return (
    <div style={{ padding: 16 }}>
      <h2>PLY parsing via Rust WASM</h2>
      <div style={{ opacity: 0.8 }}>
        Demo file: <code>src/pages/RustWasmPlyParsePage.tsx</code>
      </div>
      <p>
        This page parses a 3D Gaussian Splats PLY via Rust→WASM and also runs
        the existing TS parser for comparison (the TS parser is kept as-is).
      </p>

      {state.kind === "idle" && <div>Idle…</div>}
      {state.kind === "loading" && <div>Loading PLY + WASM…</div>}
      {state.kind === "error" && (
        <div style={{ color: "crimson" }}>
          <div>Error:</div>
          <pre style={{ whiteSpace: "pre-wrap" }}>{state.error}</pre>
        </div>
      )}
      {state.kind === "ready" && (
        <div style={{ display: "grid", gap: 10, maxWidth: 760 }}>
          <div>
            <b>File</b>: {Math.round(state.bytes / 1024 / 1024)} MB
          </div>
          <div>
            <b>Rust WASM</b>: {state.rustMs.toFixed(2)} ms, count=
            {state.rust.count}, format={state.rust.format}
          </div>
          <div>
            <b>TS</b>: {state.tsMs.toFixed(2)} ms, count={state.ts.count},
            format=
            {state.ts.format}
          </div>
          <div>
            <b>BBox (Rust)</b>: min=[
            {state.rust.bboxMin.map((v) => v.toFixed(3)).join(", ")}], max=[
            {state.rust.bboxMax.map((v) => v.toFixed(3)).join(", ")}]
          </div>
          <div style={{ opacity: 0.8 }}>
            Note: WASM buffers are returned as views over WASM memory (no
            copies).
          </div>
        </div>
      )}
    </div>
  );
}
