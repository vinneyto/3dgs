import { useEffect, useState } from "react";
import { getRustWasm } from "../lib/rustWasm";

export type RustSogsV2ParseState =
  | { kind: "idle" }
  | { kind: "loading" }
  | {
      kind: "ready";
      bytes: Uint8Array;
      rustMs: number;
      out: import("../wasm/pkg/rust_wasm").SplatsBuffers;
    }
  | { kind: "error"; error: string };

/**
 * Fetches a SOGS v2 file (typically a `.sog` zip) and parses it in Rust (WASM).
 * Returns both the raw bytes and the decoded splats buffers.
 */
export function useRustSogsV2Parse(url: string): RustSogsV2ParseState {
  const [state, setState] = useState<RustSogsV2ParseState>({ kind: "idle" });

  useEffect(() => {
    let cancelled = false;
    let outToFree: { free?: () => void } | null = null;

    (async () => {
      if (!url) {
        setState({ kind: "idle" });
        return;
      }
      setState({ kind: "loading" });
      try {
        const res = await fetch(url);
        if (!res.ok)
          throw new Error(`fetch failed: ${res.status} ${res.statusText}`);

        const ab = await res.arrayBuffer();
        const bytes = new Uint8Array(ab);

        const mod = await getRustWasm();

        const t0 = performance.now();
        const out = mod.parse_splat_sogs_v2(bytes);
        const t1 = performance.now();

        outToFree = out;

        console.log("[rust] parse_splat_sogs_v2() summary", {
          url,
          bytes: bytes.byteLength,
          count: out.count,
          format: out.format,
          bboxMin: Array.from(out.bboxMin),
          bboxMax: Array.from(out.bboxMax),
          centerLen: out.center.length,
          covarianceLen: out.covariance.length,
          rgbaLen: out.rgba.length,
          shDegree: out.shDegree,
        });

        if (!cancelled) {
          setState({ kind: "ready", bytes, rustMs: t1 - t0, out });
        }
      } catch (e) {
        const error = e instanceof Error ? e.message : String(e);
        if (!cancelled) setState({ kind: "error", error });
      }
    })();

    return () => {
      cancelled = true;
      outToFree?.free?.();
    };
  }, [url]);

  return state;
}

