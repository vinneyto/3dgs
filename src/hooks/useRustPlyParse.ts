import { useEffect, useState } from "react";
import { getRustWasm } from "../lib/rustWasm";

export type RustPlyParseState =
  | { kind: "idle" }
  | { kind: "loading" }
  | {
      kind: "ready";
      bytes: Uint8Array;
      rustMs: number;
      out: import("../wasm/pkg/rust_wasm").SplatPlyBuffers;
    }
  | { kind: "error"; error: string };

/**
 * Fetches a PLY file and parses it in Rust (compiled to WebAssembly).
 * Also returns the raw bytes (useful for TS-side comparison).
 */
export function useRustPlyParse(url: string): RustPlyParseState {
  const [state, setState] = useState<RustPlyParseState>({ kind: "idle" });

  useEffect(() => {
    let cancelled = false;
    let outToFree: { free?: () => void } | null = null;

    (async () => {
      setState({ kind: "loading" });
      try {
        const res = await fetch(url);
        if (!res.ok)
          throw new Error(`fetch failed: ${res.status} ${res.statusText}`);

        const ab = await res.arrayBuffer();
        const bytes = new Uint8Array(ab);

        const mod = await getRustWasm();

        const t0 = performance.now();
        const out = mod.parse_splat_ply(bytes);
        const t1 = performance.now();

        outToFree = out;

        console.log("[rust] parse_splat_ply() summary", {
          url,
          bytes: bytes.byteLength,
          count: out.count,
          format: out.format,
          bboxMin: Array.from(out.bboxMin),
          bboxMax: Array.from(out.bboxMax),
          centerLen: out.center.length,
          covarianceLen: out.covariance.length,
          rgbaLen: out.rgba.length,
        });
        console.log("[rust] buffers", {
          center: out.center,
          covariance: out.covariance,
          rgba: out.rgba,
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
