import { useEffect, useState } from "react";
import { getRustWasm } from "../lib/rustWasm";
import type { PlyPacked } from "./usePlyPacked";

export function usePlyPackedRust(url: string): {
  status: string;
  data: PlyPacked | null;
} {
  const [status, setStatus] = useState("idle");
  const [data, setData] = useState<PlyPacked | null>(null);

  useEffect(() => {
    const ac = new AbortController();
    (async () => {
      try {
        setStatus(`fetching: ${url}`);
        const res = await fetch(url, { signal: ac.signal });
        if (!res.ok)
          throw new Error(`fetch failed: ${res.status} ${res.statusText}`);

        setStatus("reading arrayBuffer…");
        const buf = await res.arrayBuffer();
        const bytes = new Uint8Array(buf);

        setStatus("parsing PLY (Rust)…");
        const mod = await getRustWasm();
        const out = mod.parse_splat_ply(bytes);

        // Copy out of WASM memory to normal JS-owned TypedArrays so we can free the WASM object.
        const packed: PlyPacked = {
          count: out.count,
          center: new Float32Array(out.center),
          covariance: new Float32Array(out.covariance),
          rgba: new Uint32Array(out.rgba),
          shCoeffsL1: new Float32Array(out.shCoeffsL1),
          shCoeffsL2Packed: new Uint32Array(out.shCoeffsL2Packed),
          shCoeffsL2Scale: out.shCoeffsL2Scale,
          shCoeffsL3Packed: new Uint32Array(out.shCoeffsL3Packed),
          shCoeffsL3Scale: out.shCoeffsL3Scale,
          shDegree: out.shDegree,
        };

        out.free?.();

        console.log("[PLY buffers][rust]", {
          count: packed.count,
          centerLen: packed.center.length,
          covarianceLen: packed.covariance.length,
          rgbaLen: packed.rgba.length,
          shCoeffsL1Len: packed.shCoeffsL1?.length ?? 0,
          shCoeffsL2Len: packed.shCoeffsL2Packed?.length ?? 0,
          shCoeffsL3Len: packed.shCoeffsL3Packed?.length ?? 0,
          shDegree: packed.shDegree ?? 0,
        });

        setData(packed);
        setStatus("ready");
      } catch (e) {
        if ((e as Error)?.name === "AbortError") return;
        console.error(e);
        setStatus(`error: ${(e as Error)?.message ?? String(e)}`);
      }
    })();

    return () => ac.abort();
  }, [url]);

  return { status, data };
}
