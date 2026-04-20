import { useEffect, useState } from "react";
import { parseSplatPly } from "../loaders/ply";

export type PlyPacked = {
  count: number;
  center: Float32Array; // 3N
  covariance: Float32Array; // 6N (two vec3 per splat)
  rgba: Uint32Array; // N packed RGBA8
  shCoeffsL1?: Float32Array;
  shCoeffsL2Packed?: Uint32Array;
  shCoeffsL2Scale?: number;
  shCoeffsL3Packed?: Uint32Array;
  shCoeffsL3Scale?: number;
  shDegree?: number;
};

export function usePlyPacked(url: string): {
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

        setStatus("parsing PLY…");
        const splat = parseSplatPly(bytes);

        const packed: PlyPacked = {
          count: splat.count,
          center: splat.center,
          covariance: splat.covariance,
          rgba: splat.rgba,
          shCoeffsL1: splat.shCoeffsL1,
          shCoeffsL2Packed: splat.shCoeffsL2Packed,
          shCoeffsL2Scale: splat.shCoeffsL2Scale,
          shCoeffsL3Packed: splat.shCoeffsL3Packed,
          shCoeffsL3Scale: splat.shCoeffsL3Scale,
          shDegree: splat.shDegree,
        };

        console.log("[PLY buffers]", {
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
