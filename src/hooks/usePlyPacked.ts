import { useEffect, useState } from "react";
import { parseSplatPly } from "../loaders/ply";

export type PlyPacked = {
  count: number;
  center: Float32Array; // 3N
  covariance: Float32Array; // 6N (two vec3 per splat)
  rgba: Uint32Array; // N packed RGBA8
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
        if (!res.ok) throw new Error(`fetch failed: ${res.status} ${res.statusText}`);

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
        };

        console.log("[PLY buffers]", {
          count: packed.count,
          centerLen: packed.center.length,
          covarianceLen: packed.covariance.length,
          rgbaLen: packed.rgba.length,
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


