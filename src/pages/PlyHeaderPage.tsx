import { useEffect, useRef, useState } from "react";
import { parseHeader, parseSplatPly } from "../loaders/ply";

const PLY_URL = "/cactus_splat3_30kSteps_142k_splats.ply";

export function PlyHeaderPage() {
  const [status, setStatus] = useState("idle");
  const ranOnceRef = useRef(false);

  useEffect(() => {
    // React 18 StrictMode может запускать effect дважды в dev — не хотим дважды качать большой файл.
    if (ranOnceRef.current) return;
    ranOnceRef.current = true;

    const ac = new AbortController();
    (async () => {
      try {
        setStatus(`fetching: ${PLY_URL}`);
        const res = await fetch(PLY_URL, { signal: ac.signal });
        if (!res.ok)
          throw new Error(`fetch failed: ${res.status} ${res.statusText}`);

        setStatus("reading arrayBuffer…");
        const buf = await res.arrayBuffer();

        setStatus("parsing header… (see console)");
        const bytes = new Uint8Array(buf);
        const { header, dataOffset, newline } = parseHeader(bytes);

        // Требование: пока просто распарсить заголовок и вывести в консоль.
        console.log("[PLY header]", { header, dataOffset, newline });

        setStatus("parsing splats… (see console)");
        const t0 = performance.now();
        const splat = parseSplatPly(bytes);
        const t1 = performance.now();

        // Требование: вывести в консоль результат парсинга файла.
        console.log("[PLY splat parse]", splat);
        const firstRGBA8: number[] = [];
        for (let i = 0; i < Math.min(2, splat.rgba.length); i++) {
          const u = splat.rgba[i] >>> 0;
          firstRGBA8.push(
            u & 255,
            (u >>> 8) & 255,
            (u >>> 16) & 255,
            (u >>> 24) & 255
          );
        }
        console.log("[PLY splat parse summary]", {
          count: splat.count,
          format: splat.format,
          centerLen: splat.center.length,
          covarianceLen: splat.covariance.length,
          rgbaLen: splat.rgba.length,
          ms: Math.round((t1 - t0) * 100) / 100,
          firstCenter: Array.from(splat.center.slice(0, 6)),
          firstCovariance: Array.from(splat.covariance.slice(0, 12)),
          firstRGBA8,
        });

        setStatus("done (header + parse logged to console)");
      } catch (e) {
        if ((e as Error)?.name === "AbortError") return;
        console.error(e);
        setStatus(`error: ${(e as Error)?.message ?? String(e)}`);
      }
    })();

    return () => ac.abort();
  }, []);

  return (
    <div style={{ padding: 16 }}>
      <h2>PLY header demo</h2>
      <div>
        File: <code>{PLY_URL}</code>
      </div>
      <div style={{ marginTop: 8 }}>
        Status: <code>{status}</code>
      </div>
      <div style={{ marginTop: 8, opacity: 0.8 }}>
        Open DevTools console to see the parsed header object.
      </div>
    </div>
  );
}
