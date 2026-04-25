"use client";

import { useEffect, useRef, useState } from "react";
import { parseHeader, parseSplatPly } from "@/app/_shared/loaders/ply";

const PLY_URL = "/cactus_splat3_30kSteps_142k_splats.ply";

export default function PlyHeaderPage() {
  const [status, setStatus] = useState("idle");
  const ranOnceRef = useRef(false);

  useEffect(() => {
    // React StrictMode can run effect twice in dev — don't fetch twice for big file.
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

        console.log("[PLY header]", { header, dataOffset, newline });

        setStatus("parsing splats… (see console)");
        const t0 = performance.now();
        const splat = parseSplatPly(bytes);
        const t1 = performance.now();

        console.log("[PLY splat parse]", splat);
        const firstRGBA8: number[] = [];
        for (let i = 0; i < Math.min(2, splat.rgba.length); i++) {
          const u = splat.rgba[i] >>> 0;
          firstRGBA8.push(
            u & 255,
            (u >>> 8) & 255,
            (u >>> 16) & 255,
            (u >>> 24) & 255,
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
    <div className="page">
      <div className="pageHeader">
        <h1>PLY header demo</h1>
        <div className="muted">
          File: <code>app/(pages)/ply-header/page.tsx</code>
        </div>
        <div className="muted">
          PLY URL: <code>{PLY_URL}</code>
        </div>
        <div className="muted">
          Status: <code>{status}</code>
        </div>
        <div className="muted">Open DevTools console to see parsed output.</div>
      </div>
    </div>
  );
}
