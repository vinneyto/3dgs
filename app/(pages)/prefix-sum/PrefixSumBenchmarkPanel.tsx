"use client";

import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { MathText } from "@/app/_shared/components/MathText";
import { PrefixSumBenchmark } from "./prefixSumBenchmark";

function fmtMs(x: number | null) {
  if (x == null) return "—";
  return `${x.toFixed(2)} ms`;
}

export function PrefixSumBenchmarkPanel() {
  const bench = useMemo(() => new PrefixSumBenchmark(), []);

  const [draftN, setDraftN] = useState(50_000_000);
  const [draftIters, setDraftIters] = useState(10);
  const [draftWarmups, setDraftWarmups] = useState(2);

  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<Awaited<
    ReturnType<PrefixSumBenchmark["run"]>
  > | null>(null);

  useEffect(() => {
    return () => {
      bench.dispose();
    };
  }, [bench]);

  const n = result?.n ?? draftN;
  const iters = result?.iters ?? draftIters;

  const monoStyle = useMemo(
    () => ({
      margin: 0,
      whiteSpace: "pre-wrap" as const,
      fontFamily:
        'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
    }),
    [],
  );

  const renderLines = (lines: ReactNode[]) => (
    <div style={monoStyle}>
      {lines.filter(Boolean).map((line, i) => (
        <div key={i}>{line}</div>
      ))}
    </div>
  );

  return (
    <div className="page">
      <div className="pageHeader">
        <h1>Prefix sum benchmark (CPU + GPU)</h1>
        <p className="muted">
          Runs CPU exclusive scan first, then GPU scan via{" "}
          <code>GPUPrefixSum</code>
          on a self-created hidden <code>WebGPURenderer</code>.
        </p>
        <p className="muted" style={{ marginTop: 6, maxWidth: 980 }}>
          <MathText>{`GB/s here is an effective bandwidth estimate: `}</MathText>
          <MathText>{`we count one read of `}</MathText>
          <code>input</code>
          <MathText>{` and one write of `}</MathText>
          <code>output</code>
          <MathText>{` (\\(2 \\cdot n \\cdot 4\\) bytes for `}</MathText>
          <code>u32</code>
          <MathText>{`) and divide by median time.`}</MathText>
        </p>
      </div>

      <div
        style={{
          display: "flex",
          gap: 12,
          alignItems: "end",
          flexWrap: "wrap",
          marginBottom: 12,
          maxWidth: 980,
        }}
      >
        <label style={{ display: "grid", gap: 6 }}>
          <div className="muted">n</div>
          <input
            type="number"
            min={1}
            max={100_000_000}
            step={1}
            value={draftN}
            onChange={(e) =>
              setDraftN(
                Math.max(1, Math.min(100_000_000, e.target.valueAsNumber | 0)),
              )
            }
            style={{ width: 220 }}
            disabled={running}
          />
        </label>
        <label style={{ display: "grid", gap: 6 }}>
          <div className="muted">iters</div>
          <input
            type="number"
            min={1}
            max={50}
            step={1}
            value={draftIters}
            onChange={(e) =>
              setDraftIters(
                Math.max(1, Math.min(50, e.target.valueAsNumber | 0)),
              )
            }
            style={{ width: 120 }}
            disabled={running}
          />
        </label>
        <label style={{ display: "grid", gap: 6 }}>
          <div className="muted">warmups</div>
          <input
            type="number"
            min={0}
            max={10}
            step={1}
            value={draftWarmups}
            onChange={(e) =>
              setDraftWarmups(
                Math.max(0, Math.min(10, e.target.valueAsNumber | 0)),
              )
            }
            style={{ width: 120 }}
            disabled={running}
          />
        </label>
        <button
          type="button"
          className="button"
          disabled={running}
          onClick={async () => {
            setRunning(true);
            try {
              const r = await bench.run({
                n: draftN,
                iters: draftIters,
                warmups: draftWarmups,
              });
              setResult(r);
            } finally {
              setRunning(false);
            }
          }}
        >
          {running ? "Running…" : "Run"}
        </button>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 12,
          maxWidth: 980,
        }}
      >
        <div
          style={{
            padding: 12,
            borderRadius: 12,
            border: "1px solid rgba(255,255,255,0.08)",
            background: "rgba(255,255,255,0.02)",
          }}
        >
          <div style={{ marginBottom: 8, fontWeight: 600 }}>CPU (TS)</div>
          {renderLines([
            `n = ${n.toLocaleString()}`,
            `iters = ${iters}`,
            `input gen = ${fmtMs(result?.cpuGenMs ?? null)}`,
            `scan median = ${result ? fmtMs(result.cpuMedianMs) : "—"}`,
            <span key="cpu-bw">
              <span className="arrayLabel">
                effective bandwidth (read+write) ={" "}
              </span>
              <span className="metricBandwidth">
                {result ? `${result.cpuGbps.toFixed(2)} GB/s` : "—"}
              </span>
            </span>,
            `total sum = ${result ? result.cpuTotalSum.toLocaleString() : "—"}`,
            `check = ${result ? (result.cpuCheckOk ? "OK" : "FAILED") : "—"}`,
            result?.error ? `error = ${result.error}` : null,
          ])}
        </div>

        <div
          style={{
            padding: 12,
            borderRadius: 12,
            border: "1px solid rgba(255,255,255,0.08)",
            background: "rgba(255,255,255,0.02)",
          }}
        >
          <div style={{ marginBottom: 8, fontWeight: 600 }}>
            GPU (GPUPrefixSum)
          </div>
          {renderLines([
            `scan median = ${result ? fmtMs(result.gpuMedianMs) : "—"}`,
            <span key="gpu-bw">
              <span className="arrayLabel">
                effective bandwidth (read+write) ={" "}
              </span>
              <span className="metricBandwidth">
                {result ? `${result.gpuGbps.toFixed(2)} GB/s` : "—"}
              </span>
            </span>,
            <span key="gpu-speedup">
              <span className="arrayLabel">speedup vs CPU = </span>
              <span className="metricSpeedup">
                {result && result.cpuMedianMs > 0 && result.gpuMedianMs > 0
                  ? `${(result.cpuMedianMs / result.gpuMedianMs).toFixed(2)}x`
                  : "—"}
              </span>
            </span>,
            <span key="gpu-compare">
              <span className="arrayLabel">compare vs CPU = </span>
              <span
                className={
                  result ? (result.gpuCompareOk ? "metricOk" : "metricBad") : ""
                }
              >
                {result ? (result.gpuCompareOk ? "OK" : "FAILED") : "—"}
              </span>
            </span>,
            result?.gpuMismatch
              ? `mismatch at i=${result.gpuMismatch.i} cpu=${result.gpuMismatch.cpu} gpu=${result.gpuMismatch.gpu}`
              : null,
            result?.error ? `error = ${result.error}` : null,
          ])}
        </div>
      </div>

      <div
        style={{
          padding: 12,
          borderRadius: 12,
          border: "1px solid rgba(255,255,255,0.08)",
          background: "rgba(255,255,255,0.02)",
          maxWidth: 980,
          overflowX: "auto",
        }}
      >
        <div style={{ marginBottom: 8, fontWeight: 600 }}>Samples</div>
        <div style={{ whiteSpace: "nowrap" }}>
          {renderLines([
            result?.sampleInput?.length ? (
              <span key="sample-in">
                <span className="arrayLabel">
                  {`input[0..${result.sampleInput.length - 1}]=`}
                </span>
                <span className="arrayValueInput">
                  [{result.sampleInput.join(", ")}]
                </span>
              </span>
            ) : (
              "input[0..]=—"
            ),
            result?.samplePrefix?.length ? (
              <span key="sample-prefix">
                <span className="arrayLabel">
                  {`prefix[0..${result.samplePrefix.length - 1}]=`}
                </span>
                <span className="arrayValuePrefix">
                  [{result.samplePrefix.join(", ")}]
                </span>
              </span>
            ) : (
              "prefix[0..]=—"
            ),
          ])}
        </div>
      </div>
    </div>
  );
}
