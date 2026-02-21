"use client";

import { useMemo, useState } from "react";
import { flushSync } from "react-dom";
import { formatBytes } from "@/app/_shared/utils/formatBytes";
import {
  RADIX_SORT_WEBGPU_DEMO_N,
  RADIX_SORT_WEBGPU_DEMO_WORKGROUP_SIZE_X,
  type RadixSortWebgpuDemoResult,
  prepareRadixSortInput,
  compareCpuArraySortToGpu,
  runRadixSortCpuArraySort,
  runRadixSortWebgpuGpu,
} from "./runRadixSortWebgpuSetup";

const BYTES_PER_U32 = 4;

export default function RadixSortWebGPUDemoPage() {
  const [busy, setBusy] = useState(false);
  const [phase, setPhase] = useState<string | null>(null);
  const [result, setResult] = useState<RadixSortWebgpuDemoResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const bytes = useMemo(() => RADIX_SORT_WEBGPU_DEMO_N * BYTES_PER_U32, []);

  return (
    <div style={{ padding: 16, maxWidth: 980 }}>
      <h2 style={{ margin: "0 0 8px" }}>WebGPU radix-sort demo</h2>
      <p style={{ margin: "0 0 12px", opacity: 0.85 }}>
        Runs a 16-pass (2-bit) radix sort on GPU and reports performance + verification.
      </p>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "auto 1fr",
          gap: 8,
          alignItems: "center",
          marginBottom: 12,
        }}
      >
        <div style={{ opacity: 0.7 }}>N</div>
        <div>
          {RADIX_SORT_WEBGPU_DEMO_N.toLocaleString()}{" "}
          <span style={{ opacity: 0.7 }}>(uint32)</span>
        </div>
        <div style={{ opacity: 0.7 }}>Buffer size</div>
        <div>{formatBytes(bytes)}</div>
        <div style={{ opacity: 0.7 }}>Workgroup</div>
        <div>
          {RADIX_SORT_WEBGPU_DEMO_WORKGROUP_SIZE_X}×1×1{" "}
          <span style={{ opacity: 0.7 }}>(2D dispatch uses x+y)</span>
        </div>
      </div>

      <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
        <button
          disabled={busy}
          onClick={async () => {
            setBusy(true);
            setError(null);
            try {
              flushSync(() => setPhase("Generating input data (CPU)..."));
              const prepared = prepareRadixSortInput({
                n: RADIX_SORT_WEBGPU_DEMO_N,
                seed: 123456789,
              });

              flushSync(() => setPhase("Running GPU radix sort..."));
              const { result: gpuResult, gpuSorted } =
                await runRadixSortWebgpuGpu({ prepared });
              flushSync(() => setResult(gpuResult));

              // Let the browser actually paint GPU results before heavy CPU sort.
              await new Promise<void>((resolve) =>
                window.requestAnimationFrame(() => resolve()),
              );
              await new Promise<void>((resolve) =>
                window.requestAnimationFrame(() => resolve()),
              );

              // Ensure phase updates before the CPU blocks the main thread.
              flushSync(() => setPhase("Running CPU Array.sort..."));
              await new Promise<void>((resolve) =>
                window.requestAnimationFrame(() => resolve()),
              );
              const cpuSort = runRadixSortCpuArraySort({
                cpuValues: prepared.cpuValues,
                memorySizeGb: gpuResult.perf.memorySizeGb,
              });
              setResult((prev) =>
                prev
                  ? {
                      ...prev,
                      cpu: cpuSort.cpu,
                      timingsMs: {
                        ...prev.timingsMs,
                        cpuSort: cpuSort.cpuSortMs,
                      },
                    }
                  : prev,
              );

              flushSync(() => setPhase("Comparing CPU vs GPU..."));
              await new Promise<void>((resolve) =>
                window.requestAnimationFrame(() => resolve()),
              );
              const cpuCmp = compareCpuArraySortToGpu({
                ref: cpuSort.ref,
                gpuSorted,
              });
              setResult((prev) =>
                prev
                  ? {
                      ...prev,
                      timingsMs: {
                        ...prev.timingsMs,
                        cpuCompare: cpuCmp.cpuCompareMs,
                      },
                      verify: { ...prev.verify, cpuCompare: cpuCmp.cpuCompare },
                    }
                  : prev,
              );
            } catch (e) {
              setResult(null);
              setError(e instanceof Error ? e.message : String(e));
            } finally {
              setPhase(null);
              setBusy(false);
            }
          }}
          style={{
            padding: "8px 12px",
            borderRadius: 8,
            border: "1px solid rgba(255,255,255,0.15)",
            background: busy ? "rgba(255,255,255,0.06)" : "rgba(255,255,255,0.10)",
            cursor: busy ? "not-allowed" : "pointer",
          }}
        >
          Run radix sort
        </button>
        {busy && phase ? <div style={{ opacity: 0.75 }}>{phase}</div> : null}
      </div>

      {error ? (
        <div
          style={{
            whiteSpace: "pre-wrap",
            padding: 12,
            borderRadius: 8,
            border: "1px solid rgba(255,80,80,0.35)",
            background: "rgba(255,80,80,0.08)",
            marginBottom: 12,
          }}
        >
          {error}
        </div>
      ) : null}

      {result ? (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "auto 1fr",
            gap: 8,
            alignItems: "start",
          }}
        >
          <div style={{ opacity: 0.7 }}>Adapter</div>
          <div>{result.adapter}</div>

          <div style={{ opacity: 0.7 }}>Limits</div>
          <div>
            maxBufferSize={formatBytes(result.limits.maxBufferSize)},{" "}
            maxStorageBufferBindingSize=
            {formatBytes(result.limits.maxStorageBufferBindingSize)},{" "}
            maxComputeWorkgroupsPerDimension=
            {result.limits.maxComputeWorkgroupsPerDimension.toLocaleString()}
          </div>

          <div style={{ opacity: 0.7 }}>Perf</div>
          <div>
            GPU: {result.perf.effectiveBandwidthGbps.toFixed(2)} GB/s,{" "}
            {result.perf.uintMillionsPerSecond.toFixed(2)} M u32/s,{" "}
            {result.perf.radixSortSeconds.toFixed(3)} s
            {result.cpu.method ? (
              <>
                <br />
                CPU ({result.cpu.method}):{" "}
                {(result.cpu.effectiveBandwidthGbps ?? 0).toFixed(2)} GB/s,{" "}
                {(result.cpu.uintMillionsPerSecond ?? 0).toFixed(2)} M u32/s,{" "}
                {(result.cpu.seconds ?? 0).toFixed(3)} s
              </>
            ) : (
              <>
                <br />
                CPU: waiting…
              </>
            )}
          </div>

          <div style={{ opacity: 0.7 }}>Verify</div>
          <div>
            sortedOk={String(result.verify.sortedOk)}
            {result.verify.sortedOk
              ? ""
              : ` (firstBadIndex=${result.verify.firstBadIndex})`}
            {"; "}
            cpuCompare=
            {result.verify.cpuCompare
              ? String(result.verify.cpuCompare.ok)
              : "(skipped)"}
          </div>

          <div style={{ opacity: 0.7 }}>Details</div>
          <pre
            style={{
              margin: 0,
              padding: 12,
              borderRadius: 8,
              border: "1px solid rgba(255,255,255,0.12)",
              background: "rgba(255,255,255,0.06)",
              overflow: "auto",
              maxHeight: 360,
            }}
          >
            {JSON.stringify(result, null, 2)}
          </pre>
        </div>
      ) : null}
    </div>
  );
}

