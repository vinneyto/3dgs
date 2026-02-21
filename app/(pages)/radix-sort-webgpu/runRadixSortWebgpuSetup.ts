import { formatBytes } from "@/app/_shared/utils/formatBytes";
import { fillDeterministicU32FromIndex } from "@/app/_shared/utils/deterministicU32";
import { RADIX01_BLOCK_SIZE } from "@/app/_shared/webgpu/radixSort01LocalCounting";
import { createRadixSort01LocalCountingKernel } from "@/app/_shared/webgpu/radixSort01LocalCounting";
import {
  createRadixSort02GlobalPrefixesScanSumReductionKernel,
  RADIX02_SCAN_BLOCK_LEN,
} from "@/app/_shared/webgpu/radixSort02GlobalPrefixesScanSumReduction";
import { createRadixSort03GlobalPrefixesScanAccumulationKernel } from "@/app/_shared/webgpu/radixSort03GlobalPrefixesScanAccumulation";
import { createRadixSort04ScatterKernel } from "@/app/_shared/webgpu/radixSort04Scatter";

export const RADIX_SORT_WEBGPU_DEMO_N = 100_000_000;
export const RADIX_SORT_WEBGPU_DEMO_WORKGROUP_SIZE_X = 256;
const RADIX_SORT_WEBGPU_DEMO_SEED = 123456789;
const RADIX_SORT_WEBGPU_DEMO_SAMPLE_COUNT = 16;

export type RadixSortWebgpuDemoResult = {
  n: number;
  passes: number;
  bitsPerPass: number;
  adapter: string;
  limits: {
    maxBufferSize: number;
    maxStorageBufferBindingSize: number;
    maxComputeWorkgroupsPerDimension: number;
  };
  dispatchInit: { workgroupsTotal: number; x: number; y: number };
  timingsMs: {
    generateCPU: number;
    uploadGPU: number;
    radixSort: number;
    readback: number;
    cpuSort?: number;
    cpuCompare?: number;
  };
  perf: {
    memorySizeGb: number;
    radixSortSeconds: number;
    effectiveBandwidthGbps: number;
    uintMillionsPerSecond: number;
  };
  verify: {
    sortedOk: boolean;
    firstBadIndex: number;
    inputHash: { xor: number; sum: number; mix: number };
    outputHash: { xor: number; sum: number; mix: number };
    cpuCompare: { ok: boolean; firstBadIndex: number } | null;
  };
};

function limitsSummary(device: GPUDevice) {
  return {
    maxBufferSize: Number(device.limits.maxBufferSize),
    maxStorageBufferBindingSize: Number(
      device.limits.maxStorageBufferBindingSize,
    ),
    maxComputeWorkgroupsPerDimension: Number(
      device.limits.maxComputeWorkgroupsPerDimension,
    ),
    maxComputeInvocationsPerWorkgroup: Number(
      device.limits.maxComputeInvocationsPerWorkgroup,
    ),
    maxComputeWorkgroupSizeX: Number(device.limits.maxComputeWorkgroupSizeX),
    maxComputeWorkgroupSizeY: Number(device.limits.maxComputeWorkgroupSizeY),
    maxComputeWorkgroupSizeZ: Number(device.limits.maxComputeWorkgroupSizeZ),
  };
}

function ensureWebGPUAvailable(): void {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) {
    throw new Error("WebGPU is not available in this environment.");
  }
}

async function requestDeviceWithMaxLimits(): Promise<{
  adapter: GPUAdapter;
  device: GPUDevice;
}> {
  ensureWebGPUAvailable();

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (!adapter) {
    throw new Error("Failed to acquire WebGPU adapter.");
  }

  const requiredFeatures: GPUFeatureName[] = [];
  adapter.features.forEach((f) => requiredFeatures.push(f as GPUFeatureName));

  const requiredLimits = {
    // Ensure large storage buffers are possible (defaults can be too low).
    maxStorageBufferBindingSize: Number(
      adapter.limits.maxStorageBufferBindingSize,
    ),
    maxBufferSize: Number(adapter.limits.maxBufferSize),
  } satisfies Record<string, number>;

  const device = await adapter.requestDevice({
    requiredFeatures,
    // Some TS setups don't include full WebGPU limit typings; keep this permissive.
    requiredLimits:
      requiredLimits as unknown as GPUDeviceDescriptor["requiredLimits"],
  });

  return { adapter, device };
}

function computeDispatchShape({
  n,
  workgroupSizeX,
  maxComputeWorkgroupsPerDimension,
}: {
  n: number;
  workgroupSizeX: number;
  maxComputeWorkgroupsPerDimension: number;
}): { workgroupsTotal: number; x: number; y: number } {
  const workgroupsTotal = Math.ceil(n / workgroupSizeX);
  const x = Math.min(maxComputeWorkgroupsPerDimension, workgroupsTotal);
  const y = Math.ceil(workgroupsTotal / x);
  return { workgroupsTotal, x, y };
}

function generateCpuValuesU32(n: number, seed: number): Uint32Array {
  const out = new Uint32Array(n);
  fillDeterministicU32FromIndex(out, seed, 0);
  return out;
}

function ceilDiv(a: number, b: number): number {
  return Math.floor((a + b - 1) / b);
}

type U32Hash = { xor: number; sum: number; mix: number };

function hashU32Array(arr: Uint32Array): U32Hash {
  let xor = 0 >>> 0;
  let sum = 0 >>> 0;
  let mix = 0 >>> 0;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i]!;
    xor ^= v;
    sum = (sum + v) >>> 0;
    mix = (mix + Math.imul(v ^ (i >>> 0), 2654435761)) >>> 0;
  }
  return { xor: xor >>> 0, sum: sum >>> 0, mix: mix >>> 0 };
}

async function readbackU32Buffer(
  device: GPUDevice,
  src: GPUBuffer,
  n: number,
  label: string,
): Promise<Uint32Array> {
  const bytes = n * 4;
  const readback = device.createBuffer({
    label,
    size: bytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const enc = device.createCommandEncoder({ label: `${label}_encoder` });
  enc.copyBufferToBuffer(src, 0, readback, 0, bytes);
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
  await readback.mapAsync(GPUMapMode.READ);
  const ab = readback.getMappedRange().slice(0);
  readback.unmap();
  return new Uint32Array(ab);
}

function checkSortedU32(arr: Uint32Array): {
  ok: boolean;
  firstBadIndex: number;
} {
  let prev = arr[0] ?? 0;
  for (let i = 1; i < arr.length; i++) {
    const cur = arr[i]!;
    if (cur < prev) return { ok: false, firstBadIndex: i };
    prev = cur;
  }
  return { ok: true, firstBadIndex: -1 };
}

async function scatterSectorSanityCheck({
  device,
  scanCounts4,
  valuesOut,
  numBlocks,
  shift,
  samplePerSector = 16,
}: {
  device: GPUDevice;
  scanCounts4: GPUBuffer;
  valuesOut: GPUBuffer;
  numBlocks: number;
  shift: number;
  samplePerSector?: number;
}): Promise<void> {
  // Read totals (u32[4]) from scanCounts4[last] and verify each sector's first few values.
  const totalsReadback = device.createBuffer({
    label: "radix_totals_readback",
    size: 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encTotals = device.createCommandEncoder({
    label: "radix_totals_readback_encoder",
  });
  encTotals.copyBufferToBuffer(
    scanCounts4,
    (numBlocks - 1) * 16,
    totalsReadback,
    0,
    16,
  );
  device.queue.submit([encTotals.finish()]);
  await device.queue.onSubmittedWorkDone();
  await totalsReadback.mapAsync(GPUMapMode.READ);
  const totalsAb = totalsReadback.getMappedRange().slice(0);
  totalsReadback.unmap();
  const totals = new Uint32Array(totalsAb); // [t0,t1,t2,t3]

  const base0 = 0;
  const base1 = totals[0] >>> 0;
  const base2 = (totals[0] + totals[1]) >>> 0;
  const base3 = (totals[0] + totals[1] + totals[2]) >>> 0;

  const sampleSectors = [
    { digit: 0, base: base0 },
    { digit: 1, base: base1 },
    { digit: 2, base: base2 },
    { digit: 3, base: base3 },
  ] as const;

  const outSampleReadback = device.createBuffer({
    label: "radix_out_sample_readback",
    size: samplePerSector * 4 * sampleSectors.length,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encOutSample = device.createCommandEncoder({
    label: "radix_out_sample_encoder",
  });
  for (let s = 0; s < sampleSectors.length; s++) {
    const offBytes = sampleSectors[s]!.base * 4;
    const dstBytes = s * samplePerSector * 4;
    encOutSample.copyBufferToBuffer(
      valuesOut,
      offBytes,
      outSampleReadback,
      dstBytes,
      samplePerSector * 4,
    );
  }
  device.queue.submit([encOutSample.finish()]);
  await device.queue.onSubmittedWorkDone();
  await outSampleReadback.mapAsync(GPUMapMode.READ);
  const outSampleAb = outSampleReadback.getMappedRange().slice(0);
  outSampleReadback.unmap();
  const outSample = new Uint32Array(outSampleAb);

  const sectorOk: Record<number, boolean> = {};
  for (let s = 0; s < sampleSectors.length; s++) {
    const digit = sampleSectors[s]!.digit;
    let ok = true;
    for (let i = 0; i < samplePerSector; i++) {
      const v = outSample[s * samplePerSector + i]!;
      const d = (v >>> shift) & 3;
      if (d !== digit) {
        ok = false;
        break;
      }
    }
    sectorOk[digit] = ok;
  }

  console.info("[radix-sort-webgpu] scatter sector sanity", {
    totals: Array.from(totals),
    bases: [base0, base1, base2, base3],
    sectorOk,
  });
}

export async function runRadixSortWebgpuSetupDemo(): Promise<RadixSortWebgpuDemoResult> {
  const n = RADIX_SORT_WEBGPU_DEMO_N;
  const workgroupSizeX = RADIX_SORT_WEBGPU_DEMO_WORKGROUP_SIZE_X;
  const seed = RADIX_SORT_WEBGPU_DEMO_SEED;
  const sampleCount = RADIX_SORT_WEBGPU_DEMO_SAMPLE_COUNT;
  const passes = 16; // 32-bit / 2-bits per pass
  const bitsPerPass = 2;

  console.info("[radix-sort-webgpu] init", {
    n,
    workgroupSizeX,
    seed,
    sampleCount,
    passes,
  });

  const bytes = n * 4;

  const { adapter, device } = await requestDeviceWithMaxLimits();

  const lim = limitsSummary(device);
  console.info("[radix-sort-webgpu] maxComputeWorkgroupsPerDimension", {
    maxComputeWorkgroupsPerDimension: lim.maxComputeWorkgroupsPerDimension,
  });

  if (bytes > lim.maxBufferSize) {
    throw new Error(
      `Buffer too large for this device: need ${formatBytes(bytes)}, ` +
        `maxBufferSize=${formatBytes(lim.maxBufferSize)}.`,
    );
  }
  if (bytes > lim.maxStorageBufferBindingSize) {
    throw new Error(
      `Storage binding too large: need ${formatBytes(bytes)}, ` +
        `maxStorageBufferBindingSize=${formatBytes(lim.maxStorageBufferBindingSize)}.`,
    );
  }

  // Keep this around for later GPU compute kernels (radix sort will likely need 2D dispatch).
  const dispatch = computeDispatchShape({
    n,
    workgroupSizeX,
    maxComputeWorkgroupsPerDimension: lim.maxComputeWorkgroupsPerDimension,
  });

  console.info("[radix-sort-webgpu] preparing buffers", {
    valuesBytes: bytes,
    sampleBytes: sampleCount * 4,
  });
  const values = device.createBuffer({
    label: "radix_values_u32",
    size: bytes,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });

  // counts4_out: u32[numBlocks * 4] (packed uint4 per 512-element block).
  const numBlocks = ceilDiv(n, RADIX01_BLOCK_SIZE);
  const countsU32 = numBlocks * 4;
  const countsBytes = countsU32 * 4;
  const counts4Out = device.createBuffer({
    label: "radix_counts4_out_u32",
    size: countsBytes,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });
  void counts4Out; // placeholder: will be used by radix stage 01
  console.info("[radix-sort-webgpu] allocated counts4_out", {
    numBlocks,
    countsU32,
    countsBytes: formatBytes(countsBytes),
    formulaBytes: "ceil(n/512) * 4(digits) * 4(bytes/u32)",
  });

  const t0 =
    typeof performance !== "undefined" ? performance.now() : Date.now();
  console.info("[radix-sort-webgpu] generating CPU input array");
  const cpuValues = generateCpuValuesU32(n, seed);
  const inputHash = hashU32Array(cpuValues);
  const t1 =
    typeof performance !== "undefined" ? performance.now() : Date.now();

  console.info(
    "[radix-sort-webgpu] uploading CPU array to GPU (single writeBuffer)",
    {
      bytes: formatBytes(cpuValues.byteLength),
    },
  );
  device.queue.writeBuffer(
    values,
    0,
    cpuValues.buffer,
    cpuValues.byteOffset,
    cpuValues.byteLength,
  );
  const t2 =
    typeof performance !== "undefined" ? performance.now() : Date.now();

  console.info("[radix-sort-webgpu] waiting for upload to complete");
  await device.queue.onSubmittedWorkDone();

  // Allocate scan buffers for stage 02/03 (hierarchical scan over blocks).
  const scanCounts4 = device.createBuffer({
    label: "radix_scan_counts4_u32",
    size: countsBytes,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });

  const numChunks = ceilDiv(numBlocks, RADIX02_SCAN_BLOCK_LEN);
  const sumsBytes = numChunks * 4 * 4;
  const sums1_4 = device.createBuffer({
    label: "radix_sums1_4_u32",
    size: sumsBytes,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });
  const scanSums1_4 = device.createBuffer({
    label: "radix_scan_sums1_4_u32",
    size: sumsBytes,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });
  const dummySums4 = device.createBuffer({
    label: "radix_dummy_sums4_u32",
    size: 4 * 4,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });

  console.info("[radix-sort-webgpu] allocated scan buffers", {
    numBlocks,
    numChunks,
    scanCounts4Bytes: formatBytes(countsBytes),
    sumsBytes: formatBytes(sumsBytes),
  });

  const valuesOut = device.createBuffer({
    label: "radix_values_out_u32",
    size: bytes,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });

  if (
    Number(device.limits.maxComputeWorkgroupSizeX) < 256 ||
    Number(device.limits.maxComputeInvocationsPerWorkgroup) < 256
  ) {
    throw new Error(
      `Device does not support workgroup_size_x=256 (limits: maxComputeWorkgroupSizeX=${device.limits.maxComputeWorkgroupSizeX}, maxComputeInvocationsPerWorkgroup=${device.limits.maxComputeInvocationsPerWorkgroup}).`,
    );
  }

  const radix01 = await createRadixSort01LocalCountingKernel({ device });

  // Stage 02/03: hierarchical scan over blocks for each digit column (packed uint4).
  console.info(
    "[radix-sort-webgpu] dispatch radix_sort_02/03 (scan counts4 over blocks)",
  );
  const scanL0 = await createRadixSort02GlobalPrefixesScanSumReductionKernel({
    device,
  });
  const scanL1 = await createRadixSort02GlobalPrefixesScanSumReductionKernel({
    device,
    label: "radix_sort_02_global_prefixes_scan_sum_reduction_top",
  });
  const downL0 = await createRadixSort03GlobalPrefixesScanAccumulationKernel({
    device,
  });

  const scatter = await createRadixSort04ScatterKernel({ device });

  // Multi-pass radix sort (2-bit digits): shift = 0,2,...,30; swap buffers each pass.
  let inBuf = values;
  let outBuf = valuesOut;
  const tSort0 =
    typeof performance !== "undefined" ? performance.now() : Date.now();
  for (let pass = 0; pass < passes; pass++) {
    const shift = pass * bitsPerPass;
    console.info("[radix-sort-webgpu] pass", { pass, shift });

    const enc = device.createCommandEncoder({
      label: `radix_pass_shift${shift}_encoder`,
    });
    radix01.encode(enc, { values: inBuf, counts4Out, n, shift });
    scanL0.encode(enc, {
      input4: counts4Out,
      outputScan4: scanCounts4,
      outputBlockSums4: sums1_4,
      len: numBlocks,
    });
    scanL1.encode(enc, {
      input4: sums1_4,
      outputScan4: scanSums1_4,
      outputBlockSums4: dummySums4,
      len: numChunks,
    });
    downL0.encode(enc, {
      scanInOut4: scanCounts4,
      upperScan4: scanSums1_4,
      len: numBlocks,
    });
    scatter.encode(enc, {
      valuesIn: inBuf,
      scanCounts4,
      valuesOut: outBuf,
      n,
      shift,
    });
    device.queue.submit([enc.finish()]);

    if (pass === 0) {
      await device.queue.onSubmittedWorkDone();
      await scatterSectorSanityCheck({
        device,
        scanCounts4,
        valuesOut: outBuf,
        numBlocks,
        shift,
      });
    }

    const tmp = inBuf;
    inBuf = outBuf;
    outBuf = tmp;
  }

  // Wait for all queued passes before readback/verification.
  await device.queue.onSubmittedWorkDone();
  const tSort1 =
    typeof performance !== "undefined" ? performance.now() : Date.now();
  const radixSortSeconds = Math.max(0, tSort1 - tSort0) / 1000;

  // Like in C++: treat the full sort as one effective pass (read + write).
  const memorySizeGb = (4 * 2 * n) / 1024 / 1024 / 1024;
  const effectiveBandwidthGbps =
    radixSortSeconds > 0 ? memorySizeGb / radixSortSeconds : Infinity;
  const uintMillionsPerSecond =
    radixSortSeconds > 0 ? n / 1_000_000 / radixSortSeconds : Infinity;

  console.info("[radix-sort-webgpu] GPU radix-sort effective VRAM bandwidth", {
    memorySizeGb,
    seconds: radixSortSeconds,
    gbPerSecond: effectiveBandwidthGbps,
    uintMillionsPerSecond,
  });

  const tRead0 =
    typeof performance !== "undefined" ? performance.now() : Date.now();
  // Pass count is even (16), so final result ends up back in the original `values` buffer.
  const gpuSorted = await readbackU32Buffer(
    device,
    inBuf,
    n,
    "radix_sorted_readback_u32",
  );
  const tRead1 =
    typeof performance !== "undefined" ? performance.now() : Date.now();

  const sortedCheck = checkSortedU32(gpuSorted);
  const outputHash = hashU32Array(gpuSorted);

  // "Like C++": compare with CPU reference sort when feasible.
  const cpuReferenceMaxN = 5_000_000;
  let cpuCompare: { ok: boolean; firstBadIndex: number } | null = null;
  let cpuSortMs: number | undefined;
  let cpuCompareMs: number | undefined;
  if (n <= cpuReferenceMaxN) {
    console.info("[radix-sort-webgpu] cpu reference sort (Uint32Array.sort)", {
      n,
    });
    const tCpu0 =
      typeof performance !== "undefined" ? performance.now() : Date.now();
    const ref = new Uint32Array(cpuValues);
    ref.sort();
    const tCpu1 =
      typeof performance !== "undefined" ? performance.now() : Date.now();
    cpuSortMs = Math.max(0, tCpu1 - tCpu0);

    const tCmp0 =
      typeof performance !== "undefined" ? performance.now() : Date.now();
    let bad = -1;
    for (let i = 0; i < n; i++) {
      if (ref[i] !== gpuSorted[i]) {
        bad = i;
        break;
      }
    }
    const tCmp1 =
      typeof performance !== "undefined" ? performance.now() : Date.now();
    cpuCompareMs = Math.max(0, tCmp1 - tCmp0);
    cpuCompare = { ok: bad === -1, firstBadIndex: bad };
  } else {
    console.info("[radix-sort-webgpu] cpu reference sort skipped (too large)", {
      n,
      cpuReferenceMaxN,
    });
  }

  const result: RadixSortWebgpuDemoResult = {
    n,
    passes,
    bitsPerPass,
    adapter: adapter.info?.description ?? "(no adapter info)",
    limits: {
      maxBufferSize: lim.maxBufferSize,
      maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize,
      maxComputeWorkgroupsPerDimension: lim.maxComputeWorkgroupsPerDimension,
    },
    dispatchInit: dispatch,
    timingsMs: {
      generateCPU: Math.max(0, t1 - t0),
      uploadGPU: Math.max(0, t2 - t1),
      radixSort: Math.max(0, tSort1 - tSort0),
      readback: Math.max(0, tRead1 - tRead0),
      cpuSort: cpuSortMs,
      cpuCompare: cpuCompareMs,
    },
    perf: {
      memorySizeGb,
      radixSortSeconds,
      effectiveBandwidthGbps,
      uintMillionsPerSecond,
    },
    verify: {
      sortedOk: sortedCheck.ok,
      firstBadIndex: sortedCheck.firstBadIndex,
      inputHash,
      outputHash,
      cpuCompare,
    },
  };

  console.info("[radix-sort-webgpu] done", {
    ...result,
    limitsFormatted: {
      maxBufferSize: formatBytes(result.limits.maxBufferSize),
      maxStorageBufferBindingSize: formatBytes(
        result.limits.maxStorageBufferBindingSize,
      ),
    },
  });

  return result;
}
