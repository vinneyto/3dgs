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
  cpu: {
    method: "Array.sort" | "Uint32Array.sort" | null;
    seconds: number | null;
    effectiveBandwidthGbps: number | null;
    uintMillionsPerSecond: number | null;
  };
  verify: {
    sortedOk: boolean;
    firstBadIndex: number;
    inputHash: { xor: number; sum: number; mix: number };
    outputHash: { xor: number; sum: number; mix: number };
    cpuCompare: { ok: boolean; firstBadIndex: number } | null;
  };
};

export type RadixSortPreparedInput = {
  n: number;
  seed: number;
  cpuValues: Uint32Array;
  inputHash: { xor: number; sum: number; mix: number };
  generateMs: number;
};

export function prepareRadixSortInput({
  n,
  seed,
}: {
  n: number;
  seed: number;
}): RadixSortPreparedInput {
  const t0 =
    typeof performance !== "undefined" ? performance.now() : Date.now();
  const cpuValues = generateCpuValuesU32(n, seed);
  const inputHash = hashU32Array(cpuValues);
  const t1 =
    typeof performance !== "undefined" ? performance.now() : Date.now();
  return { n, seed, cpuValues, inputHash, generateMs: Math.max(0, t1 - t0) };
}

export type RadixSortWebgpuGpuRun = {
  result: RadixSortWebgpuDemoResult;
  gpuSorted: Uint32Array;
};

export async function runRadixSortWebgpuGpu({
  prepared,
  workgroupSizeX = RADIX_SORT_WEBGPU_DEMO_WORKGROUP_SIZE_X,
  passes = 16,
  bitsPerPass = 2,
}: {
  prepared: RadixSortPreparedInput;
  workgroupSizeX?: number;
  passes?: number;
  bitsPerPass?: number;
}): Promise<RadixSortWebgpuGpuRun> {
  const { n, cpuValues, inputHash } = prepared;
  const bytes = n * 4;

  const { adapter, device } = await requestDeviceWithMaxLimits();
  const lim = limitsSummary(device);

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

  const dispatchInit = computeDispatchShape({
    n,
    workgroupSizeX,
    maxComputeWorkgroupsPerDimension: lim.maxComputeWorkgroupsPerDimension,
  });

  const values = device.createBuffer({
    label: "radix_values_u32",
    size: bytes,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });

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

  const tUpload0 =
    typeof performance !== "undefined" ? performance.now() : Date.now();
  device.queue.writeBuffer(
    values,
    0,
    cpuValues.buffer,
    cpuValues.byteOffset,
    cpuValues.byteLength,
  );
  await device.queue.onSubmittedWorkDone();
  const tUpload1 =
    typeof performance !== "undefined" ? performance.now() : Date.now();

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

  let inBuf = values;
  let outBuf = valuesOut;
  const tSort0 =
    typeof performance !== "undefined" ? performance.now() : Date.now();
  for (let pass = 0; pass < passes; pass++) {
    const shift = pass * bitsPerPass;
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

  await device.queue.onSubmittedWorkDone();
  const tSort1 =
    typeof performance !== "undefined" ? performance.now() : Date.now();
  const radixSortSeconds = Math.max(0, tSort1 - tSort0) / 1000;

  const memorySizeGb = (4 * 2 * n) / 1024 / 1024 / 1024;
  const effectiveBandwidthGbps =
    radixSortSeconds > 0 ? memorySizeGb / radixSortSeconds : Infinity;
  const uintMillionsPerSecond =
    radixSortSeconds > 0 ? n / 1_000_000 / radixSortSeconds : Infinity;

  const tRead0 =
    typeof performance !== "undefined" ? performance.now() : Date.now();
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
    dispatchInit,
    timingsMs: {
      generateCPU: prepared.generateMs,
      uploadGPU: Math.max(0, tUpload1 - tUpload0),
      radixSort: Math.max(0, tSort1 - tSort0),
      readback: Math.max(0, tRead1 - tRead0),
    },
    perf: {
      memorySizeGb,
      radixSortSeconds,
      effectiveBandwidthGbps,
      uintMillionsPerSecond,
    },
    cpu: {
      method: null,
      seconds: null,
      effectiveBandwidthGbps: null,
      uintMillionsPerSecond: null,
    },
    verify: {
      sortedOk: sortedCheck.ok,
      firstBadIndex: sortedCheck.firstBadIndex,
      inputHash,
      outputHash,
      cpuCompare: null,
    },
  };

  return { result, gpuSorted };
}

export function runRadixSortCpuArraySort({
  cpuValues,
  memorySizeGb,
}: {
  cpuValues: Uint32Array;
  memorySizeGb: number;
}): {
  ref: number[];
  cpu: RadixSortWebgpuDemoResult["cpu"];
  cpuSortMs: number;
} {
  const n = cpuValues.length;
  const cpuSortMethod = "Array.sort" as const;

  const tCpu0 =
    typeof performance !== "undefined" ? performance.now() : Date.now();
  const ref = Array.from(cpuValues);
  ref.sort((a, b) => a - b);
  const tCpu1 =
    typeof performance !== "undefined" ? performance.now() : Date.now();
  const cpuSortMs = Math.max(0, tCpu1 - tCpu0);
  const cpuSortSeconds = cpuSortMs / 1000;
  const cpuEffectiveBandwidthGbps =
    cpuSortSeconds > 0 ? memorySizeGb / cpuSortSeconds : Infinity;
  const cpuUintMillionsPerSecond =
    cpuSortSeconds > 0 ? n / 1_000_000 / cpuSortSeconds : Infinity;

  return {
    ref,
    cpu: {
      method: cpuSortMethod,
      seconds: cpuSortSeconds,
      effectiveBandwidthGbps: cpuEffectiveBandwidthGbps,
      uintMillionsPerSecond: cpuUintMillionsPerSecond,
    },
    cpuSortMs,
  };
}

export function compareCpuArraySortToGpu({
  ref,
  gpuSorted,
}: {
  ref: number[];
  gpuSorted: Uint32Array;
}): {
  cpuCompareMs: number;
  cpuCompare: { ok: boolean; firstBadIndex: number };
} {
  const n = gpuSorted.length;
  const tCmp0 =
    typeof performance !== "undefined" ? performance.now() : Date.now();
  let bad = -1;
  for (let i = 0; i < n; i++) {
    if (ref[i] >>> 0 !== gpuSorted[i]) {
      bad = i;
      break;
    }
  }
  const tCmp1 =
    typeof performance !== "undefined" ? performance.now() : Date.now();
  const cpuCompareMs = Math.max(0, tCmp1 - tCmp0);
  return { cpuCompareMs, cpuCompare: { ok: bad === -1, firstBadIndex: bad } };
}

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
  const prepared = prepareRadixSortInput({
    n: RADIX_SORT_WEBGPU_DEMO_N,
    seed: RADIX_SORT_WEBGPU_DEMO_SEED,
  });
  const { result, gpuSorted } = await runRadixSortWebgpuGpu({
    prepared,
    workgroupSizeX: RADIX_SORT_WEBGPU_DEMO_WORKGROUP_SIZE_X,
    passes: 16,
    bitsPerPass: 2,
  });
  const cpuSort = runRadixSortCpuArraySort({
    cpuValues: prepared.cpuValues,
    memorySizeGb: result.perf.memorySizeGb,
  });
  const cpuCmp = compareCpuArraySortToGpu({ ref: cpuSort.ref, gpuSorted });
  return {
    ...result,
    cpu: cpuSort.cpu,
    timingsMs: {
      ...result.timingsMs,
      cpuSort: cpuSort.cpuSortMs,
      cpuCompare: cpuCmp.cpuCompareMs,
    },
    verify: { ...result.verify, cpuCompare: cpuCmp.cpuCompare },
  };
}
