import type { StorageBufferNode, WebGPURenderer } from "three/webgpu";
import { WebGPURenderer as ThreeWebGPURenderer } from "three/webgpu";
import { instancedArray, storage } from "three/tsl";
import { formatBytes } from "@/app/_shared/utils/formatBytes";
import {
  RADIX01_BLOCK_SIZE,
  RADIX02_SCAN_BLOCK_LEN,
  createRadixSort01LocalCountingCompute,
  createRadixSort02ScanSumReductionCompute,
  createRadixSort03ScanAccumulationCompute,
  createRadixSort04ScatterCompute,
} from "@/app/_shared/tsl/compute/radixSortWebgpuPassNodes";
import type {
  RadixSortPreparedInput,
  RadixSortWebgpuDemoResult,
} from "../radix-sort-webgpu/runRadixSortWebgpuSetup";

function ceilDiv(a: number, b: number): number {
  return Math.floor((a + b - 1) / b);
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
  };
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

export class RadixSortThreeTslBenchmark {
  private canvas: HTMLCanvasElement | null = null;
  private renderer: WebGPURenderer | null = null;
  private adapter: GPUAdapter | null = null;
  private device: GPUDevice | null = null;
  private initialized = false;

  async init(): Promise<void> {
    if (this.initialized) return;

    if (typeof navigator === "undefined" || !("gpu" in navigator)) {
      throw new Error("WebGPU is not available in this environment.");
    }

    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter) {
      throw new Error("Failed to acquire WebGPU adapter.");
    }

    const requiredFeatures: GPUFeatureName[] = [];
    adapter.features.forEach((f) => requiredFeatures.push(f as GPUFeatureName));
    const requiredLimits = {
      maxStorageBufferBindingSize: Number(
        adapter.limits.maxStorageBufferBindingSize,
      ),
      maxBufferSize: Number(adapter.limits.maxBufferSize),
    } satisfies Record<string, number>;

    const device = await adapter.requestDevice({
      requiredFeatures,
      requiredLimits:
        requiredLimits as unknown as GPUDeviceDescriptor["requiredLimits"],
    });

    const canvas = document.createElement("canvas");
    canvas.width = 1;
    canvas.height = 1;

    const renderer = new ThreeWebGPURenderer({
      canvas,
      antialias: false,
      alpha: true,
      depth: false,
      stencil: false,
      powerPreference: "high-performance",
      device,
    });

    await renderer.init();
    renderer.setSize(1, 1, false);

    this.canvas = canvas;
    this.renderer = renderer;
    this.adapter = adapter;
    this.device = device;
    this.initialized = true;
  }

  dispose(): void {
    try {
      this.renderer?.dispose();
    } catch {
      // ignore
    }
    this.canvas?.remove();
    this.canvas = null;
    this.renderer = null;
    this.adapter = null;
    this.device = null;
    this.initialized = false;
  }

  private ensureStorageAttrUsages(nodes: StorageBufferNode[]): void {
    const gl = this.renderer as unknown as {
      backend?: {
        attributeUtils?: {
          createAttribute?: (attr: unknown, usage: number) => void;
        };
      };
    };
    const au = gl.backend?.attributeUtils;
    if (!au?.createAttribute) return;

    const usage =
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST;

    for (const n of nodes) {
      au.createAttribute(n.value, usage);
    }
  }

  private async readbackU32(
    renderer: WebGPURenderer,
    node: StorageBufferNode,
    n: number,
  ): Promise<Uint32Array> {
    const anyRenderer = renderer as unknown as {
      getArrayBufferAsync?: (attr: unknown) => Promise<ArrayBuffer>;
    };
    if (!anyRenderer.getArrayBufferAsync) {
      throw new Error(
        "WebGPURenderer.getArrayBufferAsync is unavailable for readback.",
      );
    }

    const ab = await anyRenderer.getArrayBufferAsync(node.value);
    const src = new Uint32Array(ab);
    return new Uint32Array(src.subarray(0, n));
  }

  async runGpu({
    prepared,
    workgroupSizeX = 256,
    passes = 16,
    bitsPerPass = 2,
  }: {
    prepared: RadixSortPreparedInput;
    workgroupSizeX?: number;
    passes?: number;
    bitsPerPass?: number;
  }): Promise<{
    result: RadixSortWebgpuDemoResult;
    gpuSorted: Uint32Array;
  }> {
    await this.init();

    const renderer = this.renderer!;
    const device = this.device!;
    const adapter = this.adapter!;

    const { n, cpuValues, inputHash } = prepared;
    const bytes = n * 4;
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
    if (
      Number(device.limits.maxComputeWorkgroupSizeX) < 256 ||
      Number(device.limits.maxComputeInvocationsPerWorkgroup) < 256
    ) {
      throw new Error(
        `Device does not support workgroup_size_x=256 (limits: maxComputeWorkgroupSizeX=${device.limits.maxComputeWorkgroupSizeX}, maxComputeInvocationsPerWorkgroup=${device.limits.maxComputeInvocationsPerWorkgroup}).`,
      );
    }
    if (workgroupSizeX !== 256) {
      throw new Error(
        `This TSL implementation currently requires workgroupSizeX=256, got ${workgroupSizeX}.`,
      );
    }

    const numBlocks = ceilDiv(n, RADIX01_BLOCK_SIZE);
    const countsU32 = numBlocks * 4;
    const numChunks = ceilDiv(numBlocks, RADIX02_SCAN_BLOCK_LEN);
    const sumsU32 = numChunks * 4;

    const valuesA = instancedArray(n, "uint") as StorageBufferNode;
    const valuesB = instancedArray(n, "uint") as StorageBufferNode;
    const counts4Out = instancedArray(countsU32, "uint") as StorageBufferNode;
    const scanCounts4 = instancedArray(countsU32, "uint") as StorageBufferNode;
    const sums1_4 = instancedArray(sumsU32, "uint") as StorageBufferNode;
    const scanSums1_4 = instancedArray(sumsU32, "uint") as StorageBufferNode;
    const dummySums4 = instancedArray(4, "uint") as StorageBufferNode;

    this.ensureStorageAttrUsages([
      valuesA,
      valuesB,
      counts4Out,
      scanCounts4,
      sums1_4,
      scanSums1_4,
      dummySums4,
    ]);

    const tUpload0 =
      typeof performance !== "undefined" ? performance.now() : Date.now();
    (valuesA.value.array as Uint32Array).set(cpuValues);
    valuesA.value.needsUpdate = true;
    await device.queue.onSubmittedWorkDone();
    const tUpload1 =
      typeof performance !== "undefined" ? performance.now() : Date.now();

    let inBuf = valuesA;
    let outBuf = valuesB;

    const tSort0 =
      typeof performance !== "undefined" ? performance.now() : Date.now();
    for (let pass = 0; pass < passes; pass++) {
      const shift = pass * bitsPerPass;
      const inRead = storage(inBuf.value, "uint", n).toReadOnly();
      const countsRead = storage(scanCounts4.value, "uint", countsU32).toReadOnly();
      const sumsRead = storage(sums1_4.value, "uint", sumsU32).toReadOnly();
      const scanSumsRead = storage(
        scanSums1_4.value,
        "uint",
        sumsU32,
      ).toReadOnly();

      renderer.compute(
        createRadixSort01LocalCountingCompute({
          values: inRead,
          counts4Out,
          n,
          shift,
        }),
      );

      renderer.compute(
        createRadixSort02ScanSumReductionCompute({
          input4: storage(counts4Out.value, "uint", countsU32).toReadOnly(),
          outputScan4: scanCounts4,
          outputBlockSums4: sums1_4,
          len: numBlocks,
          name: "radix_sort_02_scan_sum_reduction_l0_tsl",
        }),
      );

      renderer.compute(
        createRadixSort02ScanSumReductionCompute({
          input4: sumsRead,
          outputScan4: scanSums1_4,
          outputBlockSums4: dummySums4,
          len: numChunks,
          name: "radix_sort_02_scan_sum_reduction_l1_tsl",
        }),
      );

      renderer.compute(
        createRadixSort03ScanAccumulationCompute({
          scanInOut4: scanCounts4,
          upperScan4: scanSumsRead,
          len: numBlocks,
        }),
      );

      renderer.compute(
        createRadixSort04ScatterCompute({
          valuesIn: inRead,
          scanCounts4: countsRead,
          valuesOut: outBuf,
          n,
          shift,
        }),
      );

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
    const gpuSorted = await this.readbackU32(renderer, inBuf, n);
    const tRead1 =
      typeof performance !== "undefined" ? performance.now() : Date.now();

    const sortedCheck = checkSortedU32(gpuSorted);
    const outputHash = hashU32Array(gpuSorted);
    const workgroupsTotal = Math.ceil(n / workgroupSizeX);
    const x = Math.min(lim.maxComputeWorkgroupsPerDimension, workgroupsTotal);
    const y = Math.ceil(workgroupsTotal / x);

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
      dispatchInit: { workgroupsTotal, x, y },
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
}
