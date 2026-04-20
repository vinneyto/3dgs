import { instancedArray } from "three/tsl";
import type { StorageBufferNode, WebGPURenderer } from "three/webgpu";
import { WebGPURenderer as ThreeWebGPURenderer } from "three/webgpu";
import { GPUPrefixSum } from "@/app/_shared/gpu/GPUPrefixSum";

function fillInputLikeCpp(out: Uint32Array) {
  for (let i = 0; i < out.length; i++) out[i] = (3 * (i + 5) + 7) % 17;
}

function exclusiveScanU32(inData: Uint32Array, outData: Uint32Array): number {
  let sum = 0;
  for (let i = 0; i < inData.length; i++) {
    outData[i] = sum;
    sum += inData[i];
    if (sum > 0xffffffff) throw new Error("u32 overflow in CPU prefix-sum.");
  }
  return sum;
}

function medianMs(values: number[]): number {
  if (values.length === 0) return 0;
  const s = [...values].sort((a, b) => a - b);
  const mid = (s.length / 2) | 0;
  return s.length % 2 === 0 ? (s[mid - 1] + s[mid]) * 0.5 : s[mid];
}

function firstMismatchU32(a: Uint32Array, b: Uint32Array) {
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) if (a[i] !== b[i]) return { i, a: a[i], b: b[i] };
  return null;
}

export type PrefixSumBenchmarkParams = {
  n: number;
  iters: number;
  warmups?: number;
};

export type PrefixSumBenchmarkResult = {
  n: number;
  iters: number;
  bytesPerIter: number;
  gbPerIter: number;

  cpuGenMs: number;
  cpuTimesMs: number[];
  cpuMedianMs: number;
  cpuGbps: number;
  cpuTotalSum: number;
  cpuCheckOk: boolean;

  gpuTimesMs: number[];
  gpuMedianMs: number;
  gpuGbps: number;
  gpuCompareOk: boolean;
  gpuMismatch: { i: number; cpu: number; gpu: number } | null;

  sampleInput: number[];
  samplePrefix: number[];
  error: string | null;
};

export class PrefixSumBenchmark {
  private canvas: HTMLCanvasElement | null = null;
  private renderer: WebGPURenderer | null = null;
  private prefixSum: GPUPrefixSum | null = null;
  private initialized = false;

  async init(): Promise<void> {
    if (this.initialized) return;

    if (typeof navigator === "undefined" || !("gpu" in navigator)) {
      throw new Error("WebGPU is not available (navigator.gpu missing).");
    }

    // Request a device with higher limits when available (needed for large N).
    // Default WebGPU limits can cap storage buffer bindings at 128 MiB.
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter) {
      throw new Error("WebGPU adapter is unavailable.");
    }

    const canvas = document.createElement("canvas");
    canvas.width = 1;
    canvas.height = 1;

    // For large buffers, we need to ensure required limits cover the binding size.
    const limits = adapter.limits;
    const supportedMaxStorage = Number(limits.maxStorageBufferBindingSize);
    const supportedMaxBuffer = Number(limits.maxBufferSize);

    // Create a device now; per-run we will validate against device limits.
    const device = await adapter.requestDevice({
      // TypeScript DOM typings model this as `Iterable<GPUFeatureName>`,
      // but some TS setups treat `adapter.features` as `GPUFeatureName[] | Set<string>`.
      // We don't require any optional features for this benchmark.
      requiredFeatures: [],
      requiredLimits: {
        // Ask for the highest supported caps so we don't have to recreate the device
        // when user changes N within the supported range.
        maxStorageBufferBindingSize: supportedMaxStorage,
        maxBufferSize: supportedMaxBuffer,
      },
    });

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
    this.prefixSum = new GPUPrefixSum(renderer);
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
    this.prefixSum = null;
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
      // Must call as a method to preserve `this`.
      au.createAttribute(n.value, usage);
    }
  }

  async run(
    params: PrefixSumBenchmarkParams,
  ): Promise<PrefixSumBenchmarkResult> {
    const n = Math.max(0, params.n | 0);
    const iters = Math.max(1, params.iters | 0);
    const warmups = Math.max(0, (params.warmups ?? 2) | 0);

    const bytesPerIter = 2 * n * 4;
    const gbPerIter = bytesPerIter / 1024 / 1024 / 1024;

    try {
      // CPU first.
      const input = new Uint32Array(n);
      const cpuOut = new Uint32Array(n);

      const tg0 = performance.now();
      fillInputLikeCpp(input);
      const tg1 = performance.now();
      const cpuGenMs = tg1 - tg0;

      const cpuTimes: number[] = [];
      let cpuTotalSum = 0;
      for (let k = 0; k < iters; k++) {
        const t0 = performance.now();
        cpuTotalSum = exclusiveScanU32(input, cpuOut);
        const t1 = performance.now();
        cpuTimes.push(t1 - t0);
        await new Promise((r) => setTimeout(r, 0));
      }

      const cpuMedianMs = medianMs(cpuTimes);
      const cpuGbps = cpuMedianMs > 0 ? gbPerIter / (cpuMedianMs / 1000) : 0;
      const cpuCheckOk =
        n === 0
          ? true
          : cpuOut[0] === 0 && cpuOut[n - 1] + input[n - 1] === cpuTotalSum;

      const sampleCount = Math.min(15, n);
      const sampleInput = Array.from(input.subarray(0, sampleCount));
      const samplePrefix = Array.from(cpuOut.subarray(0, sampleCount));

      // GPU second.
      await this.init();
      const renderer = this.renderer!;
      const ps = this.prefixSum!;

      // Validate current device limits for this N (storage bindings are sized to the full buffer).
      const dev = (renderer as unknown as { backend?: { device?: GPUDevice } })
        .backend?.device;
      const maxStorage = Number(dev?.limits?.maxStorageBufferBindingSize ?? 0);
      const maxBuffer = Number(dev?.limits?.maxBufferSize ?? 0);
      const bytesNeeded = n * 4;
      if (bytesNeeded > maxStorage || bytesNeeded > maxBuffer) {
        throw new Error(
          `N=${n.toLocaleString()} requires a u32 buffer of ${bytesNeeded.toLocaleString()} bytes, but device limits are maxStorageBufferBindingSize=${maxStorage.toLocaleString()} and maxBufferSize=${maxBuffer.toLocaleString()}.`,
        );
      }

      const inNode = instancedArray(n, "uint") as StorageBufferNode;
      const outNode = instancedArray(n, "uint") as StorageBufferNode;

      // Upload input.
      (inNode.value.array as Uint32Array).set(input);
      inNode.value.needsUpdate = true;

      ps.setInputBuffer(inNode, n);
      ps.setOutputBuffer(outNode);
      ps.setCount(n);

      this.ensureStorageAttrUsages([inNode, outNode]);

      // Warmup (compile pipelines, allocate internals).
      for (let k = 0; k < warmups; k++) {
        await ps.computeAsync();
        await new Promise((r) => setTimeout(r, 0));
      }

      const gpuTimes: number[] = [];
      for (let k = 0; k < iters; k++) {
        const t0 = performance.now();
        await ps.computeAsync();
        const t1 = performance.now();
        gpuTimes.push(t1 - t0);
        await new Promise((r) => setTimeout(r, 0));
      }

      const gpuMedianMs = medianMs(gpuTimes);
      const gpuGbps = gpuMedianMs > 0 ? gbPerIter / (gpuMedianMs / 1000) : 0;

      const anyGl = renderer as unknown as {
        getArrayBufferAsync?: (attr: unknown) => Promise<ArrayBuffer>;
      };
      if (!anyGl.getArrayBufferAsync) {
        throw new Error(
          "WebGPURenderer.getArrayBufferAsync is not available (cannot read back GPU buffer).",
        );
      }

      const ab = await anyGl.getArrayBufferAsync(outNode.value);
      const gpuOut = new Uint32Array(ab).subarray(0, n);

      const mismatch = firstMismatchU32(cpuOut, gpuOut);

      return {
        n,
        iters,
        bytesPerIter,
        gbPerIter,

        cpuGenMs,
        cpuTimesMs: cpuTimes,
        cpuMedianMs,
        cpuGbps,
        cpuTotalSum,
        cpuCheckOk,

        gpuTimesMs: gpuTimes,
        gpuMedianMs,
        gpuGbps,
        gpuCompareOk: mismatch == null,
        gpuMismatch: mismatch
          ? { i: mismatch.i, cpu: mismatch.a, gpu: mismatch.b }
          : null,

        sampleInput,
        samplePrefix,
        error: null,
      };
    } catch (e) {
      return {
        n,
        iters,
        bytesPerIter,
        gbPerIter,

        cpuGenMs: 0,
        cpuTimesMs: [],
        cpuMedianMs: 0,
        cpuGbps: 0,
        cpuTotalSum: 0,
        cpuCheckOk: false,

        gpuTimesMs: [],
        gpuMedianMs: 0,
        gpuGbps: 0,
        gpuCompareOk: false,
        gpuMismatch: null,

        sampleInput: [],
        samplePrefix: [],
        error: e instanceof Error ? e.message : String(e),
      };
    }
  }
}
