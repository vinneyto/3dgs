import type {
  ComputeNode,
  StorageBufferNode,
  WebGPURenderer,
} from "three/webgpu";
import { instancedArray } from "three/tsl";
import {
  PREFIX_SUM_BLOCK_SIZE,
  createAddBlockOffsetsCompute,
  createScan512AddOffsetsCompute,
  createScan512WriteSumsCompute,
} from "@/app/_shared/tsl/compute/prefixSumNodes";

function ceilDiv(a: number, b: number): number {
  return Math.floor((a + b - 1) / b);
}

type PrefixSumBuffers = {
  input: StorageBufferNode;
  output: StorageBufferNode;
};

/**
 * Imperative multi-dispatch exclusive prefix-sum (scan) for `uint` buffers.
 *
 * Algorithm shape matches the OpenCL version in GPGPUTasks2025:
 * - scan blocks (512 elems) and write per-block sums
 * - recursively scan sums to build per-block offsets (hierarchical)
 * - propagate offsets down
 * - final fused pass: scan + add offsets
 */
export class GPUPrefixSum {
  private gl: WebGPURenderer;

  private count = 16;
  private input: StorageBufferNode = instancedArray(16, "uint");
  private output: StorageBufferNode = instancedArray(16, "uint");

  private dispatches: ComputeNode[] = [];
  private dirty = true;

  // scratch (rebuilt when count changes)
  private dummySums: StorageBufferNode = instancedArray(1, "uint");
  private sums: StorageBufferNode[] = [];
  private offs: StorageBufferNode[] = [];

  constructor(gl: WebGPURenderer) {
    this.gl = gl;
  }

  /** Sets input buffer (and optionally element count). */
  setInputBuffer(buf: StorageBufferNode, count?: number): this {
    this.input = buf;
    if (typeof count === "number") this.count = Math.max(0, count | 0);
    this.dirty = true;
    return this;
  }

  /** Sets output buffer (must be at least `count` u32 elements). */
  setOutputBuffer(buf: StorageBufferNode): this {
    this.output = buf;
    this.dirty = true;
    return this;
  }

  /** Sets how many u32 elements to scan. */
  setCount(count: number): this {
    this.count = Math.max(0, count | 0);
    this.dirty = true;
    return this;
  }

  /** Current IO buffers (useful for callers that want defaults). */
  getBuffers(): PrefixSumBuffers {
    return { input: this.input, output: this.output };
  }

  getInputBuffer(): StorageBufferNode {
    return this.input;
  }

  getOutputBuffer(): StorageBufferNode {
    return this.output;
  }

  private rebuildIfNeeded(): void {
    if (!this.dirty) return;
    this.dirty = false;

    const n = this.count;
    if (n <= 0) {
      this.dispatches = [];
      return;
    }

    // Build sizes hierarchy: sizes[0]=n, sizes[1]=ceil(n/512), ... until <=512
    const sizes: number[] = [n];
    while (sizes[sizes.length - 1] > PREFIX_SUM_BLOCK_SIZE) {
      sizes.push(ceilDiv(sizes[sizes.length - 1], PREFIX_SUM_BLOCK_SIZE));
    }
    const levels = Math.max(0, sizes.length - 1);

    // Reallocate scratch buffers sized to current n.
    this.dummySums = instancedArray(1, "uint");
    this.sums = [];
    this.offs = [];
    for (let ell = 0; ell < levels; ell++) {
      this.sums.push(instancedArray(sizes[ell + 1], "uint"));
      this.offs.push(instancedArray(sizes[ell + 1], "uint"));
    }

    // Build dispatch list (same order as `usePrefixSumCompute.ts`)
    const ops: ComputeNode[] = [];
    if (levels === 0) {
      ops.push(
        createScan512WriteSumsCompute({
          inData: this.input,
          outData: this.output,
          blockSums: this.dummySums,
          n,
        }),
      );
      this.dispatches = ops;
      return;
    }

    // 1) Scan input -> output, write sums[0]
    ops.push(
      createScan512WriteSumsCompute({
        inData: this.input,
        outData: this.output,
        blockSums: this.sums[0],
        n,
      }),
    );

    // 2) Recursively scan sums upward: sums[ell] -> offs[ell], write sums[ell+1]
    for (let ell = 0; ell < levels - 1; ell++) {
      const nin = sizes[ell + 1];
      ops.push(
        createScan512WriteSumsCompute({
          inData: this.sums[ell],
          outData: this.offs[ell],
          blockSums: this.sums[ell + 1],
          n: nin,
        }),
      );
    }

    // 3) Top: scan last sums into offs[last] (one workgroup); dummy sums output.
    ops.push(
      createScan512WriteSumsCompute({
        inData: this.sums[levels - 1],
        outData: this.offs[levels - 1],
        blockSums: this.dummySums,
        n: sizes[levels],
      }),
    );

    // 4) Down: offs[ell] += offs[ell+1] per block.
    for (let ell = levels - 2; ell >= 0; ell--) {
      const nout = sizes[ell + 1];
      ops.push(
        createAddBlockOffsetsCompute({
          outData: this.offs[ell],
          blockOffsets: this.offs[ell + 1],
          n: nout,
        }),
      );
    }

    // 5) Final fused pass: scan input + add offs[0] into output.
    ops.push(
      createScan512AddOffsetsCompute({
        inData: this.input,
        outData: this.output,
        blockOffsets: this.offs[0],
        n,
      }),
    );

    this.dispatches = ops;
  }

  /**
   * Dispatches the prefix-sum passes on the renderer queue (fire-and-forget).
   *
   * Use this in render loops when you don't want to stall the CPU.
   */
  compute(): void {
    this.rebuildIfNeeded();
    for (const c of this.dispatches) {
      this.gl.compute(c);
    }
  }

  /**
   * Dispatches the prefix-sum passes and awaits GPU completion.
   *
   * Use this in benchmarks/tests when you need timing or readback correctness.
   */
  async computeAsync(): Promise<void> {
    this.compute();

    // three.js exposes the WebGPU device on the renderer backend.
    const backend = (this.gl as unknown as { backend?: { device?: GPUDevice } })
      .backend;
    const queue = backend?.device?.queue;
    if (!queue) {
      throw new Error("WebGPU queue is unavailable on renderer backend.");
    }
    await queue.onSubmittedWorkDone();
  }
}
