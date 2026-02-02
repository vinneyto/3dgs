import type {
  ComputeNode,
  StorageBufferNode,
  WebGPURenderer,
} from "three/webgpu";
import { IndirectStorageBufferAttribute } from "three/webgpu";
import { instancedArray, storage } from "three/tsl";
import {
  RADIX_WORKGROUP_SIZE,
  createBlockGroupBaseActiveCountCompute,
  createBlockBuildGroupHistsCompute,
  createBlockBuildGroupHistsActiveCountCompute,
  createBlockClearGroupHistsCompute,
  createBlockGroupBaseCompute,
  createBlockScatterStableCompute,
  createBlockScatterStableActiveCountCompute,
  createBlockTotalsActiveCountCompute,
  createBlockTotalsCompute,
  createRadixBuildIndirectDispatchArgsActiveCountCompute,
  createRadixInitIndicesCompute,
  createRadixInitIndicesActiveCountCompute,
  createScan256ExclusiveCompute,
} from "@/app/_shared/tsl/gaussian/radixSortIndices";

function ceilDiv(a: number, b: number): number {
  return Math.floor((a + b - 1) / b);
}

export type GPURadixSortBuffers = {
  indicesA: StorageBufferNode;
  indicesB: StorageBufferNode;
  groupHists: StorageBufferNode; // atomic<u32>[numGroups*256]
  totals: StorageBufferNode; // u32[256]
  bucketBase: StorageBufferNode; // u32[256]
  groupBase: StorageBufferNode; // u32[numGroups*256]
};

type DispatchSize = number[] | number | IndirectStorageBufferAttribute | null;
type DispatchOp = { node: ComputeNode; dispatchSize?: DispatchSize };

/**
 * Imperative multi-dispatch radix sort for indices by a u32 key buffer.
 *
 * This is intentionally non-React. Wrap it in a hook for R3F usage.
 *
 * Current implementation uses the deterministic "block radix" path:
 * - per-group histograms
 * - reduce to totals[256]
 * - scan totals -> bucketBase[256]
 * - build groupBase[numGroups*256]
 * - stable scatter per group (lane0)
 *
 * Sort result is written to `indicesA` (for 4 passes / 8 bits per pass).
 */
export class GPURadixSort {
  private gl: WebGPURenderer;

  private depthKeys: StorageBufferNode | null = null;
  private count = 0; // fixed-count mode
  private activeCount: StorageBufferNode | null = null; // active-count mode (u32[1])
  private maxCount = 0; // active-count mode dispatch size
  private descending = true;

  private numGroups = 0;
  private allocatedCount = 0;
  private allocatedNumGroups = 0;

  private indicesA: StorageBufferNode = instancedArray(0, "uint");
  private indicesB: StorageBufferNode = instancedArray(0, "uint");
  private groupHists: StorageBufferNode = instancedArray(0, "uint").toAtomic();
  private totals: StorageBufferNode = instancedArray(256, "uint");
  private bucketBase: StorageBufferNode = instancedArray(256, "uint");
  private groupBase: StorageBufferNode = instancedArray(0, "uint");

  // Active-count indirect dispatch (workgroupsIndirect args: u32[3])
  private activeDispatchAttr: IndirectStorageBufferAttribute | null = null;
  private activeDispatchArgs: StorageBufferNode | null = null;

  private dispatches: DispatchOp[] = [];
  private dirty = true;

  constructor(gl: WebGPURenderer) {
    this.gl = gl;
  }

  setDepthKeysBuffer(buf: StorageBufferNode | null): this {
    if (this.depthKeys !== buf) {
      this.depthKeys = buf;
      this.dirty = true;
    }
    return this;
  }

  setCount(count: number): this {
    const n = Math.max(0, count | 0);
    if (this.count !== n) {
      this.count = n;
      this.dirty = true;
    }
    return this;
  }

  /**
   * Enables "active-count" mode (GPU-computed element count).
   *
   * - `activeCount` is a u32[1] buffer containing N active elements.
   * - `maxCount` is the maximum capacity / dispatch size.
   */
  setActiveCountBuffer(
    activeCount: StorageBufferNode | null,
    maxCount?: number,
  ): this {
    const m = Math.max(0, (maxCount ?? this.maxCount) | 0);
    if (this.activeCount !== activeCount || this.maxCount !== m) {
      this.activeCount = activeCount;
      this.maxCount = m;
      this.dirty = true;
    }
    return this;
  }

  setDescending(descending: boolean): this {
    if (this.descending !== descending) {
      this.descending = descending;
      this.dirty = true;
    }
    return this;
  }

  /** Final output buffer: `sortedIndices[drawInstance] -> originalIndex`. */
  getSortedIndicesBuffer(): StorageBufferNode {
    return this.indicesA;
  }

  getBuffers(): GPURadixSortBuffers {
    return {
      indicesA: this.indicesA,
      indicesB: this.indicesB,
      groupHists: this.groupHists,
      totals: this.totals,
      bucketBase: this.bucketBase,
      groupBase: this.groupBase,
    };
  }

  /**
   * Ensures internal buffers and dispatch list are built for current settings,
   * but does NOT submit any GPU work.
   *
   * Call this in React render/memo to avoid zero-sized bindings before first frame.
   */
  prepare(): void {
    this.rebuildIfNeeded();
  }

  private getDispatchCount(): number {
    if (this.activeCount) return this.maxCount;
    return this.count;
  }

  private rebuildIfNeeded(): void {
    if (!this.dirty) return;
    this.dirty = false;

    const dispatchCount = this.getDispatchCount();
    const depthKeys = this.depthKeys;
    if (!depthKeys || dispatchCount <= 0) {
      // Keep buffers as-is (so callers can keep references) but no-op dispatch list.
      this.dispatches = [];
      return;
    }

    const numGroups = ceilDiv(dispatchCount, RADIX_WORKGROUP_SIZE);
    this.numGroups = numGroups;

    const groupBins = numGroups * 256;

    // Allocate / reallocate scratch buffers only when sizes change.
    if (this.allocatedCount !== dispatchCount) {
      this.indicesA = instancedArray(dispatchCount, "uint");
      this.indicesB = instancedArray(dispatchCount, "uint");
      this.allocatedCount = dispatchCount;
    }

    if (this.allocatedNumGroups !== numGroups) {
      this.groupHists = instancedArray(groupBins, "uint").toAtomic();
      this.groupBase = instancedArray(groupBins, "uint");
      this.allocatedNumGroups = numGroups;
    }

    // These are always 256, but keep them re-creatable in case three.js backend
    // needs fresh nodes after context changes.
    this.totals = instancedArray(256, "uint");
    this.bucketBase = instancedArray(256, "uint");

    // Build multi-dispatch list.
    const ops: DispatchOp[] = [];
    if (this.activeCount) {
      // Indirect dispatch args (u32[3]) used to reduce idle workgroups.
      if (!this.activeDispatchAttr) {
        const init = new Uint32Array(3);
        init[0] = 1;
        init[1] = 1;
        init[2] = 1;
        this.activeDispatchAttr = new IndirectStorageBufferAttribute(init, 3);
        this.activeDispatchAttr.needsUpdate = true;
        this.activeDispatchArgs = storage(this.activeDispatchAttr, "uint", 3);
      }

      ops.push({
        node: createRadixBuildIndirectDispatchArgsActiveCountCompute({
          activeCount: this.activeCount,
          dispatchArgs: this.activeDispatchArgs!,
        }),
      });

      ops.push(
        {
          node: createRadixInitIndicesActiveCountCompute({
            indices: this.indicesA,
            activeCount: this.activeCount,
            maxCount: dispatchCount,
          }),
          dispatchSize: this.activeDispatchAttr,
        },
      );
    } else {
      ops.push({ node: createRadixInitIndicesCompute(this.indicesA, dispatchCount) });
    }

    const shifts = [0, 8, 16, 24] as const;
    for (let passIndex = 0; passIndex < shifts.length; passIndex++) {
      const shift = shifts[passIndex];
      const inBuf = passIndex % 2 === 0 ? this.indicesA : this.indicesB;
      const outBuf = passIndex % 2 === 0 ? this.indicesB : this.indicesA;

      const activeDispatch = this.activeCount ? this.activeDispatchAttr : undefined;

      ops.push({
        node: createBlockClearGroupHistsCompute(this.groupHists, groupBins),
        dispatchSize: activeDispatch,
      });

      ops.push(
        this.activeCount
          ? {
              node: createBlockBuildGroupHistsActiveCountCompute({
                depthKeys,
                indicesIn: inBuf,
                groupHists: this.groupHists,
                activeCount: this.activeCount,
                maxCount: dispatchCount,
                numGroups,
                shift,
                descending: this.descending,
              }),
              dispatchSize: activeDispatch,
            }
          : {
              node: createBlockBuildGroupHistsCompute({
                depthKeys,
                indicesIn: inBuf,
                groupHists: this.groupHists,
                count: dispatchCount,
                numGroups,
                shift,
                descending: this.descending,
              }),
            },
      );

      ops.push(
        this.activeCount
          ? {
              node: createBlockTotalsActiveCountCompute({
                groupHists: this.groupHists,
                totals: this.totals,
                activeCount: this.activeCount,
              }),
            }
          : {
              node: createBlockTotalsCompute({
                groupHists: this.groupHists,
                totals: this.totals,
                numGroups,
              }),
            },
      );

      ops.push({
        node: createScan256ExclusiveCompute({
          input: this.totals,
          output: this.bucketBase,
          name: "BlockRadixBucketBase256",
        }),
      });

      ops.push(
        this.activeCount
          ? {
              node: createBlockGroupBaseActiveCountCompute({
                groupHists: this.groupHists,
                bucketBase: this.bucketBase,
                groupBase: this.groupBase,
                activeCount: this.activeCount,
              }),
            }
          : {
              node: createBlockGroupBaseCompute({
                groupHists: this.groupHists,
                bucketBase: this.bucketBase,
                groupBase: this.groupBase,
                numGroups,
              }),
            },
      );

      ops.push(
        this.activeCount
          ? {
              node: createBlockScatterStableActiveCountCompute({
                depthKeys,
                indicesIn: inBuf,
                indicesOut: outBuf,
                groupBase: this.groupBase,
                activeCount: this.activeCount,
                maxCount: dispatchCount,
                numGroups,
                shift,
                descending: this.descending,
              }),
              dispatchSize: activeDispatch,
            }
          : {
              node: createBlockScatterStableCompute({
                depthKeys,
                indicesIn: inBuf,
                indicesOut: outBuf,
                groupBase: this.groupBase,
                count: dispatchCount,
                numGroups,
                shift,
                descending: this.descending,
              }),
            },
      );
    }

    this.dispatches = ops;
  }

  /** Dispatches the radix-sort passes on the renderer queue (fire-and-forget). */
  dispatch(): void {
    this.rebuildIfNeeded();
    for (const { node, dispatchSize } of this.dispatches) {
      this.gl.compute(node, dispatchSize);
    }
  }

  /** Dispatches the passes and awaits GPU completion (useful for benchmarks/tests). */
  async dispatchAsync(): Promise<void> {
    this.dispatch();

    const backend = (this.gl as unknown as { backend?: { device?: GPUDevice } })
      .backend;
    const queue = backend?.device?.queue;
    if (!queue) {
      throw new Error("WebGPU queue is unavailable on renderer backend.");
    }
    await queue.onSubmittedWorkDone();
  }
}
