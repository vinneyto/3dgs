import type {
  ComputeNode,
  StorageBufferNode,
  WebGPURenderer,
} from "three/webgpu";
import { Fn, If, instanceIndex, uint } from "three/tsl";
import { instancedArray } from "three/tsl";
import { GPURadixSort } from "@/app/_shared/gpu/GPURadixSort";

function createGatherSortedSplatIndicesCompute({
  sortedPositions,
  visibleIndices,
  visibleCount,
  sortedSplatIndices,
  maxCount,
}: {
  sortedPositions: StorageBufferNode;
  visibleIndices: StorageBufferNode;
  visibleCount: StorageBufferNode; // u32[1]
  sortedSplatIndices: StorageBufferNode;
  maxCount: number;
}): ComputeNode {
  const maxCountU = uint(maxCount);
  return Fn(() => {
    If(instanceIndex.lessThan(maxCountU), () => {
      const n = visibleCount.element(uint(0));
      If(instanceIndex.lessThan(n), () => {
        const pos = sortedPositions.element(instanceIndex);
        const splatIndex = visibleIndices.element(pos);
        sortedSplatIndices.element(instanceIndex).assign(splatIndex);
      });
    });
  })()
    .compute(maxCount, [256, 1, 1])
    .setName("SplatDepthSort_GatherSortedSplatIndices");
}

export type SplatDepthSortOutputs = {
  /** Maps draw-instance -> original splat index (sorted back-to-front). */
  sortedSplatIndices: StorageBufferNode;
};

/**
 * GPU depth-key sort for an already-compacted visible list:
 *
 * Inputs:
 * - `visibleIndices[i]` is original splat index (dense in [0..visibleCount)).
 * - `visibleDepthKeys[i]` is u32 key for that splat (same dense indexing).
 * - `visibleCount[0]` is number of active elements (computed on GPU).
 *
 * Output:
 * - `sortedSplatIndices[i]` maps draw instance -> original splat index, sorted.
 */
export class GPUSplatDepthSort {
  private gl: WebGPURenderer;
  private maxCount: number;

  private visibleIndices: StorageBufferNode;
  private visibleDepthKeys: StorageBufferNode;
  private visibleCount: StorageBufferNode;

  private radix: GPURadixSort;
  private sortedSplatIndices: StorageBufferNode;
  private computeGather: ComputeNode;

  private descending = true;

  constructor(
    gl: WebGPURenderer,
    params: {
      visibleIndices: StorageBufferNode;
      visibleDepthKeys: StorageBufferNode;
      visibleCount: StorageBufferNode;
      maxCount: number;
      descending?: boolean;
    },
  ) {
    this.gl = gl;
    this.visibleIndices = params.visibleIndices;
    this.visibleDepthKeys = params.visibleDepthKeys;
    this.visibleCount = params.visibleCount;
    this.maxCount = Math.max(0, params.maxCount | 0);
    this.descending = params.descending ?? true;

    this.radix = new GPURadixSort(gl)
      .setDepthKeysBuffer(this.visibleDepthKeys)
      .setActiveCountBuffer(this.visibleCount, this.maxCount)
      .setDescending(this.descending);

    // IMPORTANT: allocate radix internal buffers now, so `computeGather` captures
    // the correct (non-zero-sized) `sortedPositions` storage buffer node.
    this.radix.prepare();

    this.sortedSplatIndices = instancedArray(this.maxCount, "uint");

    this.computeGather = createGatherSortedSplatIndicesCompute({
      sortedPositions: this.radix.getSortedIndicesBuffer(),
      visibleIndices: this.visibleIndices,
      visibleCount: this.visibleCount,
      sortedSplatIndices: this.sortedSplatIndices,
      maxCount: this.maxCount,
    });
  }

  setDescending(descending: boolean): void {
    if (this.descending === descending) return;
    this.descending = descending;
    this.radix.setDescending(descending);
  }

  getOutputs(): SplatDepthSortOutputs {
    return { sortedSplatIndices: this.sortedSplatIndices };
  }

  dispatch(): void {
    if (this.maxCount <= 0) return;
    this.radix.dispatch();
    this.gl.compute(this.computeGather);
  }
}
