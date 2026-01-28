import type { ComputeNode, StorageBufferNode, WebGPURenderer } from "three/webgpu";
import { IndirectStorageBufferAttribute, Matrix4 } from "three/webgpu";
import {
  Fn,
  If,
  Loop,
  add,
  atomicAdd,
  atomicLoad,
  atomicStore,
  bitAnd,
  bitXor,
  float,
  instanceIndex,
  int,
  invocationLocalIndex,
  mul,
  shiftRight,
  storage,
  uint,
  uniform,
  vec4,
  workgroupArray,
  workgroupBarrier,
} from "three/tsl";
import { instancedArray } from "three/tsl";
import { GPUPrefixSum } from "@/app/_shared/gpu/GPUPrefixSum";

export type SplatIndirectCullSortOutputs = {
  /** Maps draw-instance -> original splat index (sorted back-to-front). */
  sortedSplatIndices: StorageBufferNode;
  /** u32[1] (computed on GPU). */
  visibleCount: StorageBufferNode;
  /** Indirect args buffer: u32[5] for drawIndexedIndirect. */
  indirectAttribute: IndirectStorageBufferAttribute;
};

type UniformMatrices = {
  uModelViewMatrix: ReturnType<typeof uniform<Matrix4>>;
  uProjectionMatrix: ReturnType<typeof uniform<Matrix4>>;
};

function createSharedMatricesUniforms(): UniformMatrices {
  const uModelViewMatrix = uniform<Matrix4>(new Matrix4()).setName(
    "uCullModelViewMatrix",
  );
  const uProjectionMatrix = uniform<Matrix4>(new Matrix4()).setName(
    "uCullProjectionMatrix",
  );
  return { uModelViewMatrix, uProjectionMatrix };
}

function createFrustumFlagsCompute({
  centers,
  flags,
  maxCount,
  uniforms,
}: {
  centers: StorageBufferNode;
  flags: StorageBufferNode;
  maxCount: number;
  uniforms: UniformMatrices;
}): ComputeNode {
  const maxCountU = uint(maxCount);
  const zeroF = float(0.0);

  return Fn(() => {
    If(instanceIndex.lessThan(maxCountU), () => {
      const center = centers.element(instanceIndex);
      const clipPos = uniforms.uProjectionMatrix
        .mul(uniforms.uModelViewMatrix)
        .mul(vec4(center, 1.0));

      const w = clipPos.w;
      const minusW = mul(w, -1.0);

      // WebGPU NDC: x,y in [-w,w], z in [0,w], w > 0
      const inside = w
        .greaterThan(zeroF)
        .and(clipPos.x.greaterThanEqual(minusW))
        .and(clipPos.x.lessThanEqual(w))
        .and(clipPos.y.greaterThanEqual(minusW))
        .and(clipPos.y.lessThanEqual(w))
        .and(clipPos.z.greaterThanEqual(zeroF))
        .and(clipPos.z.lessThanEqual(w));

      flags
        .element(instanceIndex)
        .assign(inside.select(uint(1), uint(0)));
    });
  })()
    .compute(maxCount, [256, 1, 1])
    .setName("SplatCull_FrustumFlags");
}

function createCompactVisibleIndicesCompute({
  flags,
  scan,
  visibleIndices,
  maxCount,
}: {
  flags: StorageBufferNode;
  scan: StorageBufferNode;
  visibleIndices: StorageBufferNode;
  maxCount: number;
}): ComputeNode {
  const maxCountU = uint(maxCount);
  return Fn(() => {
    If(instanceIndex.lessThan(maxCountU), () => {
      const f = flags.element(instanceIndex);
      If(f.equal(uint(1)), () => {
        const pos = scan.element(instanceIndex);
        visibleIndices.element(pos).assign(instanceIndex);
      });
    });
  })()
    .compute(maxCount, [256, 1, 1])
    .setName("SplatCull_CompactVisibleIndices");
}

function createVisibleCountAndIndirectArgsCompute({
  flags,
  scan,
  visibleCount,
  indirectArgs,
  maxCount,
}: {
  flags: StorageBufferNode;
  scan: StorageBufferNode;
  visibleCount: StorageBufferNode; // u32[1]
  indirectArgs: StorageBufferNode; // u32[5] (indirect storage buffer)
  maxCount: number;
}): ComputeNode {
  const lastIndexU = uint(Math.max(0, maxCount - 1));

  return Fn(() => {
    // single invocation (maxCount is fixed; caller guards maxCount>0)
    const total = add(scan.element(lastIndexU), flags.element(lastIndexU));
    visibleCount.element(uint(0)).assign(total);
    // indirectArgs[1] = instanceCount
    indirectArgs.element(uint(1)).assign(total);
  })()
    .compute(1, [1, 1, 1])
    .setName("SplatCull_VisibleCount_IndirectArgs");
}

function createVisibleDepthKeysCompute({
  centers,
  visibleIndices,
  visibleCount,
  visibleDepthKeys,
  maxCount,
  uniforms,
}: {
  centers: StorageBufferNode;
  visibleIndices: StorageBufferNode;
  visibleCount: StorageBufferNode; // u32[1]
  visibleDepthKeys: StorageBufferNode;
  maxCount: number;
  uniforms: UniformMatrices;
}): ComputeNode {
  const maxCountU = uint(maxCount);
  return Fn(() => {
    If(instanceIndex.lessThan(maxCountU), () => {
      const active = visibleCount.element(uint(0));
      If(instanceIndex.lessThan(active), () => {
        const splatIndex = visibleIndices.element(instanceIndex);
        const center = centers.element(splatIndex);
        const clipPos = uniforms.uProjectionMatrix
          .mul(uniforms.uModelViewMatrix)
          .mul(vec4(center, 1.0));
        const di = int(mul(clipPos.z, 4096.0));
        const key = bitXor(uint(di), uint(0x80000000));
        visibleDepthKeys.element(instanceIndex).assign(key);
      });
    });
  })()
    .compute(maxCount, [256, 1, 1])
    .setName("SplatCull_VisibleDepthKeys");
}

type ActiveCountRadixSort = {
  computes: ComputeNode[];
  sortedIndices: StorageBufferNode;
};

function createRadixSortIndicesActiveCount({
  depthKeys,
  indicesA,
  indicesB,
  hist,
  offsets,
  counters,
  activeCount,
  maxCount,
  descending,
}: {
  depthKeys: StorageBufferNode;
  indicesA: StorageBufferNode;
  indicesB: StorageBufferNode;
  hist: StorageBufferNode; // atomic<u32>[256]
  offsets: StorageBufferNode; // u32[256]
  counters: StorageBufferNode; // atomic<u32>[256]
  activeCount: StorageBufferNode; // u32[1]
  maxCount: number;
  descending: boolean;
}): ActiveCountRadixSort {
  const maxCountU = uint(maxCount);

  const init: ComputeNode = Fn(() => {
    If(instanceIndex.lessThan(maxCountU), () => {
      const n = activeCount.element(uint(0));
      If(instanceIndex.lessThan(n), () => {
        indicesA.element(instanceIndex).assign(instanceIndex);
      });
    });
  })()
    .compute(maxCount, [256, 1, 1])
    .setName("RadixActive_InitIndices");

  const clearHist: ComputeNode = Fn(() => {
    atomicStore(hist.element(instanceIndex), uint(0));
  })()
    .compute(256, [256, 1, 1])
    .setName("RadixActive_ClearHist256");

  const scanToOffsetsAndInitCounters: ComputeNode = Fn(() => {
    const i = invocationLocalIndex; // 0..255
    const shared = workgroupArray("uint", 256);

    shared.element(i).assign(atomicLoad(hist.element(i)));
    workgroupBarrier();

    const offset = uint(1).toVar();
    const t = uint(0).toVar();
    Loop(offset.lessThan(uint(256)), () => {
      t.assign(uint(0));
      If(i.greaterThanEqual(offset), () => {
        t.assign(shared.element(i.sub(offset)));
      });
      workgroupBarrier();
      shared.element(i).assign(add(shared.element(i), t));
      workgroupBarrier();
      offset.assign(add(offset, offset));
    });

    const exclusive = uint(0).toVar();
    If(i.greaterThan(uint(0)), () => {
      exclusive.assign(shared.element(i.sub(uint(1))));
    });
    offsets.element(i).assign(exclusive);
    atomicStore(counters.element(i), exclusive);
  })()
    .compute(256, [256, 1, 1])
    .setName("RadixActive_Scan256_ToOffsets_InitCounters");

  // 4 passes (8 bits each)
  const shifts = [0, 8, 16, 24] as const;
  const computes: ComputeNode[] = [init];

  // Ping-pong: we use indicesA as "current input" for histogram and alternate after scatter.
  // We'll implement by swapping references per pass with separate scatter nodes.
  let inBuf: StorageBufferNode = indicesA;
  let outBuf: StorageBufferNode = indicesB;

  for (const shift of shifts) {
    const histogram = Fn(() => {
      If(instanceIndex.lessThan(maxCountU), () => {
        const n = activeCount.element(uint(0));
        If(instanceIndex.lessThan(n), () => {
          const idx = inBuf.element(instanceIndex);
          const key = depthKeys.element(idx);
          const rawBucket = bitAnd(shiftRight(key, uint(shift)), uint(0xff));
          const bucket = descending ? uint(255).sub(rawBucket) : rawBucket;
          atomicAdd(hist.element(bucket), uint(1));
        });
      });
    })()
      .compute(maxCount, [256, 1, 1])
      .setName(
        `RadixActive_Hist_shift${shift}${descending ? "_desc" : "_asc"}`,
      );

    const scatter = Fn(() => {
      If(instanceIndex.lessThan(maxCountU), () => {
        const n = activeCount.element(uint(0));
        If(instanceIndex.lessThan(n), () => {
          const idx = inBuf.element(instanceIndex);
          const key = depthKeys.element(idx);
          const rawBucket = bitAnd(shiftRight(key, uint(shift)), uint(0xff));
          const bucket = descending ? uint(255).sub(rawBucket) : rawBucket;
          const pos = atomicAdd(counters.element(bucket), uint(1));
          outBuf.element(pos).assign(idx);
        });
      });
    })()
      .compute(maxCount, [256, 1, 1])
      .setName(
        `RadixActive_Scatter_shift${shift}${descending ? "_desc" : "_asc"}`,
      );

    computes.push(clearHist, histogram, scanToOffsetsAndInitCounters, scatter);

    // swap
    const tmp = inBuf;
    inBuf = outBuf;
    outBuf = tmp;
  }

  // After 4 passes, data ends up in inBuf (because we swap at end of each pass).
  // Ensure sortedIndices points to the final inBuf.
  return { computes, sortedIndices: inBuf };
}

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
    .setName("SplatCull_GatherSortedSplatIndices");
}

/**
 * GPU-driven frustum culling + prefix-sum compaction + depth-key sort + indirect instanceCount.
 *
 * This class is intentionally non-React. Wrap it in a hook to integrate with R3F.
 */
export class GPUSplatIndirectCullSort {
  private gl: WebGPURenderer;
  private maxCount: number;
  private centers: StorageBufferNode;

  private uniforms: UniformMatrices;

  // Buffers (u32 unless noted)
  private flags: StorageBufferNode;
  private scan: StorageBufferNode;
  private visibleIndices: StorageBufferNode;
  private visibleCount: StorageBufferNode; // u32[1]
  private visibleDepthKeys: StorageBufferNode;
  private posA: StorageBufferNode;
  private posB: StorageBufferNode;
  private hist256: StorageBufferNode;
  private offsets256: StorageBufferNode;
  private counters256: StorageBufferNode;
  private sortedSplatIndices: StorageBufferNode;

  // Indirect draw args
  private indirectAttr: IndirectStorageBufferAttribute;
  private indirectArgs: StorageBufferNode;

  // Compute passes
  private computeFlags: ComputeNode;
  private prefixSum: GPUPrefixSum;
  private computeCompact: ComputeNode;
  private computeVisibleCountAndIndirect: ComputeNode;
  private computeVisibleDepthKeys: ComputeNode;
  private radixSorter: ActiveCountRadixSort;
  private computeGatherSorted: ComputeNode;

  constructor(gl: WebGPURenderer, params: { centers: StorageBufferNode; maxCount: number }) {
    this.gl = gl;
    this.centers = params.centers;
    this.maxCount = Math.max(0, params.maxCount | 0);

    this.uniforms = createSharedMatricesUniforms();

    // Allocate buffers
    this.flags = instancedArray(this.maxCount, "uint");
    this.scan = instancedArray(this.maxCount, "uint");
    this.visibleIndices = instancedArray(this.maxCount, "uint");
    this.visibleCount = instancedArray(1, "uint");
    this.visibleDepthKeys = instancedArray(this.maxCount, "uint");
    this.posA = instancedArray(this.maxCount, "uint");
    this.posB = instancedArray(this.maxCount, "uint");
    this.hist256 = instancedArray(256, "uint").toAtomic();
    this.offsets256 = instancedArray(256, "uint");
    this.counters256 = instancedArray(256, "uint").toAtomic();
    this.sortedSplatIndices = instancedArray(this.maxCount, "uint");

    // Indirect args (indexed): u32[5]
    const init = new Uint32Array(5);
    init[0] = 0; // indexCount (set later)
    init[1] = 0; // instanceCount (written by GPU each frame)
    init[2] = 0; // firstIndex
    init[3] = 0; // baseVertex
    init[4] = 0; // firstInstance
    this.indirectAttr = new IndirectStorageBufferAttribute(init, 5);
    this.indirectAttr.needsUpdate = true;
    this.indirectArgs = storage(this.indirectAttr, "uint", 5);

    // Build compute nodes
    this.computeFlags = createFrustumFlagsCompute({
      centers: this.centers,
      flags: this.flags,
      maxCount: this.maxCount,
      uniforms: this.uniforms,
    });

    this.prefixSum = new GPUPrefixSum(gl)
      .setInputBuffer(this.flags, this.maxCount)
      .setOutputBuffer(this.scan);

    this.computeCompact = createCompactVisibleIndicesCompute({
      flags: this.flags,
      scan: this.scan,
      visibleIndices: this.visibleIndices,
      maxCount: this.maxCount,
    });

    this.computeVisibleCountAndIndirect = createVisibleCountAndIndirectArgsCompute({
      flags: this.flags,
      scan: this.scan,
      visibleCount: this.visibleCount,
      indirectArgs: this.indirectArgs,
      maxCount: this.maxCount,
    });

    this.computeVisibleDepthKeys = createVisibleDepthKeysCompute({
      centers: this.centers,
      visibleIndices: this.visibleIndices,
      visibleCount: this.visibleCount,
      visibleDepthKeys: this.visibleDepthKeys,
      maxCount: this.maxCount,
      uniforms: this.uniforms,
    });

    this.radixSorter = createRadixSortIndicesActiveCount({
      depthKeys: this.visibleDepthKeys,
      indicesA: this.posA,
      indicesB: this.posB,
      hist: this.hist256,
      offsets: this.offsets256,
      counters: this.counters256,
      activeCount: this.visibleCount,
      maxCount: this.maxCount,
      descending: true,
    });

    this.computeGatherSorted = createGatherSortedSplatIndicesCompute({
      sortedPositions: this.radixSorter.sortedIndices,
      visibleIndices: this.visibleIndices,
      visibleCount: this.visibleCount,
      sortedSplatIndices: this.sortedSplatIndices,
      maxCount: this.maxCount,
    });
  }

  /** Sets `indexCount` inside indirect args (u32[5]). Call once when geometry is ready. */
  setIndirectIndexCount(indexCount: number): void {
    const arr = this.indirectAttr.array as Uint32Array;
    arr[0] = Math.max(0, indexCount | 0);
    this.indirectAttr.needsUpdate = true;
  }

  setMatrices(modelView: Matrix4, projection: Matrix4): void {
    this.uniforms.uModelViewMatrix.value.copy(modelView);
    this.uniforms.uProjectionMatrix.value.copy(projection);
  }

  /** Runs the full cull → scan → compact → sort → gather → indirect pipeline. */
  dispatch(): void {
    if (this.maxCount <= 0) return;

    this.gl.compute(this.computeFlags);
    this.prefixSum.compute();
    this.gl.compute(this.computeCompact);
    this.gl.compute(this.computeVisibleCountAndIndirect);
    this.gl.compute(this.computeVisibleDepthKeys);
    for (const c of this.radixSorter.computes) this.gl.compute(c);
    this.gl.compute(this.computeGatherSorted);
  }

  getOutputs(): SplatIndirectCullSortOutputs {
    return {
      sortedSplatIndices: this.sortedSplatIndices,
      visibleCount: this.visibleCount,
      indirectAttribute: this.indirectAttr,
    };
  }
}

