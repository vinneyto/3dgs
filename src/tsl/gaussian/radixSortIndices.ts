import type { ComputeNode, StorageBufferNode } from "three/webgpu";
import {
  Fn,
  If,
  Loop,
  add,
  bitAnd,
  instanceIndex,
  invocationLocalIndex,
  workgroupId,
  shiftRight,
  uint,
  workgroupArray,
  workgroupBarrier,
} from "three/tsl";
import { atomicAdd, atomicLoad, atomicStore } from "three/tsl";

export type RadixSortIndices = {
  /**
   * Run these compute passes in order each frame after `depthKeys` has been filled.
   * (Radix sort is a multi-dispatch algorithm.)
   */
  computes: ComputeNode[];
  /** Final output after all passes (for 4 passes this is always `indicesA`). */
  sortedIndices: StorageBufferNode;
  buffers: {
    indicesA: StorageBufferNode;
    indicesB: StorageBufferNode;
    hist: StorageBufferNode;
    counters: StorageBufferNode;
    offsets: StorageBufferNode;
  };
};

export type BlockRadixSortIndices = {
  computes: ComputeNode[];
  sortedIndices: StorageBufferNode;
  buffers: {
    indicesA: StorageBufferNode;
    indicesB: StorageBufferNode;
    groupHists: StorageBufferNode;
    totals: StorageBufferNode;
    bucketBase: StorageBufferNode;
    groupBase: StorageBufferNode;
  };
};

type Pass = {
  clearHist: ComputeNode;
  histogram: ComputeNode;
  scanToOffsetsAndInitCounters: ComputeNode;
  scatter: ComputeNode;
};

function createInitIndicesCompute(
  indices: StorageBufferNode,
  count: number
): ComputeNode {
  return Fn(() => {
    // three.js does not auto-guard out-of-range invocations when count is not a multiple of workgroup size.
    If(instanceIndex.lessThan(uint(count)), () => {
      indices.element(instanceIndex).assign(instanceIndex);
    });
  })()
    .compute(count, [256, 1, 1])
    .setName("RadixInitIndices");
}

function createClearHistogramCompute(hist: StorageBufferNode): ComputeNode {
  // hist length is 256
  return Fn(() => {
    // hist is atomic<u32> => must use atomicStore
    atomicStore(hist.element(instanceIndex), uint(0));
  })()
    .compute(256, [256, 1, 1])
    .setName("RadixClearHistogram256");
}

function createHistogramCompute({
  depthKeys,
  indicesIn,
  hist,
  count,
  shift,
  descending,
}: {
  depthKeys: StorageBufferNode;
  indicesIn: StorageBufferNode;
  hist: StorageBufferNode;
  count: number;
  shift: number;
  descending: boolean;
}): ComputeNode {
  const shiftU = uint(shift);
  const maskU = uint(0xff);
  const maxBucketU = uint(255);

  return Fn(() => {
    If(instanceIndex.lessThan(uint(count)), () => {
      const idx = indicesIn.element(instanceIndex);
      const key = depthKeys.element(idx);

      const rawBucket = bitAnd(shiftRight(key, shiftU), maskU);
      const bucket = descending ? maxBucketU.sub(rawBucket) : rawBucket;

      // hist must be atomic (created with .toAtomic()).
      atomicAdd(hist.element(bucket), uint(1));
    });
  })()
    .compute(count, [256, 1, 1])
    .setName(`RadixHistogram_shift${shift}${descending ? "_desc" : "_asc"}`);
}

function createScanCompute({
  hist,
  offsets,
  counters,
}: {
  hist: StorageBufferNode;
  offsets: StorageBufferNode;
  counters: StorageBufferNode;
}): ComputeNode {
  // Parallel exclusive prefix-sum over 256 bins using 1 workgroup of 256 invocations.
  // This avoids the "single-thread scan" and is the canonical radix 256 scan step.
  return Fn(() => {
    const i = invocationLocalIndex; // 0..255
    const shared = workgroupArray("uint", 256);

    // Load atomic histogram into workgroup shared memory
    shared.element(i).assign(atomicLoad(hist.element(i)));
    workgroupBarrier();

    // Hillis–Steele inclusive scan (log2(256)=8 steps), then convert to exclusive.
    // Implemented as a while-style loop to avoid manual unrolling.
    const offset = uint(1).toVar();
    const t = uint(0).toVar();
    Loop(offset.lessThan(uint(256)), () => {
      // Read-from-previous state phase (must happen before any writes this iteration)
      t.assign(uint(0));
      If(i.greaterThanEqual(offset), () => {
        t.assign(shared.element(i.sub(offset)));
      });
      workgroupBarrier();

      // Write phase
      shared.element(i).assign(add(shared.element(i), t));
      workgroupBarrier();

      // offset *= 2
      offset.assign(add(offset, offset));
    });

    // Convert inclusive -> exclusive
    const exclusive = uint(0).toVar();
    If(i.greaterThan(uint(0)), () => {
      exclusive.assign(shared.element(i.sub(uint(1))));
    });

    offsets.element(i).assign(exclusive);
    atomicStore(counters.element(i), exclusive);
  })()
    .compute(256, [256, 1, 1])
    .setName("RadixScan256_ToOffsets_InitCounters");
}

function createScatterCompute({
  depthKeys,
  indicesIn,
  indicesOut,
  counters,
  count,
  shift,
  descending,
}: {
  depthKeys: StorageBufferNode;
  indicesIn: StorageBufferNode;
  indicesOut: StorageBufferNode;
  counters: StorageBufferNode;
  count: number;
  shift: number;
  descending: boolean;
}): ComputeNode {
  const shiftU = uint(shift);
  const maskU = uint(0xff);
  const maxBucketU = uint(255);

  return Fn(() => {
    If(instanceIndex.lessThan(uint(count)), () => {
      const idx = indicesIn.element(instanceIndex);
      const key = depthKeys.element(idx);

      const rawBucket = bitAnd(shiftRight(key, shiftU), maskU);
      const bucket = descending ? maxBucketU.sub(rawBucket) : rawBucket;

      // pos = counters[bucket]++
      const pos = atomicAdd(counters.element(bucket), uint(1));
      indicesOut.element(pos).assign(idx);
    });
  })()
    .compute(count, [256, 1, 1])
    .setName(`RadixScatter_shift${shift}${descending ? "_desc" : "_asc"}`);
}

function createPass({
  depthKeys,
  indicesIn,
  indicesOut,
  hist,
  offsets,
  counters,
  count,
  shift,
  descending,
}: {
  depthKeys: StorageBufferNode;
  indicesIn: StorageBufferNode;
  indicesOut: StorageBufferNode;
  hist: StorageBufferNode;
  offsets: StorageBufferNode;
  counters: StorageBufferNode;
  count: number;
  shift: number;
  descending: boolean;
}): Pass {
  return {
    clearHist: createClearHistogramCompute(hist),
    histogram: createHistogramCompute({
      depthKeys,
      indicesIn,
      hist,
      count,
      shift,
      descending,
    }),
    scanToOffsetsAndInitCounters: createScanCompute({
      hist,
      offsets,
      counters,
    }),
    scatter: createScatterCompute({
      depthKeys,
      indicesIn,
      indicesOut,
      counters,
      count,
      shift,
      descending,
    }),
  };
}

/**
 * LSD radix sort of `indices` by `depthKeys[ index ]`.
 *
 * - 32-bit keys
 * - 8 bits per digit => 4 passes
 * - stable (LSD) => correct total ordering
 *
 * `descending=true` sorts far->near (higher key first) without modifying `depthKeys`,
 * by flipping the bucket index: bucket' = 255 - bucket.
 */
export function createRadixSortIndices({
  depthKeys,
  count,
  indicesA,
  indicesB,
  hist,
  offsets,
  counters,
  descending = true,
}: {
  depthKeys: StorageBufferNode;
  count: number;
  indicesA: StorageBufferNode;
  indicesB: StorageBufferNode;
  hist: StorageBufferNode;
  offsets: StorageBufferNode;
  counters: StorageBufferNode;
  descending?: boolean;
}): RadixSortIndices {
  const init = createInitIndicesCompute(indicesA, count);

  const shifts = [0, 8, 16, 24] as const;
  const passes: Pass[] = shifts.map((shift, passIndex) => {
    const inBuf = passIndex % 2 === 0 ? indicesA : indicesB;
    const outBuf = passIndex % 2 === 0 ? indicesB : indicesA;

    return createPass({
      depthKeys,
      indicesIn: inBuf,
      indicesOut: outBuf,
      hist,
      offsets,
      counters,
      count,
      shift,
      descending,
    });
  });

  const computes: ComputeNode[] = [init];
  for (const p of passes) {
    computes.push(
      p.clearHist,
      p.histogram,
      p.scanToOffsetsAndInitCounters,
      p.scatter
    );
  }

  return {
    computes,
    sortedIndices: indicesA,
    buffers: { indicesA, indicesB, hist, counters, offsets },
  };
}

// ----------------------------
// Deterministic Block-Radix Sort
// ----------------------------

function createBlockClearGroupHistsCompute(
  groupHists: StorageBufferNode,
  groupHistsCount: number
): ComputeNode {
  return Fn(() => {
    atomicStore(groupHists.element(instanceIndex), uint(0));
  })()
    .compute(groupHistsCount, [256, 1, 1])
    .setName("BlockRadixClearGroupHists");
}

function createBlockBuildGroupHistsCompute({
  depthKeys,
  indicesIn,
  groupHists,
  count,
  numGroups,
  shift,
  descending,
}: {
  depthKeys: StorageBufferNode;
  indicesIn: StorageBufferNode;
  groupHists: StorageBufferNode;
  count: number;
  numGroups: number;
  shift: number;
  descending: boolean;
}): ComputeNode {
  const shiftU = uint(shift);
  const maskU = uint(0xff);
  const maxBucketU = uint(255);
  const numGroupsU = uint(numGroups);

  return Fn(() => {
    If(instanceIndex.lessThan(uint(count)), () => {
      const g = workgroupId.x;
      If(g.lessThan(numGroupsU), () => {
        const idx = indicesIn.element(instanceIndex);
        const key = depthKeys.element(idx);

        const rawBucket = bitAnd(shiftRight(key, shiftU), maskU);
        const bucket = descending ? maxBucketU.sub(rawBucket) : rawBucket;

        const base = g.mul(uint(256));
        atomicAdd(groupHists.element(add(base, bucket)), uint(1));
      });
    });
  })()
    .compute(count, [256, 1, 1])
    .setName(
      `BlockRadixBuildGroupHists_shift${shift}${descending ? "_desc" : "_asc"}`
    );
}

function createBlockTotalsCompute({
  groupHists,
  totals,
  numGroups,
}: {
  groupHists: StorageBufferNode;
  totals: StorageBufferNode;
  numGroups: number;
}): ComputeNode {
  const numGroupsU = uint(numGroups);
  return Fn(() => {
    const b = invocationLocalIndex; // 0..255
    const sum = uint(0).toVar();

    const g = uint(0).toVar();
    Loop(g.lessThan(numGroupsU), () => {
      const base = g.mul(uint(256));
      sum.assign(add(sum, atomicLoad(groupHists.element(add(base, b)))));
      g.assign(add(g, uint(1)));
    });

    totals.element(b).assign(sum);
  })()
    .compute(256, [256, 1, 1])
    .setName("BlockRadixTotals256");
}

function createScan256ExclusiveCompute({
  input,
  output,
  name,
}: {
  input: StorageBufferNode;
  output: StorageBufferNode;
  name: string;
}): ComputeNode {
  return Fn(() => {
    const b = invocationLocalIndex; // 0..255
    const shared = workgroupArray("uint", 256);

    shared.element(b).assign(input.element(b));
    workgroupBarrier();

    const offset = uint(1).toVar();
    const t = uint(0).toVar();
    Loop(offset.lessThan(uint(256)), () => {
      t.assign(uint(0));
      If(b.greaterThanEqual(offset), () => {
        t.assign(shared.element(b.sub(offset)));
      });
      workgroupBarrier();
      shared.element(b).assign(add(shared.element(b), t));
      workgroupBarrier();
      offset.assign(add(offset, offset));
    });

    const exclusive = uint(0).toVar();
    If(b.greaterThan(uint(0)), () => {
      exclusive.assign(shared.element(b.sub(uint(1))));
    });
    output.element(b).assign(exclusive);
  })()
    .compute(256, [256, 1, 1])
    .setName(name);
}

function createBlockGroupBaseCompute({
  groupHists,
  bucketBase,
  groupBase,
  numGroups,
}: {
  groupHists: StorageBufferNode;
  bucketBase: StorageBufferNode;
  groupBase: StorageBufferNode;
  numGroups: number;
}): ComputeNode {
  const numGroupsU = uint(numGroups);
  return Fn(() => {
    const b = invocationLocalIndex; // 0..255
    const running = bucketBase.element(b).toVar();

    const g = uint(0).toVar();
    Loop(g.lessThan(numGroupsU), () => {
      const base = g.mul(uint(256));
      groupBase.element(add(base, b)).assign(running);
      running.assign(
        add(running, atomicLoad(groupHists.element(add(base, b))))
      );
      g.assign(add(g, uint(1)));
    });
  })()
    .compute(256, [256, 1, 1])
    .setName("BlockRadixGroupBase");
}

function createBlockScatterStableCompute({
  depthKeys,
  indicesIn,
  indicesOut,
  groupBase,
  count,
  numGroups,
  shift,
  descending,
}: {
  depthKeys: StorageBufferNode;
  indicesIn: StorageBufferNode;
  indicesOut: StorageBufferNode;
  groupBase: StorageBufferNode;
  count: number;
  numGroups: number;
  shift: number;
  descending: boolean;
}): ComputeNode {
  const shiftU = uint(shift);
  const maskU = uint(0xff);
  const maxBucketU = uint(255);
  const numGroupsU = uint(numGroups);

  return Fn(() => {
    const g = workgroupId.x;
    If(g.lessThan(numGroupsU), () => {
      // Only lane 0 does the serial, deterministic scatter for stability.
      If(invocationLocalIndex.equal(uint(0)), () => {
        const localCounts = workgroupArray("uint", 256);

        // init localCounts[b] = 0
        const b = uint(0).toVar();
        Loop(b.lessThan(uint(256)), () => {
          localCounts.element(b).assign(uint(0));
          b.assign(add(b, uint(1)));
        });

        const base = g.mul(uint(256));
        const j = uint(0).toVar();
        Loop(j.lessThan(uint(256)), () => {
          const globalIdx = add(base, j);
          If(globalIdx.lessThan(uint(count)), () => {
            const idx = indicesIn.element(globalIdx);
            const key = depthKeys.element(idx);
            const rawBucket = bitAnd(shiftRight(key, shiftU), maskU);
            const bucket = descending ? maxBucketU.sub(rawBucket) : rawBucket;

            const gb = groupBase.element(add(g.mul(uint(256)), bucket));
            const lc = localCounts.element(bucket);
            const pos = add(gb, lc);
            indicesOut.element(pos).assign(idx);
            localCounts.element(bucket).assign(add(lc, uint(1)));
          });
          j.assign(add(j, uint(1)));
        });
      });
    });
  })()
    .compute(count, [256, 1, 1])
    .setName(
      `BlockRadixScatterStable_shift${shift}${descending ? "_desc" : "_asc"}`
    );
}

export function createBlockRadixSortIndices({
  depthKeys,
  count,
  numGroups,
  indicesA,
  indicesB,
  groupHists,
  totals,
  bucketBase,
  groupBase,
  descending = true,
}: {
  depthKeys: StorageBufferNode;
  count: number;
  numGroups: number;
  indicesA: StorageBufferNode;
  indicesB: StorageBufferNode;
  groupHists: StorageBufferNode;
  totals: StorageBufferNode;
  bucketBase: StorageBufferNode;
  groupBase: StorageBufferNode;
  descending?: boolean;
}): BlockRadixSortIndices {
  const init = createInitIndicesCompute(indicesA, count);
  const groupHistsCount = numGroups * 256;

  const shifts = [0, 8, 16, 24] as const;
  const computes: ComputeNode[] = [init];

  for (let passIndex = 0; passIndex < shifts.length; passIndex++) {
    const shift = shifts[passIndex];
    const inBuf = passIndex % 2 === 0 ? indicesA : indicesB;
    const outBuf = passIndex % 2 === 0 ? indicesB : indicesA;

    computes.push(
      createBlockClearGroupHistsCompute(groupHists, groupHistsCount),
      createBlockBuildGroupHistsCompute({
        depthKeys,
        indicesIn: inBuf,
        groupHists,
        count,
        numGroups,
        shift,
        descending,
      }),
      createBlockTotalsCompute({ groupHists, totals, numGroups }),
      createScan256ExclusiveCompute({
        input: totals,
        output: bucketBase,
        name: "BlockRadixBucketBase256",
      }),
      createBlockGroupBaseCompute({
        groupHists,
        bucketBase,
        groupBase,
        numGroups,
      }),
      createBlockScatterStableCompute({
        depthKeys,
        indicesIn: inBuf,
        indicesOut: outBuf,
        groupBase,
        count,
        numGroups,
        shift,
        descending,
      })
    );
  }

  return {
    computes,
    sortedIndices: indicesA,
    buffers: { indicesA, indicesB, groupHists, totals, bucketBase, groupBase },
  };
}
