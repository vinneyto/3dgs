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

export const RADIX_WORKGROUP_SIZE = 256;

export function createRadixInitIndicesCompute(
  indices: StorageBufferNode,
  count: number,
): ComputeNode {
  return Fn(() => {
    // three.js does not auto-guard out-of-range invocations when count is not a multiple of workgroup size.
    If(instanceIndex.lessThan(uint(count)), () => {
      indices.element(instanceIndex).assign(instanceIndex);
    });
  })()
    .compute(count, [RADIX_WORKGROUP_SIZE, 1, 1])
    .setName("RadixInitIndices");
}

/**
 * Initializes `indices[i]=i` for `i < activeCount[0]`, dispatching `maxCount` threads.
 *
 * This is used for pipelines where the number of active elements is computed on-GPU.
 */
export function createRadixInitIndicesActiveCountCompute({
  indices,
  activeCount,
  maxCount,
  name = "RadixActive_InitIndices",
}: {
  indices: StorageBufferNode;
  activeCount: StorageBufferNode; // u32[1]
  maxCount: number;
  name?: string;
}): ComputeNode {
  const maxCountU = uint(maxCount);
  return Fn(() => {
    If(instanceIndex.lessThan(maxCountU), () => {
      const n = activeCount.element(uint(0));
      If(instanceIndex.lessThan(n), () => {
        indices.element(instanceIndex).assign(instanceIndex);
      });
    });
  })()
    .compute(maxCount, [RADIX_WORKGROUP_SIZE, 1, 1])
    .setName(name);
}

// ----------------------------
// Deterministic Block-Radix Sort
// ----------------------------

export function createBlockClearGroupHistsCompute(
  groupHists: StorageBufferNode,
  groupHistsCount: number,
): ComputeNode {
  return Fn(() => {
    atomicStore(groupHists.element(instanceIndex), uint(0));
  })()
    .compute(groupHistsCount, [RADIX_WORKGROUP_SIZE, 1, 1])
    .setName("BlockRadixClearGroupHists");
}

export function createBlockBuildGroupHistsCompute({
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
    .compute(count, [RADIX_WORKGROUP_SIZE, 1, 1])
    .setName(
      `BlockRadixBuildGroupHists_shift${shift}${descending ? "_desc" : "_asc"}`,
    );
}

/**
 * Same as `createBlockBuildGroupHistsCompute`, but guards work by `activeCount[0]`
 * while still dispatching `maxCount` threads.
 */
export function createBlockBuildGroupHistsActiveCountCompute({
  depthKeys,
  indicesIn,
  groupHists,
  activeCount,
  maxCount,
  numGroups,
  shift,
  descending,
}: {
  depthKeys: StorageBufferNode;
  indicesIn: StorageBufferNode;
  groupHists: StorageBufferNode;
  activeCount: StorageBufferNode; // u32[1]
  maxCount: number;
  numGroups: number;
  shift: number;
  descending: boolean;
}): ComputeNode {
  const shiftU = uint(shift);
  const maskU = uint(0xff);
  const maxBucketU = uint(255);
  const numGroupsU = uint(numGroups);
  const maxCountU = uint(maxCount);

  return Fn(() => {
    If(instanceIndex.lessThan(maxCountU), () => {
      const n = activeCount.element(uint(0));
      If(instanceIndex.lessThan(n), () => {
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
    });
  })()
    .compute(maxCount, [RADIX_WORKGROUP_SIZE, 1, 1])
    .setName(
      `BlockRadixBuildGroupHistsActive_shift${shift}${
        descending ? "_desc" : "_asc"
      }`,
    );
}

export function createBlockTotalsCompute({
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
    .compute(256, [RADIX_WORKGROUP_SIZE, 1, 1])
    .setName("BlockRadixTotals256");
}

export function createScan256ExclusiveCompute({
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
    .compute(256, [RADIX_WORKGROUP_SIZE, 1, 1])
    .setName(name);
}

export function createBlockGroupBaseCompute({
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
        add(running, atomicLoad(groupHists.element(add(base, b)))),
      );
      g.assign(add(g, uint(1)));
    });
  })()
    .compute(256, [RADIX_WORKGROUP_SIZE, 1, 1])
    .setName("BlockRadixGroupBase");
}

export function createBlockScatterStableCompute({
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
    .compute(count, [RADIX_WORKGROUP_SIZE, 1, 1])
    .setName(
      `BlockRadixScatterStable_shift${shift}${descending ? "_desc" : "_asc"}`,
    );
}

/**
 * Same as `createBlockScatterStableCompute`, but uses `activeCount[0]` as the effective
 * element count while dispatching `maxCount` threads.
 */
export function createBlockScatterStableActiveCountCompute({
  depthKeys,
  indicesIn,
  indicesOut,
  groupBase,
  activeCount,
  maxCount,
  numGroups,
  shift,
  descending,
}: {
  depthKeys: StorageBufferNode;
  indicesIn: StorageBufferNode;
  indicesOut: StorageBufferNode;
  groupBase: StorageBufferNode;
  activeCount: StorageBufferNode; // u32[1]
  maxCount: number;
  numGroups: number;
  shift: number;
  descending: boolean;
}): ComputeNode {
  const shiftU = uint(shift);
  const maskU = uint(0xff);
  const maxBucketU = uint(255);
  const numGroupsU = uint(numGroups);
  const maxCountU = uint(maxCount);

  return Fn(() => {
    If(instanceIndex.lessThan(maxCountU), () => {
      const g = workgroupId.x;
      If(g.lessThan(numGroupsU), () => {
        // Only lane 0 does the serial, deterministic scatter for stability.
        If(invocationLocalIndex.equal(uint(0)), () => {
          const active = activeCount.element(uint(0));
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
            If(globalIdx.lessThan(active), () => {
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
    });
  })()
    .compute(maxCount, [RADIX_WORKGROUP_SIZE, 1, 1])
    .setName(
      `BlockRadixScatterStableActive_shift${shift}${
        descending ? "_desc" : "_asc"
      }`,
    );
}
