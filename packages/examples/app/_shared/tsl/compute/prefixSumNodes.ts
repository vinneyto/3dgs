import type { ComputeNode, StorageBufferNode } from "three/webgpu";
import {
  If,
  Loop,
  Fn,
  invocationLocalIndex,
  numWorkgroups,
  workgroupArray,
  workgroupBarrier,
  uint,
  workgroupId,
} from "three/tsl";

export const PREFIX_SUM_WORKGROUP_SIZE = 256;
export const PREFIX_SUM_BLOCK_SIZE = PREFIX_SUM_WORKGROUP_SIZE * 2; // 512

function ceilDiv(a: number, b: number): number {
  return Math.floor((a + b - 1) / b);
}

function invocationsForN(n: number): number {
  const blocks = Math.max(1, ceilDiv(n, PREFIX_SUM_BLOCK_SIZE));
  return blocks * PREFIX_SUM_WORKGROUP_SIZE;
}

/**
 * Per-block Blelloch exclusive scan (512 elems per workgroup):
 * - out = exclusive_scan(in) within each 512-sized block
 * - blockSums[blockId] = sum(in block)
 */
export function createScan512WriteSumsCompute({
  inData,
  outData,
  blockSums,
  n,
}: {
  inData: StorageBufferNode;
  outData: StorageBufferNode;
  blockSums: StorageBufferNode;
  n: number;
}): ComputeNode {
  const invocations = invocationsForN(n);

  return Fn(() => {
    const N = uint(n >>> 0);
    const WG = uint(PREFIX_SUM_WORKGROUP_SIZE);
    const BLOCK = uint(PREFIX_SUM_BLOCK_SIZE);
    const last = BLOCK.sub(uint(1));

    const tid = invocationLocalIndex; // 0..WG-1
    // Global workgroup id across 2D dispatch (three.js may spill into Y when X > maxComputeWorkgroupsPerDimension).
    const bid = workgroupId.x.add(workgroupId.y.mul(numWorkgroups.x));
    const nblocks = N.add(BLOCK.sub(uint(1))).div(BLOCK);
    const base = bid.mul(BLOCK);
    const i0 = base.add(tid);
    const i1 = base.add(tid).add(WG);

    const temp = workgroupArray("uint", PREFIX_SUM_BLOCK_SIZE);

    // Guard for the padded workgroups in the last Y row.
    If(bid.lessThan(nblocks), () => {
      // load (guard)
      temp.element(tid).assign(uint(0));
      temp.element(tid.add(WG)).assign(uint(0));
      If(i0.lessThan(N), () => {
        temp.element(tid).assign(inData.element(i0));
      });
      If(i1.lessThan(N), () => {
        temp.element(tid.add(WG)).assign(inData.element(i1));
      });

      workgroupBarrier();

      // upsweep (reduce)
      const offset = uint(1).toVar();
      Loop(offset.lessThan(BLOCK), () => {
        const idx = tid.add(uint(1)).mul(offset).mul(uint(2)).sub(uint(1));
        If(idx.lessThan(BLOCK), () => {
          temp
            .element(idx)
            .assign(temp.element(idx).add(temp.element(idx.sub(offset))));
        });
        workgroupBarrier();
        offset.assign(offset.mul(uint(2)));
      });

      // save sum + exclusive root
      If(tid.equal(uint(0)), () => {
        blockSums.element(bid).assign(temp.element(last));
        temp.element(last).assign(uint(0));
      });
      workgroupBarrier();

      // downsweep
      offset.assign(BLOCK.div(uint(2)));
      const t = uint(0).toVar();
      Loop(offset.greaterThan(uint(0)), () => {
        const idx = tid.add(uint(1)).mul(offset).mul(uint(2)).sub(uint(1));
        If(idx.lessThan(BLOCK), () => {
          const left = idx.sub(offset);
          t.assign(temp.element(left));
          temp.element(left).assign(temp.element(idx));
          temp.element(idx).assign(temp.element(idx).add(t));
        });
        workgroupBarrier();
        offset.assign(offset.div(uint(2)));
      });

      // store (guard)
      If(i0.lessThan(N), () => {
        outData.element(i0).assign(temp.element(tid));
      });
      If(i1.lessThan(N), () => {
        outData.element(i1).assign(temp.element(tid.add(WG)));
      });
    });
  })()
    .compute(invocations, [PREFIX_SUM_WORKGROUP_SIZE, 1, 1])
    .setName("PrefixSum_scan512_writeSums");
}

/**
 * Same scan as `createScan512WriteSumsCompute`, but adds precomputed per-block offsets:
 * out[i] = exclusive_scan(in)[i] + blockOffsets[blockId]
 */
export function createScan512AddOffsetsCompute({
  inData,
  outData,
  blockOffsets,
  n,
}: {
  inData: StorageBufferNode;
  outData: StorageBufferNode;
  blockOffsets: StorageBufferNode;
  n: number;
}): ComputeNode {
  const invocations = invocationsForN(n);

  return Fn(() => {
    const N = uint(n >>> 0);
    const WG = uint(PREFIX_SUM_WORKGROUP_SIZE);
    const BLOCK = uint(PREFIX_SUM_BLOCK_SIZE);
    const last = BLOCK.sub(uint(1));

    const tid = invocationLocalIndex; // 0..WG-1
    const bid = workgroupId.x.add(workgroupId.y.mul(numWorkgroups.x));
    const nblocks = N.add(BLOCK.sub(uint(1))).div(BLOCK);
    const base = bid.mul(BLOCK);
    const i0 = base.add(tid);
    const i1 = base.add(tid).add(WG);

    const temp = workgroupArray("uint", PREFIX_SUM_BLOCK_SIZE);

    If(bid.lessThan(nblocks), () => {
      // load (guard)
      temp.element(tid).assign(uint(0));
      temp.element(tid.add(WG)).assign(uint(0));
      If(i0.lessThan(N), () => {
        temp.element(tid).assign(inData.element(i0));
      });
      If(i1.lessThan(N), () => {
        temp.element(tid.add(WG)).assign(inData.element(i1));
      });

      workgroupBarrier();

      // upsweep
      const offset = uint(1).toVar();
      Loop(offset.lessThan(BLOCK), () => {
        const idx = tid.add(uint(1)).mul(offset).mul(uint(2)).sub(uint(1));
        If(idx.lessThan(BLOCK), () => {
          temp
            .element(idx)
            .assign(temp.element(idx).add(temp.element(idx.sub(offset))));
        });
        workgroupBarrier();
        offset.assign(offset.mul(uint(2)));
      });

      // exclusive root
      If(tid.equal(uint(0)), () => {
        temp.element(last).assign(uint(0));
      });
      workgroupBarrier();

      // downsweep
      offset.assign(BLOCK.div(uint(2)));
      const t = uint(0).toVar();
      Loop(offset.greaterThan(uint(0)), () => {
        const idx = tid.add(uint(1)).mul(offset).mul(uint(2)).sub(uint(1));
        If(idx.lessThan(BLOCK), () => {
          const left = idx.sub(offset);
          t.assign(temp.element(left));
          temp.element(left).assign(temp.element(idx));
          temp.element(idx).assign(temp.element(idx).add(t));
        });
        workgroupBarrier();
        offset.assign(offset.div(uint(2)));
      });

      // add offsets + store
      const off = blockOffsets.element(bid).toVar();
      If(i0.lessThan(N), () => {
        outData.element(i0).assign(temp.element(tid).add(off));
      });
      If(i1.lessThan(N), () => {
        outData.element(i1).assign(temp.element(tid.add(WG)).add(off));
      });
    });
  })()
    .compute(invocations, [PREFIX_SUM_WORKGROUP_SIZE, 1, 1])
    .setName("PrefixSum_scan512_addOffsets");
}

/**
 * Adds per-block offsets in-place:
 * outData[i] += blockOffsets[block(i)].
 */
export function createAddBlockOffsetsCompute({
  outData,
  blockOffsets,
  n,
}: {
  outData: StorageBufferNode;
  blockOffsets: StorageBufferNode;
  n: number;
}): ComputeNode {
  const invocations = invocationsForN(n);

  return Fn(() => {
    const N = uint(n >>> 0);
    const WG = uint(PREFIX_SUM_WORKGROUP_SIZE);
    const BLOCK = uint(PREFIX_SUM_BLOCK_SIZE);

    const tid = invocationLocalIndex;
    const bid = workgroupId.x.add(workgroupId.y.mul(numWorkgroups.x));
    const nblocks = N.add(BLOCK.sub(uint(1))).div(BLOCK);

    If(bid.lessThan(nblocks), () => {
      const base = bid.mul(BLOCK);
      const i0 = base.add(tid);
      const i1 = base.add(tid).add(WG);
      const off = blockOffsets.element(bid).toVar();

      If(i0.lessThan(N), () => {
        outData.element(i0).assign(outData.element(i0).add(off));
      });
      If(i1.lessThan(N), () => {
        outData.element(i1).assign(outData.element(i1).add(off));
      });
    });
  })()
    .compute(invocations, [PREFIX_SUM_WORKGROUP_SIZE, 1, 1])
    .setName("PrefixSum_addBlockOffsets");
}
