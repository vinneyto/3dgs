import type {
  ComputeNode,
  StorageBufferNode,
  WebGPURenderer,
} from "three/webgpu";
import { IndirectStorageBufferAttribute, Matrix4 } from "three/webgpu";
import {
  Fn,
  If,
  add,
  bitXor,
  float,
  instanceIndex,
  int,
  mul,
  storage,
  uint,
  uniform,
  vec4,
} from "three/tsl";
import { instancedArray } from "three/tsl";
import { GPUPrefixSum } from "@/app/_shared/gpu/GPUPrefixSum";

export type SplatIndirectCullOutputs = {
  /** Maps draw-instance -> original splat index (compacted, unsorted). */
  visibleIndices: StorageBufferNode;
  /** Depth keys for visible splats, dense in [0..visibleCount). */
  visibleDepthKeys: StorageBufferNode;
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

      flags.element(instanceIndex).assign(inside.select(uint(1), uint(0)));
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
  indirectArgs: StorageBufferNode; // u32[5]
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

/**
 * GPU-driven frustum culling + prefix-sum compaction + indirect instanceCount.
 *
 * This class intentionally does NOT sort. Sorting can be layered on top.
 */
export class GPUSplatIndirectCull {
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

  // Indirect draw args
  private indirectAttr: IndirectStorageBufferAttribute;
  private indirectArgs: StorageBufferNode;

  // Compute passes
  private computeFlags: ComputeNode;
  private prefixSum: GPUPrefixSum;
  private computeCompact: ComputeNode;
  private computeVisibleCountAndIndirect: ComputeNode;
  private computeVisibleDepthKeys: ComputeNode;

  constructor(
    gl: WebGPURenderer,
    params: { centers: StorageBufferNode; maxCount: number },
  ) {
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

    this.computeVisibleCountAndIndirect =
      createVisibleCountAndIndirectArgsCompute({
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

  /** Runs: flags → scan → compact → visibleCount+indirect → visibleDepthKeys. */
  dispatch(): void {
    if (this.maxCount <= 0) return;

    this.gl.compute(this.computeFlags);
    this.prefixSum.compute();
    this.gl.compute(this.computeCompact);
    this.gl.compute(this.computeVisibleCountAndIndirect);
    this.gl.compute(this.computeVisibleDepthKeys);
  }

  getOutputs(): SplatIndirectCullOutputs {
    return {
      visibleIndices: this.visibleIndices,
      visibleDepthKeys: this.visibleDepthKeys,
      visibleCount: this.visibleCount,
      indirectAttribute: this.indirectAttr,
    };
  }
}
