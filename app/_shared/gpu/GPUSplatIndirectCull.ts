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
  dot,
  float,
  instanceIndex,
  int,
  max,
  mul,
  storage,
  sub,
  sqrt,
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
  /**
   * Distance-based LOD (world units; assumes your scene units are meters).
   * If enabled, far-away splats need to be "big enough" (based on covariance) to be kept.
   */
  uLodEnabled: ReturnType<typeof uniform<number>>;
  uLodStartDistanceM: ReturnType<typeof uniform<number>>;
  uLodBaseMinRadiusM: ReturnType<typeof uniform<number>>;
  uLodMinRadiusSlope: ReturnType<typeof uniform<number>>;
};

// Distance-based LOD defaults (tweak as you like).
// Idea: after `LOD_START_DISTANCE_M`, progressively drop splats whose world-space "radius"
// (approximated from covariance diagonal) is too small.
const LOD_ENABLED = 1.0;
const LOD_START_DISTANCE_M = 6.0;
const LOD_BASE_MIN_RADIUS_M = 0.02;
const LOD_MIN_RADIUS_SLOPE = 0.001;

function createSharedMatricesUniforms(): UniformMatrices {
  const uModelViewMatrix = uniform<Matrix4>(new Matrix4()).setName(
    "uCullModelViewMatrix",
  );
  const uProjectionMatrix = uniform<Matrix4>(new Matrix4()).setName(
    "uCullProjectionMatrix",
  );
  const uLodEnabled = uniform(LOD_ENABLED).setName("uCullLodEnabled");
  const uLodStartDistanceM = uniform(LOD_START_DISTANCE_M).setName(
    "uCullLodStartDistanceM",
  );
  const uLodBaseMinRadiusM = uniform(LOD_BASE_MIN_RADIUS_M).setName(
    "uCullLodBaseMinRadiusM",
  );
  const uLodMinRadiusSlope = uniform(LOD_MIN_RADIUS_SLOPE).setName(
    "uCullLodMinRadiusSlope",
  );

  return {
    uModelViewMatrix,
    uProjectionMatrix,
    uLodEnabled,
    uLodStartDistanceM,
    uLodBaseMinRadiusM,
    uLodMinRadiusSlope,
  };
}

function createFrustumFlagsCompute({
  centers,
  cov,
  flags,
  maxCount,
  uniforms,
}: {
  centers: StorageBufferNode;
  // 2 vec3 entries per splat (covariance packed as 6 floats: [xx,xy,xz, yy,yz,zz]).
  cov: StorageBufferNode;
  flags: StorageBufferNode;
  maxCount: number;
  uniforms: UniformMatrices;
}): ComputeNode {
  const maxCountU = uint(maxCount);
  const zeroF = float(0.0);
  const trueB = float(1.0).greaterThan(zeroF);

  return Fn(() => {
    If(instanceIndex.lessThan(maxCountU), () => {
      const center = centers.element(instanceIndex);
      const viewPos = uniforms.uModelViewMatrix.mul(vec4(center, 1.0));
      const clipPos = uniforms.uProjectionMatrix.mul(viewPos);

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

      // Extra distance-based LOD: far away splats must be "big enough".
      // Approximate world-space radius from covariance diagonal: radius ~= sqrt(max(xx,yy,zz)).
      const i2 = mul(instanceIndex, uint(2));
      const cov0 = cov.element(i2);
      const cov1 = cov.element(add(i2, uint(1)));
      const diagMax = max(cov0.x, max(cov1.x, cov1.z));
      const radiusM = sqrt(max(diagMax, float(1e-8)));

      // Distance from camera in view space (same units as center/cov).
      const distM = sqrt(dot(viewPos.xyz, viewPos.xyz));

      const enabled = float(uniforms.uLodEnabled).greaterThan(0.5);
      const pastStart = distM.greaterThan(float(uniforms.uLodStartDistanceM));

      // minRadius(dist) = base + slope * max(dist - start, 0)
      const minRadiusM = add(
        float(uniforms.uLodBaseMinRadiusM),
        mul(
          float(uniforms.uLodMinRadiusSlope),
          max(sub(distM, float(uniforms.uLodStartDistanceM)), zeroF),
        ),
      );
      const bigEnough = radiusM.greaterThanEqual(minRadiusM);
      const lodOk = pastStart.select(bigEnough, trueB);
      const keep = inside.and(enabled.select(lodOk, trueB));

      flags.element(instanceIndex).assign(keep.select(uint(1), uint(0)));
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
  private cov: StorageBufferNode;

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
    params: { centers: StorageBufferNode; cov: StorageBufferNode; maxCount: number },
  ) {
    this.gl = gl;
    this.centers = params.centers;
    this.cov = params.cov;
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
      cov: this.cov,
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

  /**
   * Configure distance-based LOD in "meters" (scene units).
   *
   * - startDistanceM: after this, small splats will be dropped more aggressively.
   * - baseMinRadiusM: minimum world-space radius at startDistanceM.
   * - minRadiusSlope: linear increase of min radius per extra meter.
   */
  setDistanceLod(params: {
    enabled?: boolean;
    startDistanceM?: number;
    baseMinRadiusM?: number;
    minRadiusSlope?: number;
  }): void {
    if (params.enabled !== undefined) {
      this.uniforms.uLodEnabled.value = params.enabled ? 1.0 : 0.0;
    }
    if (params.startDistanceM !== undefined) {
      this.uniforms.uLodStartDistanceM.value = params.startDistanceM;
    }
    if (params.baseMinRadiusM !== undefined) {
      this.uniforms.uLodBaseMinRadiusM.value = params.baseMinRadiusM;
    }
    if (params.minRadiusSlope !== undefined) {
      this.uniforms.uLodMinRadiusSlope.value = params.minRadiusSlope;
    }
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
