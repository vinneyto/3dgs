import type { ComputeNode, StorageBufferNode } from "three/webgpu";
import { Matrix4 } from "three/webgpu";
import {
  Fn,
  bitXor,
  int,
  instanceIndex,
  mul,
  uniform,
  uint,
  vec4,
} from "three/tsl";

export type DepthKeyCompute = {
  compute: ComputeNode;
  uniforms: {
    /** modelViewMatrix = camera.matrixWorldInverse * mesh.matrixWorld */
    uModelViewMatrix: ReturnType<typeof uniform<Matrix4>>;
    /** camera projection matrix (camera.projectionMatrix) */
    uProjectionMatrix: ReturnType<typeof uniform<Matrix4>>;
  };
  buffers: {
    centers: StorageBufferNode;
    depthKeys: StorageBufferNode;
  };
};

/**
 * Computes a quantized depth key per instance and stores it in `depthKeys`.
 *
 * GaussianSplats3D-style key:
 * - compute clip-space Z without perspective divide: z = (P * MV * vec4(center, 1)).z
 * - quantize: di = int(z * 4096.0)
 * - store as sortable uint by flipping the sign bit (two's complement order -> unsigned order)
 */
export function createDepthKeyCompute(
  centers: StorageBufferNode,
  depthKeys: StorageBufferNode,
  count: number
): DepthKeyCompute {
  const uModelViewMatrix = uniform<Matrix4>(new Matrix4()).setName(
    "uDepthModelViewMatrix"
  );
  const uProjectionMatrix = uniform<Matrix4>(new Matrix4()).setName(
    "uDepthProjectionMatrix"
  );

  const compute: ComputeNode = Fn(() => {
    const center = centers.element(instanceIndex);

    const clipPos = uProjectionMatrix
      .mul(uModelViewMatrix)
      .mul(vec4(center, 1.0));

    // Quantize (z without divide) to integer bins.
    const di = int(mul(clipPos.z, 4096.0));

    // Make signed int sortable as uint (flip sign bit).
    const key = bitXor(uint(di), uint(0x80000000));
    depthKeys.element(instanceIndex).assign(key);
  })()
    .compute(count)
    .setName("ComputeDepthKeys");

  return {
    compute,
    uniforms: { uModelViewMatrix, uProjectionMatrix },
    buffers: { centers, depthKeys },
  };
}
