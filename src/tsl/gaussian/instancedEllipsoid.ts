import type { Node, StorageBufferNode } from "three/webgpu";
import {
  cameraProjectionMatrix,
  cameraViewMatrix,
  instanceIndex,
  modelNormalMatrix,
  modelViewMatrix,
  normalLocal,
  positionLocal,
  uniform,
  vec3,
  vec4,
} from "three/tsl";
import { cholesky3DFromCov, sqrtCutoff } from "./covarianceMath";

export type InstancedEllipsoidNodes = {
  nodes: {
    vertexNode: Node;
    normalNode: Node;
  };
  uniforms: {
    /** Iso-surface cutoff (shared for all instances). */
    uCutoff: ReturnType<typeof uniform<number>>;
  };
  buffers: {
    /** Storage buffer with SplatInstanceStruct layout. */
    splats: StorageBufferNode;
  };
};

export function createInstancedEllipsoidNodes(
  splats: StorageBufferNode
): InstancedEllipsoidNodes {
  const uCutoff = uniform(1.0).setName("uCutoff");
  const { radius } = sqrtCutoff(uCutoff);

  const s = splats.element(instanceIndex);

  // Per-instance data (vec4 packing)
  const center4 = s.get("center");
  const covA4 = s.get("covA");
  const covB4 = s.get("covB");
  const center = vec3(center4.x, center4.y, center4.z);

  const { L, invLT } = cholesky3DFromCov({
    covA: covA4,
    covB: covB4,
  });

  // Deform local sphere position into ellipsoid iso-surface:
  // p' = center + L * (positionLocal * sqrt(cutoff))
  const p = vec3(positionLocal.x, positionLocal.y, positionLocal.z).mul(radius);
  const localPos = L.mul(p).add(center);

  const vertexNode = cameraProjectionMatrix
    .mul(modelViewMatrix)
    .mul(vec4(localPos, 1.0));

  // `MeshStandardNodeMaterial.normalNode` is expected to be in view-space.
  const normalLocalEllipsoid = invLT
    .mul(vec3(normalLocal.x, normalLocal.y, normalLocal.z))
    .normalize();
  const normalNode = cameraViewMatrix
    .transformDirection(modelNormalMatrix.mul(normalLocalEllipsoid))
    .normalize();

  return {
    nodes: { vertexNode, normalNode },
    uniforms: { uCutoff },
    buffers: { splats },
  };
}
