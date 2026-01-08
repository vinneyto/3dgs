import type { Node, StorageBufferNode } from "three/webgpu";
import {
  add,
  bitAnd,
  cameraProjectionMatrix,
  cameraViewMatrix,
  div,
  Fn,
  float,
  instanceIndex,
  mul,
  modelNormalMatrix,
  modelViewMatrix,
  normalLocal,
  positionLocal,
  shiftRight,
  uniform,
  uint,
  vec3,
  vec4,
} from "three/tsl";
import { cholesky3DFromCov, sqrtCutoff } from "./covarianceMath";

const ellipsoidColorOpacityVec4Fn = Fn(
  ({ rgbaPacked, alphaDiscard }: { rgbaPacked: Node; alphaDiscard: Node }) => {
    const inv255 = 1.0 / 255.0;
    const rU = bitAnd(rgbaPacked, uint(0x000000ff));
    const gU = bitAnd(shiftRight(rgbaPacked, uint(8)), uint(0x000000ff));
    const bU = bitAnd(shiftRight(rgbaPacked, uint(16)), uint(0x000000ff));
    const aU = bitAnd(shiftRight(rgbaPacked, uint(24)), uint(0x000000ff));

    const opacity = div(float(aU), 255.0);

    // Discard very low-opacity fragments to avoid rendering huge nearly-transparent ellipsoids.
    float(opacity).lessThan(float(alphaDiscard)).discard();

    const color = vec3(
      mul(float(rU), inv255),
      mul(float(gU), inv255),
      mul(float(bU), inv255)
    );

    return vec4(color, opacity);
  }
);

export type InstancedEllipsoidPlyNodes = {
  nodes: {
    vertexNode: Node;
    normalNode: Node;
    colorNode: Node;
    opacityNode: Node;
  };
  uniforms: {
    /** Iso-surface cutoff (shared for all instances). */
    uCutoff: ReturnType<typeof uniform<number>>;
    /** Discard fragments if opacity is below this threshold. */
    uAlphaDiscard: ReturnType<typeof uniform<number>>;
  };
  buffers: {
    centers: StorageBufferNode;
    cov: StorageBufferNode;
    rgba: StorageBufferNode;
    sortedIndices?: StorageBufferNode | null;
  };
};

export function createInstancedEllipsoidPlyNodes(
  centers: StorageBufferNode,
  cov: StorageBufferNode,
  rgba: StorageBufferNode,
  sortedIndices?: StorageBufferNode | null
): InstancedEllipsoidPlyNodes {
  const uCutoff = uniform(1.0).setName("uCutoff");
  // 2/255: drop nearly-transparent splats that tend to show up as large dark ellipsoids.
  const uAlphaDiscard = uniform(2.0 / 255.0).setName("uAlphaDiscard");
  const { radius } = sqrtCutoff(uCutoff);

  const splatIndex = sortedIndices
    ? sortedIndices.element(instanceIndex)
    : instanceIndex;

  const center = centers.element(splatIndex);
  const rgbaPacked = rgba.element(splatIndex);

  // cov entries live at indices (2*i) and (2*i+1)
  const covBase = splatIndex.mul(2);
  const covA3 = cov.element(covBase);
  const covB3 = cov.element(add(covBase, 1));

  const { L, invLT } = cholesky3DFromCov({
    covA: covA3,
    covB: covB3,
  });

  // Deform local sphere position into ellipsoid iso-surface:
  // p' = center + L * (positionLocal * sqrt(cutoff))
  const p = vec3(positionLocal.x, positionLocal.y, positionLocal.z).mul(radius);
  const localPos = L.mul(p).add(center);

  // Use modelViewMatrix so mesh transforms (scale/rotation/translation) affect the result.
  const vertexNode = cameraProjectionMatrix
    .mul(modelViewMatrix)
    .mul(vec4(localPos, 1.0));

  // `MeshStandardNodeMaterial.normalNode` is expected to be in view-space.
  // We first compute the ellipsoid surface normal in local space, then transform it
  // through the object's normal matrix and finally into view space.
  const normalLocalEllipsoid = invLT
    .mul(vec3(normalLocal.x, normalLocal.y, normalLocal.z))
    .normalize();
  const normalNode = cameraViewMatrix
    .transformDirection(modelNormalMatrix.mul(normalLocalEllipsoid))
    .normalize();

  const colorOpacity4 = ellipsoidColorOpacityVec4Fn({
    rgbaPacked,
    alphaDiscard: uAlphaDiscard,
  });
  const colorNode = colorOpacity4.xyz;
  const opacityNode = colorOpacity4.w;

  return {
    nodes: { vertexNode, normalNode, colorNode, opacityNode },
    uniforms: { uCutoff, uAlphaDiscard },
    buffers: { centers, cov, rgba, sortedIndices: sortedIndices ?? null },
  };
}
