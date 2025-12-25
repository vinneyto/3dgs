import type { Node, StorageBufferNode } from "three/webgpu";
import {
  add,
  bitAnd,
  cameraProjectionMatrix,
  div,
  float,
  instanceIndex,
  mul,
  modelViewMatrix,
  normalLocal,
  positionLocal,
  shiftRight,
  uniform,
  uint,
  vec3,
  vec4,
} from "three/tsl";
import { cholesky3DFromCov, sqrtCutoff } from "./gaussianCommon";

function unpackRGBA8UintToColorOpacity(rgbaPacked: Node): {
  colorNode: Node;
  opacityNode: Node;
} {
  const inv255 = 1.0 / 255.0;
  const rU = bitAnd(rgbaPacked, uint(0x000000ff));
  const gU = bitAnd(shiftRight(rgbaPacked, uint(8)), uint(0x000000ff));
  const bU = bitAnd(shiftRight(rgbaPacked, uint(16)), uint(0x000000ff));
  const aU = bitAnd(shiftRight(rgbaPacked, uint(24)), uint(0x000000ff));

  const colorNode = vec3(
    mul(float(rU), inv255),
    mul(float(gU), inv255),
    mul(float(bU), inv255)
  );
  const opacityNode = div(float(aU), 255.0);

  return { colorNode, opacityNode };
}

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
  };
  buffers: {
    centers: StorageBufferNode;
    cov: StorageBufferNode;
    rgba: StorageBufferNode;
  };
};

export function createInstancedEllipsoidPlyNodes(
  centers: StorageBufferNode,
  cov: StorageBufferNode,
  rgba: StorageBufferNode
): InstancedEllipsoidPlyNodes {
  const uCutoff = uniform(1.0).setName("uCutoff");
  const { radius } = sqrtCutoff(uCutoff);

  const center = centers.element(instanceIndex);
  const rgbaPacked = rgba.element(instanceIndex);

  // cov entries live at indices (2*i) and (2*i+1)
  const covBase = instanceIndex.mul(2);
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

  const normalNode = invLT
    .mul(vec3(normalLocal.x, normalLocal.y, normalLocal.z))
    .normalize();

  const { colorNode, opacityNode } = unpackRGBA8UintToColorOpacity(rgbaPacked);

  return {
    nodes: { vertexNode, normalNode, colorNode, opacityNode },
    uniforms: { uCutoff },
    buffers: { centers, cov, rgba },
  };
}
