import type { Node, StorageBufferNode } from "three/webgpu";
import { Vector3 } from "three/webgpu";
import {
  cameraProjectionMatrix,
  cameraViewMatrix,
  float,
  instanceIndex,
  positionLocal,
  uniform,
  vec2,
  vec3,
} from "three/tsl";
import {
  splatQuadFragmentNodeFn,
  splatQuadVertexNodeFn,
  sqrtCutoff,
  unpackCovariance3D,
} from "./gaussianCommon";

export type InstancedSplatQuadNodes = {
  nodes: {
    vertexNode: Node;
    colorNode: Node;
    opacityNode: Node;
  };
  uniforms: {
    /** Global cutoff (same meaning as in GaussianSplats3D: default 8). */
    uCutoff: ReturnType<typeof uniform<number>>;
    /** Screen-space scale multiplier for splats (default 1). */
    uSplatScale: ReturnType<typeof uniform<number>>;
    /**
     * Packed debug params:
     * x = opacityMultiplier
     * y = showQuadBg (0/1)
     * z = quadBgAlpha
     */
    uParams: ReturnType<typeof uniform<Vector3>>;
  };
  buffers: {
    splats: StorageBufferNode;
  };
};

export function createInstancedSplatQuadNodes(
  splats: StorageBufferNode
): InstancedSplatQuadNodes {
  const uCutoff = uniform(8.0).setName("uCutoff");
  const uSplatScale = uniform(1.0).setName("uSplatScale");
  const uParams = uniform(new Vector3(1.0, 1.0, 0.12)).setName("uParams");

  const s = splats.element(instanceIndex);

  const center4 = s.get("center");
  const covA4 = s.get("covA");
  const covB4 = s.get("covB");
  const colorOpacity4 = s.get("colorOpacity");

  const centerWorld = vec3(center4.x, center4.y, center4.z);
  const Vrk = unpackCovariance3D({ covA: covA4, covB: covB4 });

  const { cutoff, radius } = sqrtCutoff(uCutoff);

  // quad corners from geometry [-1..1]
  const corner = vec2(positionLocal.x, positionLocal.y);
  const vPosition = corner.mul(radius).toVarying("vPosition");

  const posNDC = splatQuadVertexNodeFn({
    centerWorld,
    Vrk,
    vPosition,
    splatScale: uSplatScale,
    cameraViewMatrixNode: cameraViewMatrix,
    cameraProjectionMatrixNode: cameraProjectionMatrix,
  });

  const instanceOpacity = float(colorOpacity4.w);
  const opacityMultiplier = float(uParams.x);
  const showQuadBg = float(uParams.y);
  const quadBgAlpha = float(uParams.z);

  const instanceColor = vec3(colorOpacity4.x, colorOpacity4.y, colorOpacity4.z);

  const rgba = splatQuadFragmentNodeFn({
    instanceColor,
    instanceOpacity,
    opacityMultiplier,
    showQuadBg,
    quadBgAlpha,
    vPosition,
    cutoff,
  });

  const vertexNode = posNDC;
  const colorNode = rgba.xyz;
  const opacityNode = rgba.w;

  return {
    nodes: { vertexNode, colorNode, opacityNode },
    uniforms: { uCutoff, uSplatScale, uParams },
    buffers: { splats },
  };
}
