import type { Node } from "three/webgpu";
import { float, uniform, vec3 } from "three/tsl";
import { billboardVertexFromPosition, circleFromUV } from "../billboardCircle";

export type InstancedBillboardCircleNodes = {
  nodes: {
    vertexNode: Node;
    colorNode: Node;
    opacityNode: Node;
  };
  uniforms: {
    /** World-space radius (shared). Final pixel size is computed from camera perspective and clamped. */
    uRadiusWorld: ReturnType<typeof uniform<number>>;
    uMinSizePx: ReturnType<typeof uniform<number>>;
    uMaxSizePx: ReturnType<typeof uniform<number>>;
  };
  buffers: Record<string, never>;
};

/**
 * Instanced billboard circles from an instanced attribute/buffer.
 *
 * `centerWorld3` is expected to be a `vec3` with world-space position.
 * `color3` is expected to be a `vec3` with per-instance RGB.
 */
export function createInstancedBillboardCircleNodes({
  centerWorld3,
  color3,
  minSizePx = 4,
  maxSizePx = 48,
  radiusWorld = 0.06,
}: {
  centerWorld3: Node; // vec3
  color3: Node; // vec3
  minSizePx?: number;
  maxSizePx?: number;
  radiusWorld?: number;
}): InstancedBillboardCircleNodes {
  const uRadiusWorld = uniform(radiusWorld).setName("uRadiusWorld");
  const uMinSizePx = uniform(minSizePx).setName("uMinSizePx");
  const uMaxSizePx = uniform(maxSizePx).setName("uMaxSizePx");

  const vertexNode = billboardVertexFromPosition({
    centerWorld: vec3(centerWorld3),
    radiusWorld: float(uRadiusWorld),
    minSizePx: uMinSizePx,
    maxSizePx: uMaxSizePx,
  });

  const rgba = circleFromUV({ color: vec3(color3) });

  return {
    nodes: { vertexNode, colorNode: rgba.xyz, opacityNode: rgba.w },
    uniforms: { uRadiusWorld, uMinSizePx, uMaxSizePx },
    buffers: {},
  };
}
