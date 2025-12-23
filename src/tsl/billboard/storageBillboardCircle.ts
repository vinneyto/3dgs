import type { Node, StorageBufferNode } from "three/webgpu";
import { float, instanceIndex, struct, uniform, vec3 } from "three/tsl";
import { billboardVertexFromPosition, circleFromUV } from "../billboardCircle";

export const BillboardParticleStruct = struct(
  {
    center: "vec4", // xyz = world center
    color: "vec4", // xyz = rgb
  },
  "BillboardParticle"
);

export type StorageBillboardCircleNodes = {
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
  buffers: {
    particles: StorageBufferNode;
  };
};

export function createStorageBillboardCircleNodes(
  particles: StorageBufferNode
): StorageBillboardCircleNodes {
  const uRadiusWorld = uniform(0.06).setName("uRadiusWorld");
  const uMinSizePx = uniform(3).setName("uMinSizePx");
  const uMaxSizePx = uniform(24).setName("uMaxSizePx");

  const p = particles.element(instanceIndex);
  const center4 = p.get("center");
  const color4 = p.get("color");

  const vertexNode = billboardVertexFromPosition({
    centerWorld: vec3(center4.x, center4.y, center4.z),
    radiusWorld: float(uRadiusWorld),
    minSizePx: uMinSizePx,
    maxSizePx: uMaxSizePx,
  });

  const rgba = circleFromUV({ color: vec3(color4.x, color4.y, color4.z) });

  return {
    nodes: { vertexNode, colorNode: rgba.xyz, opacityNode: rgba.w },
    uniforms: { uRadiusWorld, uMinSizePx, uMaxSizePx },
    buffers: { particles },
  };
}
