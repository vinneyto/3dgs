import { useEffect, useMemo } from "react";
import type { StorageBufferNode } from "three/webgpu";
import {
  DoubleSide,
  MeshBasicNodeMaterial,
  MeshStandardNodeMaterial,
} from "three/webgpu";
import type { InstancedEllipsoidPlyNodes } from "../tsl/gaussian/instancedEllipsoidPly";
import { createDepthDebugColorOpacity } from "../tsl/gaussian/depthDebugNodes";

export type PlyEllipsoidsMaterialParams = {
  shader: InstancedEllipsoidPlyNodes;
  useDepth: boolean;
  debugDepth: boolean;
  depthKeysBuf: StorageBufferNode | null;
  cutoff: number;
  roughness: number;
  metalness: number;
};

export function usePlyEllipsoidsMaterial({
  shader,
  useDepth,
  debugDepth,
  depthKeysBuf,
  cutoff,
  roughness,
  metalness,
}: PlyEllipsoidsMaterialParams):
  | MeshStandardNodeMaterial
  | MeshBasicNodeMaterial {
  const material = useMemo(() => {
    const isDebugDepth = debugDepth && depthKeysBuf;

    if (isDebugDepth) {
      // Debug view should be unlit, so colors match keys directly.
      const m = new MeshBasicNodeMaterial({ side: DoubleSide });
      m.depthTest = useDepth;
      m.depthWrite = useDepth;
      m.vertexNode = shader.nodes.vertexNode;
      const dbg = createDepthDebugColorOpacity(depthKeysBuf);
      m.colorNode = dbg.colorNode;
      m.opacityNode = dbg.opacityNode;
      return m;
    }

    const m = new MeshStandardNodeMaterial({ side: DoubleSide });
    m.depthTest = useDepth;
    m.depthWrite = useDepth;
    m.vertexNode = shader.nodes.vertexNode;
    m.normalNode = shader.nodes.normalNode;
    m.colorNode = shader.nodes.colorNode;
    m.opacityNode = shader.nodes.opacityNode;
    return m;
  }, [shader, useDepth, debugDepth, depthKeysBuf]);

  useEffect(() => {
    shader.uniforms.uCutoff.value = cutoff;
    if (material instanceof MeshStandardNodeMaterial) {
      material.roughness = roughness;
      material.metalness = metalness;
    }
  }, [shader, material, cutoff, roughness, metalness]);

  return material;
}
