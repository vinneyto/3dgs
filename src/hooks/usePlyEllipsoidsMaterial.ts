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
  alphaDiscard: number;
  roughness: number;
  metalness: number;
};

export function usePlyEllipsoidsMaterial({
  shader,
  useDepth,
  debugDepth,
  depthKeysBuf,
  cutoff,
  alphaDiscard,
  roughness,
  metalness,
}: PlyEllipsoidsMaterialParams):
  | MeshStandardNodeMaterial
  | MeshBasicNodeMaterial {
  const material = useMemo(() => {
    const isDebugDepth = debugDepth && depthKeysBuf;
    const isSorted = !!shader.buffers.sortedIndices;
    const enableDepth = useDepth && !isSorted;

    if (isDebugDepth) {
      // Debug view should be unlit, so colors match keys directly.
      const m = new MeshBasicNodeMaterial({ side: DoubleSide });
      // When sorting is enabled, disable depth buffer usage entirely to avoid Z-fighting/popping
      // against other scene geometry (grid, etc) and between translucent instances.
      m.depthTest = enableDepth;
      m.depthWrite = enableDepth;
      m.vertexNode = shader.nodes.vertexNode;
      const dbg = createDepthDebugColorOpacity(
        depthKeysBuf,
        shader.buffers.sortedIndices
      );
      m.colorNode = dbg.colorNode;
      m.opacityNode = dbg.opacityNode;
      return m;
    }

    const m = new MeshStandardNodeMaterial({ side: DoubleSide });
    // Same policy as above: if sorted, do not use depth buffer.
    m.depthTest = enableDepth;
    m.depthWrite = enableDepth;
    m.vertexNode = shader.nodes.vertexNode;
    m.normalNode = shader.nodes.normalNode;
    m.colorNode = shader.nodes.colorNode;
    m.opacityNode = shader.nodes.opacityNode;
    return m;
  }, [shader, useDepth, debugDepth, depthKeysBuf]);

  useEffect(() => {
    shader.uniforms.uCutoff.value = cutoff;
    shader.uniforms.uAlphaDiscard.value = alphaDiscard;
    if (material instanceof MeshStandardNodeMaterial) {
      material.roughness = roughness;
      material.metalness = metalness;
    }
  }, [shader, material, cutoff, alphaDiscard, roughness, metalness]);

  return material;
}
