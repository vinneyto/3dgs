import { useEffect, useMemo } from "react";
import type { StorageBufferNode } from "three/webgpu";
import { DoubleSide, MeshBasicNodeMaterial } from "three/webgpu";
import type { InstancedSplatQuadPlyNodes } from "../tsl/gaussian/instancedSplatQuadPly";
import { createDepthDebugColorOpacity } from "../tsl/gaussian/depthDebugNodes";

export type PlySplatQuadsMaterialParams = {
  shader: InstancedSplatQuadPlyNodes;
  useDepth: boolean;
  debugDepth: boolean;
  depthKeysBuf: StorageBufferNode | null;
  splatScale: number;
  kernel2DSize: number;
  maxScreenSpaceSplatSize: number;
  antialiasCompensation: boolean;
  opacityMultiplier: number;
  showQuadBg: boolean;
  quadBgAlpha: number;
};

export function usePlySplatQuadsMaterial({
  shader,
  useDepth,
  debugDepth,
  depthKeysBuf,
  splatScale,
  kernel2DSize,
  maxScreenSpaceSplatSize,
  antialiasCompensation,
  opacityMultiplier,
  showQuadBg,
  quadBgAlpha,
}: PlySplatQuadsMaterialParams): MeshBasicNodeMaterial {
  const material = useMemo(() => {
    const isDebugDepth = debugDepth && depthKeysBuf;
    const isSorted = !!shader.buffers.sortedIndices;
    const enableDepth = useDepth && !isSorted;

    const m = new MeshBasicNodeMaterial({ side: DoubleSide });
    m.transparent = true;
    m.depthTest = enableDepth;
    m.depthWrite = enableDepth;
    m.vertexNode = shader.nodes.vertexNode;

    if (isDebugDepth) {
      const dbg = createDepthDebugColorOpacity(
        depthKeysBuf,
        shader.buffers.sortedIndices
      );
      m.colorNode = dbg.colorNode;
      m.opacityNode = dbg.opacityNode as never;
    } else {
      m.colorNode = shader.nodes.colorNode;
      m.opacityNode = shader.nodes.opacityNode as never;
    }

    return m;
  }, [shader, useDepth, debugDepth, depthKeysBuf]);

  useEffect(() => {
    shader.uniforms.uSplatScale.value = splatScale;
    shader.uniforms.uKernel2DSize.value = kernel2DSize;
    shader.uniforms.uMaxScreenSpaceSplatSize.value = maxScreenSpaceSplatSize;
    shader.uniforms.uAntialiasCompensation.value = antialiasCompensation
      ? 1.0
      : 0.0;
    shader.uniforms.uParams.value.set(
      opacityMultiplier,
      showQuadBg ? 1.0 : 0.0,
      quadBgAlpha
    );
  }, [
    shader,
    splatScale,
    kernel2DSize,
    maxScreenSpaceSplatSize,
    antialiasCompensation,
    opacityMultiplier,
    showQuadBg,
    quadBgAlpha,
  ]);

  return material;
}
