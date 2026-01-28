import { useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo } from "react";
import type { InstancedMesh } from "three";
import type { BufferGeometry } from "three";
import { Matrix4 } from "three/webgpu";
import type { StorageBufferNode } from "three/webgpu";
import { useWebGPU } from "./useWebGPU";
import { GPUSplatIndirectCullSort } from "@/app/_shared/gpu/GPUSplatIndirectCullSort";

export type SplatIndirectCullSortHookResult = {
  sortedSplatIndices: StorageBufferNode | null;
  visibleCount: StorageBufferNode | null;
};

/**
 * React hook wrapper around `GPUSplatIndirectCullSort`.
 *
 * - Owns the class lifecycle.
 * - Wires camera+mesh matrices each frame.
 * - Attaches the indirect buffer to the mesh geometry (WebGPU only).
 */
export function useSplatIndirectCullSort({
  enabled,
  centersBuf,
  count,
  meshRef,
}: {
  enabled: boolean;
  centersBuf: StorageBufferNode;
  count: number;
  meshRef: React.RefObject<InstancedMesh | null>;
}): SplatIndirectCullSortHookResult {
  const gl = useWebGPU();
  const camera = useThree((s) => s.camera);
  const tmpModelView = useMemo(() => new Matrix4(), []);
  const tmpProjection = useMemo(() => new Matrix4(), []);

  const pipeline = useMemo(() => {
    if (!centersBuf) return null;
    return new GPUSplatIndirectCullSort(gl, { centers: centersBuf, maxCount: count });
  }, [gl, centersBuf, count]);

  useEffect(() => {
    if (!pipeline) return;
    const mesh = meshRef.current;
    if (!mesh) return;
    const geom = mesh.geometry as BufferGeometry | undefined;
    if (!geom) return;

    // Attach indirect args buffer so WebGPURenderer uses drawIndexedIndirect.
    geom.setIndirect(pipeline.getOutputs().indirectAttribute);

    // Seed indirectArgs[0] = indexCount once geometry index exists.
    const indexCount = geom.index?.count ?? 0;
    pipeline.setIndirectIndexCount(indexCount);
  }, [pipeline, meshRef]);

  useFrame(() => {
    if (!enabled) return;
    if (!pipeline) return;
    const mesh = meshRef.current;
    if (!mesh) return;

    // modelView = view * model
    tmpModelView.multiplyMatrices(camera.matrixWorldInverse, mesh.matrixWorld);
    tmpProjection.copy(camera.projectionMatrix as unknown as Matrix4);
    pipeline.setMatrices(tmpModelView, tmpProjection);
    pipeline.dispatch();
  });

  const outputs = pipeline?.getOutputs() ?? null;
  return {
    sortedSplatIndices: enabled && outputs ? outputs.sortedSplatIndices : null,
    visibleCount: enabled && outputs ? outputs.visibleCount : null,
  };
}

