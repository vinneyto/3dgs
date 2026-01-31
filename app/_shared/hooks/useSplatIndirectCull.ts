import { useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo } from "react";
import type { BufferGeometry, InstancedMesh } from "three";
import { Matrix4 } from "three/webgpu";
import type { StorageBufferNode } from "three/webgpu";
import { useWebGPU } from "./useWebGPU";
import {
  GPUSplatIndirectCull,
  type SplatIndirectCullOutputs,
} from "@/app/_shared/gpu/GPUSplatIndirectCull";

export type SplatIndirectCullHookResult = {
  visibleIndices: StorageBufferNode | null;
  visibleDepthKeys: StorageBufferNode | null;
  visibleCount: StorageBufferNode | null;
  /** Provided for advanced callers; this hook already attaches it to geometry. */
  indirectAttribute: SplatIndirectCullOutputs["indirectAttribute"] | null;
};

/**
 * React hook wrapper around `GPUSplatIndirectCull`.
 *
 * - Owns the class lifecycle.
 * - Wires camera+mesh matrices each frame.
 * - Attaches the indirect buffer to the mesh geometry (WebGPU only).
 */
export function useSplatIndirectCull({
  enabled,
  centersBuf,
  count,
  meshRef,
}: {
  enabled: boolean;
  centersBuf: StorageBufferNode;
  count: number;
  meshRef: React.RefObject<InstancedMesh | null>;
}): SplatIndirectCullHookResult {
  const gl = useWebGPU();
  const camera = useThree((s) => s.camera);
  const tmpModelView = useMemo(() => new Matrix4(), []);
  const tmpProjection = useMemo(() => new Matrix4(), []);

  const cull = useMemo(() => {
    if (!centersBuf) return null;
    return new GPUSplatIndirectCull(gl, {
      centers: centersBuf,
      maxCount: count,
    });
  }, [gl, centersBuf, count]);

  useEffect(() => {
    if (!cull) return;
    const mesh = meshRef.current;
    if (!mesh) return;
    const geom = mesh.geometry as BufferGeometry | undefined;
    if (!geom) return;

    // Attach indirect args buffer so WebGPURenderer uses drawIndexedIndirect.
    const out = cull.getOutputs();
    geom.setIndirect(out.indirectAttribute);

    // Seed indirectArgs[0] = indexCount once geometry index exists.
    const indexCount = geom.index?.count ?? 0;
    cull.setIndirectIndexCount(indexCount);
  }, [cull, meshRef]);

  useFrame(() => {
    if (!enabled) return;
    if (!cull) return;
    const mesh = meshRef.current;
    if (!mesh) return;

    // modelView = view * model
    tmpModelView.multiplyMatrices(camera.matrixWorldInverse, mesh.matrixWorld);
    tmpProjection.copy(camera.projectionMatrix as unknown as Matrix4);
    cull.setMatrices(tmpModelView, tmpProjection);
    cull.dispatch();
  });

  const out = cull?.getOutputs() ?? null;
  return {
    visibleIndices: enabled && out ? out.visibleIndices : null,
    visibleDepthKeys: enabled && out ? out.visibleDepthKeys : null,
    visibleCount: enabled && out ? out.visibleCount : null,
    indirectAttribute: enabled && out ? out.indirectAttribute : null,
  };
}
