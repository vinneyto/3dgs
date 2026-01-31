import { useFrame } from "@react-three/fiber";
import { useMemo } from "react";
import type { StorageBufferNode } from "three/webgpu";
import { useWebGPU } from "./useWebGPU";
import { GPUSplatDepthSort } from "@/app/_shared/gpu/GPUSplatDepthSort";

export type SplatDepthSortHookResult = {
  /** Maps draw-instance -> original splat index (sorted). */
  sortedSplatIndices: StorageBufferNode | null;
};

/**
 * React hook wrapper around `GPUSplatDepthSort`.
 *
 * This assumes the input buffers are already compacted to `[0..visibleCount)` each frame.
 */
export function useSplatDepthSort({
  enabled,
  visibleIndices,
  visibleDepthKeys,
  visibleCount,
  maxCount,
  descending = true,
}: {
  enabled: boolean;
  visibleIndices: StorageBufferNode | null;
  visibleDepthKeys: StorageBufferNode | null;
  visibleCount: StorageBufferNode | null;
  maxCount: number;
  descending?: boolean;
}): SplatDepthSortHookResult {
  const gl = useWebGPU();

  const sorter = useMemo(() => {
    if (!visibleIndices || !visibleDepthKeys || !visibleCount) return null;
    return new GPUSplatDepthSort(gl, {
      visibleIndices,
      visibleDepthKeys,
      visibleCount,
      maxCount,
      descending,
    });
  }, [
    gl,
    visibleIndices,
    visibleDepthKeys,
    visibleCount,
    maxCount,
    descending,
  ]);

  useFrame(() => {
    if (!enabled) return;
    if (!sorter) return;
    sorter.dispatch();
  });

  return {
    sortedSplatIndices: sorter ? sorter.getOutputs().sortedSplatIndices : null,
  };
}
