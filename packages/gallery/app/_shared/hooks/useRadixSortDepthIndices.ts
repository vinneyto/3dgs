import { useFrame } from "@react-three/fiber";
import { useMemo } from "react";
import type { StorageBufferNode } from "three/webgpu";
import { useWebGPU } from "./useWebGPU";
import { GPURadixSort } from "@/app/_shared/gpu/GPURadixSort";

/**
 * Runs a GPU radix sort each frame to produce `sortedIndices`, ordering instances by depth.
 *
 * The output buffer can be consumed by the render shader to fetch splat/ellipsoid data as:
 * `splatIndex = sortedIndices[ instanceIndex ]`.
 */
export function useRadixSortDepthIndices({
  enabled,
  depthKeysBuf,
  count,
  descending = true,
}: {
  enabled: boolean;
  depthKeysBuf: StorageBufferNode | null;
  count: number;
  /** If true: far->near (back-to-front). */
  descending?: boolean;
}): StorageBufferNode | null {
  const gl = useWebGPU();
  const sorter = useMemo(() => new GPURadixSort(gl), [gl]);

  const sortedIndicesBuf = useMemo(() => {
    sorter
      .setDepthKeysBuffer(depthKeysBuf)
      .setCount(count)
      .setDescending(descending);

    // Important: allocate buffers during render so we never bind a 0-sized storage buffer
    // on the very first frame (before `useFrame` runs).
    sorter.prepare();

    return depthKeysBuf ? sorter.getSortedIndicesBuffer() : null;
  }, [sorter, depthKeysBuf, count, descending]);

  useFrame(() => {
    if (!enabled) return;
    if (!depthKeysBuf) return;
    sorter.dispatch();
  });

  // Important: even when `enabled` is false, keep returning the same `sortedIndices` buffer.
  // This lets render paths "freeze" the last sorted order (and avoids any consumers treating
  // the buffer as absent / resetting state).
  return sortedIndicesBuf;
}
