import { useFrame } from "@react-three/fiber";
import { useMemo } from "react";
import { instancedArray } from "three/tsl";
import type { StorageBufferNode } from "three/webgpu";
import { useWebGPU } from "./useWebGPU";
import { createBlockRadixSortIndices } from "../tsl/gaussian/radixSortIndices";

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
  const workgroupSize = 256;
  const numGroups = useMemo(() => Math.ceil(count / workgroupSize), [count]);

  // Ping-pong indices
  const indicesA = useMemo(() => instancedArray(count, "uint"), [count]);
  const indicesB = useMemo(() => instancedArray(count, "uint"), [count]);

  // Block-radix: group histograms + prefix buffers
  const groupHists = useMemo(
    () => instancedArray(numGroups * 256, "uint").toAtomic(),
    [numGroups]
  );
  const totals = useMemo(() => instancedArray(256, "uint"), []);
  const bucketBase = useMemo(() => instancedArray(256, "uint"), []);
  const groupBase = useMemo(
    () => instancedArray(numGroups * 256, "uint"),
    [numGroups]
  );

  const sorter = useMemo(() => {
    if (!depthKeysBuf) return null;
    return createBlockRadixSortIndices({
      depthKeys: depthKeysBuf,
      count,
      numGroups,
      indicesA,
      indicesB,
      groupHists,
      totals,
      bucketBase,
      groupBase,
      descending,
    });
  }, [
    depthKeysBuf,
    count,
    numGroups,
    indicesA,
    indicesB,
    groupHists,
    totals,
    bucketBase,
    groupBase,
    descending,
  ]);

  useFrame(() => {
    if (!enabled) return;
    if (!depthKeysBuf) return;
    if (!sorter) return;

    for (const c of sorter.computes) {
      gl.compute(c);
    }
  });

  return enabled && sorter ? sorter.sortedIndices : null;
}
