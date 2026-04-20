import { useMemo } from "react";
import type { StorageBufferNode } from "three/webgpu";
import { instancedArray } from "three/tsl";
import { useWebGPU } from "./useWebGPU";
import { GPUPrefixSum } from "@/app/_shared/gpu/GPUPrefixSum";

export type PrefixSumController = {
  /**
   * Runs the multi-dispatch prefix sum on the current GPU queue (fire-and-forget).
   */
  compute: () => void;
  /**
   * Runs the multi-dispatch prefix sum and awaits GPU completion.
   */
  computeAsync: () => Promise<void>;
};

/**
 * Prefix sum (exclusive scan) for a `uint` storage buffer.
 *
 * Current API shape (as requested):
 * - accepts an input `StorageBufferNode`
 * - returns an output `StorageBufferNode` (same length)
 * - returns a controller with a `compute()` method to trigger execution
 *
 * Notes:
 * - Must be used inside a R3F <Canvas> with `three/webgpu` renderer (because it calls `useWebGPU()`).
 * - Assumes the input buffer length fits in u32 and uses 512-wide blocks (256 threads, 2 elems/thread).
 */
export function usePrefixSumCompute(
  input: StorageBufferNode,
  count?: number,
): { output: StorageBufferNode; controller: PrefixSumController } {
  const gl = useWebGPU();

  const n = useMemo(() => {
    if (typeof count === "number") return Math.max(0, count | 0);
    const arr = (input.value.array as ArrayLike<number> | undefined) ?? [];
    return (arr.length | 0) >>> 0;
  }, [input, count]);

  const output = useMemo(() => instancedArray(n, "uint"), [n]);

  const prefixSum = useMemo(() => {
    const ps = new GPUPrefixSum(gl);
    ps.setInputBuffer(input, n);
    ps.setOutputBuffer(output);
    ps.setCount(n);
    return ps;
  }, [gl, input, n, output]);

  const controller = useMemo<PrefixSumController>(() => {
    return {
      compute: () => {
        prefixSum.compute();
      },
      computeAsync: async () => {
        await prefixSum.computeAsync();
      },
    };
  }, [prefixSum]);

  return { output, controller };
}
