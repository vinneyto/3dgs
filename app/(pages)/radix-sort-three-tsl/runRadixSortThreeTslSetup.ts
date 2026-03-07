import {
  compareCpuArraySortToGpu,
  prepareRadixSortInput,
  runRadixSortCpuArraySort,
  type RadixSortPreparedInput,
  type RadixSortWebgpuDemoResult,
} from "../radix-sort-webgpu/runRadixSortWebgpuSetup";
import { RadixSortThreeTslBenchmark } from "./RadixSortThreeTslBenchmark";

export const RADIX_SORT_THREE_TSL_DEMO_N = 10_000_000;
export const RADIX_SORT_THREE_TSL_DEMO_WORKGROUP_SIZE_X = 256;

let benchmarkSingleton: RadixSortThreeTslBenchmark | null = null;

function getBenchmark(): RadixSortThreeTslBenchmark {
  if (!benchmarkSingleton) {
    benchmarkSingleton = new RadixSortThreeTslBenchmark();
  }
  return benchmarkSingleton;
}

export function disposeRadixSortThreeTslBenchmark(): void {
  benchmarkSingleton?.dispose();
  benchmarkSingleton = null;
}

export {
  compareCpuArraySortToGpu,
  prepareRadixSortInput,
  runRadixSortCpuArraySort,
};
export type { RadixSortPreparedInput, RadixSortWebgpuDemoResult };

export type RadixSortThreeTslGpuRun = {
  result: RadixSortWebgpuDemoResult;
  gpuSorted: Uint32Array;
};

export async function runRadixSortThreeTslGpu({
  prepared,
  workgroupSizeX = RADIX_SORT_THREE_TSL_DEMO_WORKGROUP_SIZE_X,
  passes = 16,
  bitsPerPass = 2,
}: {
  prepared: RadixSortPreparedInput;
  workgroupSizeX?: number;
  passes?: number;
  bitsPerPass?: number;
}): Promise<RadixSortThreeTslGpuRun> {
  return getBenchmark().runGpu({
    prepared,
    workgroupSizeX,
    passes,
    bitsPerPass,
  });
}
