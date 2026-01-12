import { useThree } from "@react-three/fiber";
import type { WebGPURenderer } from "three/webgpu";

/**
 * Typed accessor for the current R3F renderer when using `three/webgpu`.
 * Note: the R3F store types `gl` as WebGLRenderer; for WebGPU we cast.
 */
export function useWebGPU(): WebGPURenderer {
  return useThree((state) => state.gl) as unknown as WebGPURenderer;
}
