import { useFrame, useThree } from "@react-three/fiber";
import { useMemo } from "react";
import type { InstancedMesh, Matrix4 } from "three";
import { Matrix4 as ThreeMatrix4 } from "three";
import { useWebGPU } from "./useWebGPU";
import { instancedArray } from "three/tsl";
import type { StorageBufferNode } from "three/webgpu";
import { createDepthKeyCompute } from "../tsl/gaussian/depthKeyCompute";

/**
 * Creates and runs a TSL/WebGPU compute pass each frame to fill `depthKeys` buffer.
 *
 * Must be called inside a R3F <Canvas> (e.g. inside WebGPUCanvasFrame children),
 * because it uses `useFrame()` and `useThree()`.
 */
export function useDepthKeyCompute({
  enabled,
  centersBuf,
  count,
  meshRef,
}: {
  enabled: boolean;
  centersBuf: StorageBufferNode;
  count: number;
  meshRef: React.RefObject<InstancedMesh | null>;
}): StorageBufferNode | null {
  const gl = useWebGPU();
  const camera = useThree((s) => s.camera);
  const tmpModelView: Matrix4 = useMemo(() => new ThreeMatrix4(), []);

  const depthKeysBuf = useMemo(() => instancedArray(count, "uint"), [count]);
  const compute = useMemo(
    () => createDepthKeyCompute(centersBuf, depthKeysBuf, count),
    [centersBuf, depthKeysBuf, count]
  );

  useFrame(() => {
    if (!enabled) return;
    const mesh = meshRef.current;
    if (!mesh) return;

    // modelView = view * model
    tmpModelView.multiplyMatrices(camera.matrixWorldInverse, mesh.matrixWorld);
    compute.uniforms.uModelViewMatrix.value.copy(tmpModelView);
    compute.uniforms.uProjectionMatrix.value.copy(camera.projectionMatrix);

    gl.compute(compute.compute);
  });

  return enabled ? depthKeysBuf : null;
}
