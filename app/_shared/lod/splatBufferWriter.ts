import type { StorageBufferNode } from "three/webgpu";
import type { PlyPacked } from "../hooks/usePlyPacked";

export type SplatBufferSet = {
  centers: StorageBufferNode;
  covariances: StorageBufferNode;
  rgba: StorageBufferNode;
  shCoeffsL1?: StorageBufferNode | null;
  shCoeffsL2?: StorageBufferNode | null;
  shCoeffsL3?: StorageBufferNode | null;
};

export type AppendResult = {
  nextBase: number;
};

export function appendChunkToBuffers(
  buffers: SplatBufferSet,
  chunk: PlyPacked,
  baseIndex: number,
  capacity: number,
): AppendResult | null {
  const nextBase = baseIndex + chunk.count;
  if (nextBase > capacity) {
    return null;
  }

  (buffers.centers.value.array as Float32Array).set(chunk.center, baseIndex * 3);
  (buffers.covariances.value.array as Float32Array).set(
    chunk.covariance,
    baseIndex * 6,
  );
  (buffers.rgba.value.array as Uint32Array).set(chunk.rgba, baseIndex);

  buffers.centers.value.needsUpdate = true;
  buffers.covariances.value.needsUpdate = true;
  buffers.rgba.value.needsUpdate = true;

  if (buffers.shCoeffsL1 && chunk.shCoeffsL1) {
    (buffers.shCoeffsL1.value.array as Float32Array).set(
      chunk.shCoeffsL1,
      baseIndex * 9,
    );
    buffers.shCoeffsL1.value.needsUpdate = true;
  }

  if (buffers.shCoeffsL2 && chunk.shCoeffsL2Packed) {
    (buffers.shCoeffsL2.value.array as Uint32Array).set(
      chunk.shCoeffsL2Packed,
      baseIndex * 10,
    );
    buffers.shCoeffsL2.value.needsUpdate = true;
  }

  if (buffers.shCoeffsL3 && chunk.shCoeffsL3Packed) {
    (buffers.shCoeffsL3.value.array as Uint32Array).set(
      chunk.shCoeffsL3Packed,
      baseIndex * 14,
    );
    buffers.shCoeffsL3.value.needsUpdate = true;
  }

  return { nextBase };
}
