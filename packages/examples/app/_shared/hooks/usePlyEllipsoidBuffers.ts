import { useEffect, useMemo } from "react";
import { instancedArray } from "three/tsl";

/**
 * Allocates the set of storage buffers used by the PLY ellipsoids demo.
 * The returned buffers are stable for a given `count`.
 */
export function usePlyEllipsoidBuffers(count: number) {
  const centersBuf = useMemo(() => instancedArray(count, "vec3"), [count]);
  // 2 vec3 entries per splat => 2N elements
  const covBuf = useMemo(() => instancedArray(count * 2, "vec3"), [count]);
  const rgbaBuf = useMemo(() => instancedArray(count, "uint"), [count]);

  return { centersBuf, covBuf, rgbaBuf };
}

type PlyPackedLike = {
  count: number;
  center: Float32Array; // 3N
  covariance: Float32Array; // 6N
  rgba: Uint32Array; // N packed RGBA8
  shCoeffsL1?: Float32Array; // 9 * N
  shCoeffsL2Packed?: Uint32Array; // 10 * N
  shCoeffsL2Scale?: number;
  shCoeffsL3Packed?: Uint32Array; // 14 * N
  shCoeffsL3Scale?: number;
  shDegree?: number;
};

/**
 * Convenience hook: allocates `centers/cov/rgba` buffers for the given PLY data and uploads them.
 */
export function usePlyEllipsoidBuffersFromData(data: PlyPackedLike) {
  const { centersBuf, covBuf, rgbaBuf } = usePlyEllipsoidBuffers(data.count);
  const shDegree = data.shDegree ?? 0;
  const shCoeffsL1Buf = useMemo(
    () => (shDegree >= 1 ? instancedArray(data.count * 3, "vec3") : null),
    [data.count, shDegree],
  );
  const shCoeffsL2Buf = useMemo(
    () => (shDegree >= 2 ? instancedArray(data.count * 10, "uint") : null),
    [data.count, shDegree],
  );
  const shCoeffsL3Buf = useMemo(
    () => (shDegree >= 3 ? instancedArray(data.count * 14, "uint") : null),
    [data.count, shDegree],
  );

  useEffect(() => {
    (centersBuf.value.array as Float32Array).set(data.center);
    (covBuf.value.array as Float32Array).set(data.covariance);
    (rgbaBuf.value.array as Uint32Array).set(data.rgba);
    centersBuf.value.needsUpdate = true;
    covBuf.value.needsUpdate = true;
    rgbaBuf.value.needsUpdate = true;
    if (shCoeffsL1Buf && data.shCoeffsL1) {
      (shCoeffsL1Buf.value.array as Float32Array).set(data.shCoeffsL1);
      shCoeffsL1Buf.value.needsUpdate = true;
    }
    if (shCoeffsL2Buf && data.shCoeffsL2Packed) {
      (shCoeffsL2Buf.value.array as Uint32Array).set(data.shCoeffsL2Packed);
      shCoeffsL2Buf.value.needsUpdate = true;
    }
    if (shCoeffsL3Buf && data.shCoeffsL3Packed) {
      (shCoeffsL3Buf.value.array as Uint32Array).set(data.shCoeffsL3Packed);
      shCoeffsL3Buf.value.needsUpdate = true;
    }
  }, [
    data,
    centersBuf,
    covBuf,
    rgbaBuf,
    shCoeffsL1Buf,
    shCoeffsL2Buf,
    shCoeffsL3Buf,
  ]);

  return {
    centersBuf,
    covBuf,
    rgbaBuf,
    shCoeffsL1Buf,
    shCoeffsL2Buf,
    shCoeffsL2Scale: data.shCoeffsL2Scale ?? 1,
    shCoeffsL3Buf,
    shCoeffsL3Scale: data.shCoeffsL3Scale ?? 1,
    shDegree,
  };
}
