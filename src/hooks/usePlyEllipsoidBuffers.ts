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
};

/**
 * Convenience hook: allocates `centers/cov/rgba` buffers for the given PLY data and uploads them.
 */
export function usePlyEllipsoidBuffersFromData(data: PlyPackedLike) {
  const { centersBuf, covBuf, rgbaBuf } = usePlyEllipsoidBuffers(data.count);

  useEffect(() => {
    (centersBuf.value.array as Float32Array).set(data.center);
    (covBuf.value.array as Float32Array).set(data.covariance);
    (rgbaBuf.value.array as Uint32Array).set(data.rgba);
    centersBuf.value.needsUpdate = true;
    covBuf.value.needsUpdate = true;
    rgbaBuf.value.needsUpdate = true;
  }, [data, centersBuf, covBuf, rgbaBuf]);

  return { centersBuf, covBuf, rgbaBuf };
}
