import { Color, Vector3, type Node } from "three/webgpu";
import {
  add,
  cameraProjectionMatrix,
  cameraViewMatrix,
  clamp,
  div,
  Fn,
  float,
  max,
  min,
  mul,
  positionLocal,
  screenSize,
  uniform,
  uv,
  vec2,
  vec3,
  vec4,
} from "three/tsl";

/**
 * Build a billboard (camera-facing quad in screen/NDC space) from a world-space center.
 *
 * Notes:
 * - Geometry is expected to be a plane with corners in `positionLocal.xy` in [-1..1]
 *   (e.g. `planeGeometry args={[2,2]}`).
 * - Size is specified in pixels and clamped to [minSizePx..maxSizePx].
 * - Uses `screenSize` so it works on both WebGL and WebGPU.
 */
export const billboardVertexFromPosition = Fn(
  ({
    centerWorld,
    radiusWorld,
    minSizePx,
    maxSizePx,
  }: {
    centerWorld: Node; // vec3
    radiusWorld: Node; // float (world-space radius)
    minSizePx: Node; // float
    maxSizePx: Node; // float
  }) => {
    // Project center -> NDC
    const viewCenter4 = cameraViewMatrix.mul(vec4(vec3(centerWorld), 1.0));
    const clipCenter4 = cameraProjectionMatrix.mul(viewCenter4);
    const ndcCenter = div(clipCenter4.xyz, clipCenter4.w);

    // Perspective size attenuation:
    // radiusPx ~= (radiusWorld * projScale / depth) * (screenSize/2)
    const depth = max(float(viewCenter4.z).negate(), 1e-6);
    const fx = float(cameraProjectionMatrix[0].x);
    const fy = float(cameraProjectionMatrix[1].y);
    const ndcRadiusX = float(radiusWorld).mul(fx).div(depth);
    const ndcRadiusY = float(radiusWorld).mul(fy).div(depth);
    const sizePxX = ndcRadiusX.mul(screenSize.x).mul(0.5);
    const sizePxY = ndcRadiusY.mul(screenSize.y).mul(0.5);
    const desiredPx = min(sizePxX, sizePxY);

    // Clamp pixel radius, then convert px -> NDC offsets
    const sPx = clamp(desiredPx, float(minSizePx), float(maxSizePx));
    const res = screenSize; // vec2 in pixels (drawing buffer size)
    const sNdc = vec2(
      div(mul(sPx, 2.0), max(res.x, 1.0)),
      div(mul(sPx, 2.0), max(res.y, 1.0))
    );

    // Quad corner from geometry in [-1..1]
    const corner = vec2(positionLocal.x, positionLocal.y);
    const offset = mul(corner, sNdc);

    const ndcPos = add(ndcCenter.xy, offset);
    return vec4(vec3(ndcPos, ndcCenter.z), 1.0);
  }
);

/**
 * Circle-in-quad fragment logic based on UV:
 * - discards outside unit circle
 * - returns RGBA (rgb = color, a = 1)
 *
 * UV space: assumes `uv()` in [0..1].
 */
export const circleFromUV = Fn(({ color }: { color: Node }) => {
  const uv0 = uv();
  const p = uv0.sub(0.5).mul(2.0); // [-1..1]
  const r2 = p.dot(p);
  r2.greaterThan(1.0).discard();

  return vec4(vec3(color), 1.0);
});

export type BillboardCircleNodes = {
  nodes: {
    vertexNode: Node;
    colorNode: Node;
    opacityNode: Node;
  };
  uniforms: {
    uColor: ReturnType<typeof uniform<Color>>;
    /** World-space radius (shared). Final pixel size is computed from camera perspective and clamped. */
    uRadiusWorld: ReturnType<typeof uniform<number>>;
    uMinSizePx: ReturnType<typeof uniform<number>>;
    uMaxSizePx: ReturnType<typeof uniform<number>>;
  };
  buffers: Record<string, never>;
};

/**
 * Convenience builder for a simple "billboard circle" material:
 * - billboard vertex from `uCenterWorld`
 * - circle fragment from `uv()`
 *
 * If you want to read `centerWorld` from an attribute/buffer instead, use
 * `billboardVertexFromPosition()` directly and build your own wrapper.
 */
export function createBillboardCircleNodes({
  minSizePx = 4,
  maxSizePx = 64,
  radiusWorld = 0.06,
  color = "#ff8a3d",
}: {
  minSizePx?: number;
  maxSizePx?: number;
  radiusWorld?: number;
  color?: string;
}): BillboardCircleNodes & {
  uniforms: BillboardCircleNodes["uniforms"] & {
    uCenterWorld: ReturnType<typeof uniform<Vector3>>;
  };
} {
  const uCenterWorld = uniform(new Vector3()).setName("uCenterWorld");
  const uColor = uniform(new Color(color)).setName("uColor");
  const uRadiusWorld = uniform(radiusWorld).setName("uRadiusWorld");
  const uMinSizePx = uniform(minSizePx).setName("uMinSizePx");
  const uMaxSizePx = uniform(maxSizePx).setName("uMaxSizePx");

  const vertexNode = billboardVertexFromPosition({
    centerWorld: vec3(uCenterWorld),
    radiusWorld: uRadiusWorld,
    minSizePx: uMinSizePx,
    maxSizePx: uMaxSizePx,
  });

  const rgba = circleFromUV({ color: vec3(uColor) });

  return {
    nodes: { vertexNode, colorNode: rgba.xyz, opacityNode: rgba.w },
    uniforms: { uCenterWorld, uColor, uRadiusWorld, uMinSizePx, uMaxSizePx },
    buffers: {},
  };
}
