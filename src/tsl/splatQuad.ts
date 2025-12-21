import { Color, Vector3, type Node } from "three/webgpu";
import {
  add,
  cameraProjectionMatrix,
  cameraViewMatrix,
  clamp,
  div,
  exp,
  float,
  mat3,
  max,
  mix,
  mul,
  positionLocal,
  sqrt,
  sub,
  uniform,
  vec2,
  vec3,
  vec4,
} from "three/tsl";

export type SplatQuadNodes = {
  uCenter: ReturnType<typeof uniform<Vector3>>;
  /** (m11, m12, m13) */
  uCovA: ReturnType<typeof uniform<Vector3>>;
  /** (m22, m23, m33) */
  uCovB: ReturnType<typeof uniform<Vector3>>;
  uColor: ReturnType<typeof uniform<Color>>;
  /**
   * Iso cutoff in "gaussian space" used for:
   * - quad radius (sqrt(cutoff))
   * - discard threshold
   *
   * Default: 8.0 (exp(-4) ≈ 0.018 at the edge).
   */
  uCutoff: ReturnType<typeof uniform<number>>;
  /**
   * Packed params:
   * x = opacity
   * y = showQuadBg (0/1)
   * z = quadBgAlpha
   */
  uParams: ReturnType<typeof uniform<Vector3>>;
  vertexNode: Node;
  colorNode: Node;
  opacityNode: Node;
};

/**
 * Splat debug step: build the "main" vertex transforms from SplatMaterial3D, but rasterize
 * just a solid-colored quad (no gaussian in fragment yet).
 *
 * Pipeline (high-level):
 * - read center + 3D covariance (symmetric 3x3)
 * - approximate perspective projection with a Jacobian J at the center
 * - project 3D covariance -> 2D covariance (screen/ndc space)
 * - Cholesky factorization of the 2D covariance (2x2) to get a deformation basis
 * - apply basis to quad corners (plane vertices) around ndcCenter
 */
export function createSplatQuadNodes(): SplatQuadNodes {
  const uCenter = uniform(new Vector3()).setName("uCenter");
  const uCovA = uniform(new Vector3(1, 0, 0)).setName("uCovA");
  const uCovB = uniform(new Vector3(1, 0, 1)).setName("uCovB");
  const uColor = uniform(new Color("#ff8a3d")).setName("uColor");
  const uCutoff = uniform(8.0).setName("uCutoff");
  const uParams = uniform(new Vector3(1.0, 1.0, 0.15)).setName("uParams");

  // 3D covariance (symmetric):
  // [ m11  m12  m13 ]
  // [ m12  m22  m23 ]
  // [ m13  m23  m33 ]
  const Vrk = mat3(
    vec3(uCovA.x, uCovA.y, uCovA.z),
    vec3(uCovA.y, uCovB.x, uCovB.y),
    vec3(uCovA.z, uCovB.y, uCovB.z)
  );

  // center in view/clip/ndc
  const viewCenter4 = cameraViewMatrix.mul(
    vec4(vec3(uCenter.x, uCenter.y, uCenter.z), 1.0)
  );
  const clipCenter4 = cameraProjectionMatrix.mul(viewCenter4);
  const ndcCenter = div(clipCenter4.xyz, clipCenter4.w);

  // Jacobian of the perspective projection at viewCenter (same structure as SplatMaterial3D).
  // We use fx/fy directly from the projection matrix (NDC units), so the resulting 2D covariance is in NDC.
  const fx = float(cameraProjectionMatrix[0].x);
  const fy = float(cameraProjectionMatrix[1].y);

  const z = float(viewCenter4.z);
  const x = float(viewCenter4.x);
  const y = float(viewCenter4.y);
  const s = div(1.0, mul(z, z));

  const J = mat3(
    vec3(div(fx, z), 0.0, 0.0),
    vec3(0.0, div(fy, z), 0.0),
    vec3(mul(mul(mul(-1.0, fx), x), s), mul(mul(mul(-1.0, fy), y), s), 0.0)
  );

  // W = transpose(mat3(viewMatrix))
  const W = mat3(cameraViewMatrix).transpose();
  const T = W.mul(J);

  // Projected covariance: cov2Dm = transpose(T) * Vrk * T
  const cov2Dm = T.transpose().mul(Vrk).mul(T);

  // 2x2 symmetric block we care about (XY):
  // a = cov00, b = cov01 (or cov10), d = cov11
  const a = float(cov2Dm[0].x);
  const b = float(cov2Dm[0].y);
  const d = float(cov2Dm[1].y);

  // 2D Cholesky for [[a,b],[b,d]] => L = [[l11,0],[l21,l22]]
  const eps = float(1e-8);
  const l11 = sqrt(max(a, eps));
  const l21 = div(b, l11);
  const l22 = sqrt(max(sub(d, mul(l21, l21)), eps));

  // plane corner coords from geometry (planeGeometry args [2,2] => [-1,1])
  const corner = vec2(positionLocal.x, positionLocal.y);
  const cutoff = max(float(uCutoff), eps);
  const radius = sqrt(cutoff);
  const vPosition = corner.mul(radius).toVarying("vPosition");

  // offset = L * vPosition
  const offset = add(
    mul(vPosition.x, vec2(l11, l21)),
    mul(vPosition.y, vec2(0.0, l22))
  );

  const ndcPos = add(ndcCenter.xy, offset);
  const vertexNode = vec4(vec3(ndcPos, ndcCenter.z), 1.0);

  // Fragment: gaussian in the "pre-deformation" coordinate space (vPosition).
  const A = vPosition.dot(vPosition);
  A.greaterThan(cutoff).discard();

  const opacity = float(uParams.x);
  const showQuadBg = float(uParams.y);
  const quadBgAlpha = float(uParams.z);

  const gaussianAlpha = exp(A.mul(-0.5)).mul(opacity);
  const baseAlpha = quadBgAlpha.mul(showQuadBg);
  const opacityNode = max(baseAlpha, gaussianAlpha);

  // Optional debug: tint background darker so the quad bounds are visible.
  const bgColor = vec3(0.12, 0.12, 0.12);
  const mask = clamp(div(gaussianAlpha, max(opacity, 1e-6)), 0.0, 1.0);
  const colorNode = mix(bgColor, vec3(uColor), mask);

  return {
    vertexNode,
    colorNode,
    opacityNode,
    uCenter,
    uCovA,
    uCovB,
    uColor,
    uCutoff,
    uParams,
  };
}
