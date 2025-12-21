import { Vector3, type Node } from "three/webgpu";
import {
  add,
  cameraProjectionMatrix,
  div,
  float,
  mat3,
  max,
  modelViewMatrix,
  mul,
  normalLocal,
  positionLocal,
  sqrt,
  sub,
  uniform,
  vec3,
  vec4,
} from "three/tsl";

export type CovarianceEllipsoidNodes = {
  uCenter: ReturnType<typeof uniform<Vector3>>;
  /** (m11, m12, m13) */
  uCovA: ReturnType<typeof uniform<Vector3>>;
  /** (m22, m23, m33) */
  uCovB: ReturnType<typeof uniform<Vector3>>;
  /**
   * Iso-surface cutoff in "gaussian space".
   * - 1.0 means "1-sigma" surface (unit sphere before deformation).
   * - 8.0 matches the common splat quad cutoff (exp(-4) ≈ 0.018).
   */
  uCutoff: ReturnType<typeof uniform<number>>;
  /** Clip-space position for the vertex shader. */
  vertexNode: Node;
  /** View/fragment normal for lighting. */
  normalNode: Node;
};

/**
 * Debug material: draws a deformed sphere as an ellipsoid defined by a 3x3 covariance.
 *
 * Important: covariance stores "squared scale" information. To actually deform a unit sphere into an ellipsoid,
 * we need a matrix A such that covariance = A * transpose(A). We compute A with a Cholesky factorization
 * (lower-triangular L): covariance = L * transpose(L). Then we use L as the deformation transform.
 */
export function createCovarianceEllipsoidNodes(): CovarianceEllipsoidNodes {
  const uCenter = uniform(new Vector3()).setName("uCenter");
  const uCovA = uniform(new Vector3(1, 0, 0)).setName("uCovA");
  const uCovB = uniform(new Vector3(1, 0, 1)).setName("uCovB");
  const uCutoff = uniform(1.0).setName("uCutoff");

  // covariance (symmetric):
  // [ a  b  c ]
  // [ b  d  e ]
  // [ c  e  f ]
  const a = float(uCovA.x);
  const b = float(uCovA.y);
  const c = float(uCovA.z);
  const d = float(uCovB.x);
  const e = float(uCovB.y);
  const f = float(uCovB.z);

  // Cholesky factorization (assumes SPD; we clamp to avoid NaNs for bad debug values)
  const eps = float(1e-8);
  const l11 = sqrt(max(a, eps));
  const l21 = div(b, l11);
  const l31 = div(c, l11);
  const l22 = sqrt(max(sub(d, mul(l21, l21)), eps));
  const l32 = div(sub(e, mul(l31, l21)), l22);
  const l33 = sqrt(max(sub(sub(f, mul(l31, l31)), mul(l32, l32)), eps));

  // This is the deformation matrix L (lower-triangular):
  //
  // [ l11   0    0  ]
  // [ l21  l22   0  ]
  // [ l31  l32  l33 ]
  //
  // In GLSL/WGSL constructors, `mat3(c0,c1,c2)` uses column vectors.
  // So columns are:
  // c0 = (l11, l21, l31)
  // c1 = (0,   l22, l32)
  // c2 = (0,   0,   l33)
  const L = mat3(vec3(l11, l21, l31), vec3(0.0, l22, l32), vec3(0.0, 0.0, l33));

  // Apply L to local vertex position (unit sphere -> ellipsoid)
  // p' = L * (p * sqrt(cutoff))
  const px = float(positionLocal.x);
  const py = float(positionLocal.y);
  const pz = float(positionLocal.z);

  const p = vec3(px, py, pz);
  const cutoff = max(float(uCutoff), eps);
  const pScaled = p.mul(sqrt(cutoff));
  const t = L.mul(pScaled);
  const tx = float(t.x);
  const ty = float(t.y);
  const tz = float(t.z);

  // Add center offset (world/local, assuming mesh has identity transform)
  const wx = add(tx, float(uCenter.x));
  const wy = add(ty, float(uCenter.y));
  const wz = add(tz, float(uCenter.z));

  const vertexNode = cameraProjectionMatrix
    .mul(modelViewMatrix)
    .mul(vec4(vec3(wx, wy, wz), 1.0));

  // Correct normal transform for p' = L * p is: n' = transpose(inverse(L)) * n
  // For lower-triangular L:
  // inv(L) =
  // [ i11   0    0  ]
  // [ i21  i22   0  ]
  // [ i31  i32  i33 ]
  const i11 = div(1.0, l11);
  const i22 = div(1.0, l22);
  const i33 = div(1.0, l33);
  const i21 = div(mul(-1.0, l21), mul(l22, l11));
  const i32 = div(mul(-1.0, l32), mul(l33, l22));
  const i31 = div(sub(mul(l32, l21), mul(l31, l22)), mul(mul(l33, l22), l11));

  // Same thing as explicit matrix nodes (columns):
  //
  // invL =
  // [ i11   0    0  ]
  // [ i21  i22   0  ]
  // [ i31  i32  i33 ]
  //
  // invLT = transpose(invL) =
  // [ i11  i21  i31 ]
  // [  0   i22  i32 ]
  // [  0    0   i33 ]
  const invL = mat3(
    vec3(i11, i21, i31),
    vec3(0.0, i22, i32),
    vec3(0.0, 0.0, i33)
  );
  const invLT = invL.transpose();

  const nx = float(normalLocal.x);
  const ny = float(normalLocal.y);
  const nz = float(normalLocal.z);

  const normalNode = invLT.mul(vec3(nx, ny, nz)).normalize();

  return { vertexNode, normalNode, uCenter, uCovA, uCovB, uCutoff };
}
