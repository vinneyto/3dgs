import type { Node } from "three/webgpu";
import { div, float, mat3, max, mul, sqrt, struct, sub, vec3 } from "three/tsl";

export const SplatInstanceStruct = struct(
  {
    center: "vec4",
    covA: "vec4",
    covB: "vec4",
    colorOpacity: "vec4",
  },
  "SplatInstance"
);

export type Cov6 = {
  /** (m11, m12, m13) */
  covA: Node;
  /** (m22, m23, m33) */
  covB: Node;
};

export function unpackCovariance3D({ covA, covB }: Cov6) {
  // 3D covariance (symmetric):
  // [ m11  m12  m13 ]
  // [ m12  m22  m23 ]
  // [ m13  m23  m33 ]
  return mat3(
    vec3(covA.x, covA.y, covA.z),
    vec3(covA.y, covB.x, covB.y),
    vec3(covA.z, covB.y, covB.z)
  );
}

export function sqrtCutoff(uCutoff: Node, eps = 1e-8) {
  const cutoff = max(float(uCutoff), float(eps));
  return { cutoff, radius: sqrt(cutoff) };
}

export function cholesky3DFromCov({ covA, covB }: Cov6, eps = 1e-8) {
  // covariance (symmetric):
  // [ a  b  c ]
  // [ b  d  e ]
  // [ c  e  f ]
  const a = float(covA.x);
  const b = float(covA.y);
  const c = float(covA.z);
  const d = float(covB.x);
  const e = float(covB.y);
  const f = float(covB.z);

  const eNode = float(eps);
  const l11 = sqrt(max(a, eNode));
  const l21 = div(b, l11);
  const l31 = div(c, l11);
  const l22 = sqrt(max(sub(d, mul(l21, l21)), eNode));
  const l32 = div(sub(e, mul(l31, l21)), l22);
  const l33 = sqrt(max(sub(sub(f, mul(l31, l31)), mul(l32, l32)), eNode));

  // columns of lower-triangular L
  const L = mat3(vec3(l11, l21, l31), vec3(0.0, l22, l32), vec3(0.0, 0.0, l33));

  // inv(L) (lower-triangular)
  const i11 = div(1.0, l11);
  const i22 = div(1.0, l22);
  const i33 = div(1.0, l33);
  const i21 = div(mul(-1.0, l21), mul(l22, l11));
  const i32 = div(mul(-1.0, l32), mul(l33, l22));
  const i31 = div(sub(mul(l32, l21), mul(l31, l22)), mul(mul(l33, l22), l11));

  const invL = mat3(
    vec3(i11, i21, i31),
    vec3(0.0, i22, i32),
    vec3(0.0, 0.0, i33)
  );

  return { L, invLT: invL.transpose() };
}

export function cholesky2D(a: Node, b: Node, d: Node, eps = 1e-8) {
  const eNode = float(eps);
  const l11 = sqrt(max(float(a), eNode));
  const l21 = div(float(b), l11);
  const l22 = sqrt(max(sub(float(d), mul(l21, l21)), eNode));
  return { l11, l21, l22 };
}
