import type { Node } from "three/webgpu";
import {
  add,
  clamp,
  div,
  exp,
  Fn,
  float,
  mat3,
  max,
  mix,
  mul,
  sqrt,
  struct,
  sub,
  vec2,
  vec3,
  vec4,
} from "three/tsl";

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

/**
 * Compute `colorNode` for splat-quad rendering:
 * mixes a dark background with instance color based on gaussian alpha.
 *
 * Mirrors the logic used in `instancedSplatQuad.ts`.
 */
export const splatQuadColorNodeFn = Fn(
  ({
    instanceColor,
    gaussianAlpha,
    instanceOpacity,
    opacityMultiplier,
    A,
    cutoff,
  }: {
    instanceColor: Node; // vec3
    gaussianAlpha: Node; // float
    instanceOpacity: Node; // float
    opacityMultiplier: Node; // float
    A: Node; // float (typically dot(vPosition, vPosition))
    cutoff: Node; // float
  }) => {
    // Discard outside ellipse cutoff (same rule as in the calling shader).
    float(A).greaterThan(float(cutoff)).discard();

    const bgColor = vec3(0.12, 0.12, 0.12);
    const denom = max(
      mul(float(instanceOpacity), float(opacityMultiplier)),
      1e-6
    );
    const mask = clamp(div(float(gaussianAlpha), denom), 0.0, 1.0);
    return mix(bgColor, vec3(instanceColor), mask);
  }
);

/**
 * Full fragment-stage splat-quad logic wrapped in `Fn`.
 *
 * Returns RGBA where:
 * - rgb = mix(bgColor, instanceColor, mask)
 * - a   = max(baseAlpha, gaussianAlpha)
 *
 * This is the exact logic that used to live in `instancedSplatQuad.ts` (79-101),
 * but packaged as a single node function for reuse across WebGPU/WebGL pipelines.
 */
export const splatQuadFragmentNodeFn = Fn(
  ({
    instanceColor,
    instanceOpacity,
    opacityMultiplier,
    showQuadBg,
    quadBgAlpha,
    vPosition,
    cutoff,
  }: {
    instanceColor: Node; // vec3
    instanceOpacity: Node; // float
    opacityMultiplier: Node; // float
    showQuadBg: Node; // float (0/1)
    quadBgAlpha: Node; // float
    vPosition: Node; // vec2 (varying)
    cutoff: Node; // float
  }) => {
    const A = vec2(vPosition).dot(vec2(vPosition));

    // Discard outside ellipse cutoff
    float(A).greaterThan(float(cutoff)).discard();

    const gaussianAlpha = exp(float(A).mul(-0.5))
      .mul(float(instanceOpacity))
      .mul(float(opacityMultiplier));

    const baseAlpha = float(quadBgAlpha).mul(float(showQuadBg));
    const outAlpha = max(baseAlpha, gaussianAlpha);

    const bgColor = vec3(0.12, 0.12, 0.12);
    const denom = max(
      mul(float(instanceOpacity), float(opacityMultiplier)),
      1e-6
    );
    const mask = clamp(div(gaussianAlpha, denom), 0.0, 1.0);
    const outColor = mix(bgColor, vec3(instanceColor), mask);

    return vec4(outColor, outAlpha);
  }
);

/**
 * Vertex node computation wrapped in `Fn`:
 * given center+covariance projected to screen, returns clip-space `vertexNode`.
 *
 * Note: `vPosition` is expected to already be a varying (usually `corner * sqrtCutoff`).
 */
export const splatQuadVertexNodeFn = Fn(
  ({
    centerWorld,
    Vrk,
    vPosition,
    splatScale,
    cameraViewMatrixNode,
    cameraProjectionMatrixNode,
  }: {
    centerWorld: Node; // vec3
    Vrk: Node; // mat3
    vPosition: Node; // vec2 (varying)
    splatScale: Node; // float
    cameraViewMatrixNode: Node; // mat4
    cameraProjectionMatrixNode: Node; // mat4
  }) => {
    // center in view/clip/ndc
    const viewCenter4 = cameraViewMatrixNode.mul(vec4(centerWorld, 1.0));
    const clipCenter4 = cameraProjectionMatrixNode.mul(viewCenter4);
    const ndcCenter = div(clipCenter4.xyz, clipCenter4.w);

    // Jacobian of perspective projection at viewCenter
    const fx = float(cameraProjectionMatrixNode[0].x);
    const fy = float(cameraProjectionMatrixNode[1].y);

    const z = float(viewCenter4.z);
    const x = float(viewCenter4.x);
    const y = float(viewCenter4.y);
    const invZ2 = div(1.0, mul(z, z));

    const J = mat3(
      vec3(div(fx, z), 0.0, 0.0),
      vec3(0.0, div(fy, z), 0.0),
      vec3(
        mul(mul(mul(-1.0, fx), x), invZ2),
        mul(mul(mul(-1.0, fy), y), invZ2),
        0.0
      )
    );

    // W = transpose(mat3(viewMatrix))
    const W = mat3(cameraViewMatrixNode).transpose();
    const T = W.mul(J);

    // cov2Dm = transpose(T) * Vrk * T
    const cov2Dm = T.transpose().mul(Vrk).mul(T);

    // 2x2 symmetric block (XY): a=cov00, b=cov01, d=cov11
    const a = float(cov2Dm[0].x);
    const b = float(cov2Dm[0].y);
    const d = float(cov2Dm[1].y);

    const { l11, l21, l22 } = cholesky2D(a, b, d);

    // offset = (L2 * vPosition) * splatScale
    const offset = mul(
      add(mul(vPosition.x, vec2(l11, l21)), mul(vPosition.y, vec2(0.0, l22))),
      float(splatScale)
    );

    const ndcPos = add(ndcCenter.xy, offset);
    return vec4(vec3(ndcPos, ndcCenter.z), 1.0);
  }
);
