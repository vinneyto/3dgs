import type { Node } from "three/webgpu";
import {
  add,
  cameraProjectionMatrix,
  div,
  Fn,
  exp,
  float,
  mat3,
  max,
  min,
  modelViewMatrix,
  mul,
  positionLocal,
  screenSize,
  sqrt,
  sub,
  vec2,
  vec3,
  vec4,
} from "three/tsl";

/**
 * Perspective projection Jacobian at `viewCenter` (view-space center), used as an affine approximation
 * for projecting 3D covariance -> 2D covariance (GaussianSplats3D reference).
 *
 * Notes:
 * - `focal` is in pixels (fx, fy).
 *
 * Ported from GaussianSplats3D:
 * `GaussianSplats3D/src/splatmesh/SplatMaterial3D.js`.
 *
 * Reference WGSL (old `wgslFn` version):
 *
 * ```wgsl
 * fn calc_project_jacobian_3x3(viewCenter: vec3f, focal: vec2f) -> mat3x3f {
 *     let z = viewCenter.z;
 *     let s = 1.0 / (z * z);
 *     return mat3x3f(
 *         vec3f(focal.x / z, 0.0, -(focal.x * viewCenter.x) * s),
 *         vec3f(0.0, focal.y / z, -(focal.y * viewCenter.y) * s),
 *         vec3f(0.0, 0.0, 0.0),
 *     );
 * }
 * ```
 */
export const calcProjectJacobian3x3 = Fn(
  ({ viewCenter, focal }: { viewCenter: Node; focal: Node }) => {
    const vc = vec3(viewCenter);
    const f = vec2(focal);
    const z = float(vc.z);
    const s = div(1.0, mul(z, z));
    return mat3(
      vec3(div(f.x, z), 0.0, mul(mul(mul(-1.0, f.x), vc.x), s)),
      vec3(0.0, div(f.y, z), mul(mul(mul(-1.0, f.y), vc.y), s)),
      vec3(0.0, 0.0, 0.0)
    );
  }
);

/**
 * Project 3D covariance `Vrk` into screen-space covariance `cov2Dm` (no blur / no AA compensation).
 *
 * This matches GaussianSplats3D's vertex shader math:
 * - W = transpose(mat3(modelViewMatrix))
 * - T = W * J
 * - cov2Dm = transpose(T) * Vrk * T
 *
 * Where:
 * - `modelViewMatrix` transforms from local/world to view space (mat4).
 * - `J` is the affine projection Jacobian (3x3) at the splat center.
 * - `Vrk` is a symmetric 3D covariance matrix in the same "model/world" basis as the splat center.
 */
/**
 * Project 3D covariance `Vrk` into screen-space covariance `cov2Dm` (+ diagonal blur).
 *
 * This matches GaussianSplats3D's vertex shader math:
 * - W = transpose(mat3(modelViewMatrix))
 * - T = W * J
 * - cov2Dm = transpose(T) * Vrk * T
 * - cov2Dm[0][0] += kernel2DSize; cov2Dm[1][1] += kernel2DSize
 *
 * Reference WGSL (old `wgslFn` version):
 *
 * ```wgsl
 * fn project_cov3d_to_cov2d(modelViewMatrix: mat4x4f, J: mat3x3f, Vrk: mat3x3f, kernel2DSize: f32) -> mat3x3f {
 *     let M3 = mat3x3f(
 *         modelViewMatrix[0].xyz,
 *         modelViewMatrix[1].xyz,
 *         modelViewMatrix[2].xyz,
 *     );
 *     let W = transpose(M3);
 *     let T = W * J;
 *     var cov2Dm = transpose(T) * Vrk * T;
 *     cov2Dm[0][0] += kernel2DSize;
 *     cov2Dm[1][1] += kernel2DSize;
 *     return cov2Dm;
 * }
 * ```
 */
export const projectCov3DToCov2D = Fn(
  ({
    modelViewMatrix: mv,
    J,
    Vrk,
    kernel2DSize,
  }: {
    modelViewMatrix: Node; // mat4
    J: Node; // mat3
    Vrk: Node; // mat3
    kernel2DSize: Node; // float
  }) => {
    // W = transpose(mat3(modelViewMatrix))
    const W = mat3(mv).transpose();
    const T = W.mul(J);
    const cov2Dm0 = T.transpose().mul(Vrk).mul(T);

    // Add constant blur to the diagonal (GaussianSplats3D kernel2DSize).
    // cov2Dm[0][0] is cov2Dm[0].x, cov2Dm[1][1] is cov2Dm[1].y (column-major).
    const k = float(kernel2DSize);
    return mat3(
      vec3(add(cov2Dm0[0].x, k), cov2Dm0[0].y, cov2Dm0[0].z),
      vec3(cov2Dm0[1].x, add(cov2Dm0[1].y, k), cov2Dm0[1].z),
      vec3(cov2Dm0[2].x, cov2Dm0[2].y, cov2Dm0[2].z)
    );
  }
);

export const DEFAULT_KERNEL_2D_SIZE = 0.3;
export const DEFAULT_SPLAT_SCALE = 1.0;
export const DEFAULT_MAX_SCREEN_SPACE_SPLAT_SIZE = 2048.0;
// GaussianSplats3D uses sqrt(8) scaling in the vertex stage, so the fragment cutoff is A > 8.
export const DEFAULT_GAUSSIAN_CUTOFF_A = 8.0;

/**
 * Gaussian-splat vertex stage (position only), wrapped in `Fn`.
 *
 * `createGaussianSplatVertexStage` returns the clip-space position (vec4).
 * Varyings are emitted via `toVaryingAssign()` side-effects.
 *
 * Note: Vertex shaders can't `discard` in WGSL/WebGL; GaussianSplats3D uses `return;` after placing
 * the vertex offscreen. Here we emulate that by selecting an offscreen clip-space position.
 */
export const createGaussianSplatVertexStage = Fn(
  ({
    center,
    Vrk,
    focalPx,
    kernel2DSize,
    splatScale,
    maxScreenSpaceSplatSize,
    inverseFocalAdjustment,
  }: {
    center: Node; // vec3
    Vrk: Node; // mat3
    focalPx: Node; // vec2 (fx, fy) in pixels
    kernel2DSize: Node; // float (pixel^2)
    splatScale: Node; // float
    maxScreenSpaceSplatSize: Node; // float (pixels)
    inverseFocalAdjustment: Node; // float
  }) => {
    const corner = vec2(positionLocal.x, positionLocal.y);
    const sqrt8 = 2.8284271247461903;

    // Center in view/clip/ndc using built-in matrices.
    const viewCenter4 = modelViewMatrix.mul(vec4(center, 1.0));
    const clipCenter4 = cameraProjectionMatrix.mul(viewCenter4);
    const ndcCenter = div(clipCenter4.xyz, clipCenter4.w);

    // J + cov2Dm (with kernel2DSize applied).
    const J = calcProjectJacobian3x3({
      viewCenter: viewCenter4.xyz,
      focal: vec2(focalPx.x, focalPx.y),
    });

    const cov2Dm = projectCov3DToCov2D({
      modelViewMatrix,
      J,
      Vrk,
      kernel2DSize: float(kernel2DSize),
    });

    // Extract 2x2 symmetric block of cov2Dm: a=cov00, b=cov01, d=cov11
    const a = float(cov2Dm[0].x);
    const b = float(cov2Dm[0].y);
    const d = float(cov2Dm[1].y);

    // Eigen decomposition (GaussianSplats3D)
    const D = sub(mul(a, d), mul(b, b));
    const trace = add(a, d);
    const traceOver2 = mul(0.5, trace);
    const term2 = sqrt(max(0.1, sub(mul(traceOver2, traceOver2), D)));
    const eigenValue1 = add(traceOver2, term2);
    const eigenValue2 = sub(traceOver2, term2);

    // Early out: invalid covariance => send vertex offscreen (matches GS3D behavior).
    const invalid = float(eigenValue2).lessThanEqual(0.0);

    const eigenVector1 = vec2(b, sub(eigenValue1, a)).normalize();
    const eigenVector2 = vec2(eigenVector1.y, mul(-1.0, eigenVector1.x));

    // sqrt(8) standard deviations (GaussianSplats3D)
    const basisScale1 = min(
      mul(sqrt8, sqrt(eigenValue1)),
      float(maxScreenSpaceSplatSize)
    );
    const basisScale2 = min(
      mul(sqrt8, sqrt(max(eigenValue2, 1e-8))),
      float(maxScreenSpaceSplatSize)
    );
    const basisVector1 = eigenVector1.mul(float(splatScale)).mul(basisScale1);
    const basisVector2 = eigenVector2.mul(float(splatScale)).mul(basisScale2);

    // Convert pixel offsets -> NDC offsets (GaussianSplats3D includes inverseFocalAdjustment).
    const basisViewport = vec2(div(1.0, screenSize.x), div(1.0, screenSize.y));
    const ndcOffset = add(
      corner.x.mul(basisVector1),
      corner.y.mul(basisVector2)
    )
      .mul(basisViewport)
      .mul(2.0)
      .mul(float(inverseFocalAdjustment));

    const posOk = vec4(vec3(add(ndcCenter.xy, ndcOffset), ndcCenter.z), 1.0);
    const posOffscreen = vec4(0.0, 0.0, 2.0, 1.0);

    return invalid.select(posOffscreen, posOk);
  }
);

/**
 * Gaussian-splat fragment stage (RGBA), wrapped in `Fn` so we can `discard`.
 *
 * Mirrors GaussianSplats3D `SplatMaterial3D.buildFragmentShader()`:
 * - A = dot(vPosition, vPosition)
 * - if (A > 8) discard
 * - opacity = exp(-0.5 * A) * vColor.a
 * - out = vec4(vColor.rgb, opacity)
 */
export const createGaussianSplatFragmentStage = Fn(
  ({
    vPosition,
    vColor,
    cutoffA,
  }: {
    vPosition: Node; // vec2 (varying)
    vColor: Node; // vec4 (varying)
    cutoffA: Node; // float
  }) => {
    const vp = vec2(vPosition);
    const vc = vec4(vColor);

    const A = vp.dot(vp);
    float(A).greaterThan(float(cutoffA)).discard();

    const opacity = exp(float(A).mul(-0.5)).mul(float(vc.w));
    return vec4(vec3(vc.x, vc.y, vc.z), opacity);
  }
);

/**
 * Real spherical harmonics eval for L=1 (DC + 3 first-order terms), returning RGB.
 *
 * Order:
 * - Y00
 * - Y1,-1 (y)
 * - Y1, 0 (z)
 * - Y1, 1 (x)
 *
 * This mirrors common 3DGS real-SH conventions.
 */
export const evalSH_L1 = Fn(
  ({
    dir,
    c0,
    c1,
    c2,
    c3,
  }: {
    dir: Node; // vec3 (normalized)
    c0: Node; // vec3
    c1: Node; // vec3
    c2: Node; // vec3
    c3: Node; // vec3
  }) => {
    const d = vec3(dir);
    const x = float(d.x);
    const y = float(d.y);
    const z = float(d.z);

    // constants
    const k0 = 0.28209479177387814;
    const k1 = 0.4886025119029199;

    // rgb = c0*Y00 + c1*Y1,-1 + c2*Y1,0 + c3*Y1,1
    const rgb0 = mul(vec3(c0), k0);
    const rgb1 = mul(vec3(c1), mul(k1, y));
    const rgb2 = mul(vec3(c2), mul(k1, z));
    const rgb3 = mul(vec3(c3), mul(k1, x));
    return add(add(rgb0, rgb1), add(rgb2, rgb3));
  }
);
