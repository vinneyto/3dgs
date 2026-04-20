import type { Node, StorageBufferNode } from "three/webgpu";
import {
  add,
  bitAnd,
  cameraPosition,
  cameraProjectionMatrix,
  div,
  float,
  instanceIndex,
  log,
  mat3,
  max,
  modelWorldMatrix,
  positionLocal,
  screenSize,
  shiftRight,
  sqrt,
  uint,
  vec2,
  vec3,
  vec4,
} from "three/tsl";
import {
  DEFAULT_GAUSSIAN_CUTOFF_A,
  DEFAULT_KERNEL_2D_SIZE,
  DEFAULT_MAX_SCREEN_SPACE_SPLAT_SIZE,
  DEFAULT_SPLAT_SCALE,
  encodeColorLinear,
  createGaussianSplatFragmentStage,
  createGaussianSplatVertexStage,
  buildShContributionFromBuffers,
} from "./helpers";

function unpackRGBA8UintToColorOpacity(rgbaPacked: Node): {
  colorNode: Node; // vec3
  opacityNode: Node; // float
} {
  const inv255 = 1.0 / 255.0;
  const rU = bitAnd(rgbaPacked, uint(0x000000ff));
  const gU = bitAnd(shiftRight(rgbaPacked, uint(8)), uint(0x000000ff));
  const bU = bitAnd(shiftRight(rgbaPacked, uint(16)), uint(0x000000ff));
  const aU = bitAnd(shiftRight(rgbaPacked, uint(24)), uint(0x000000ff));

  const colorNode = vec3(
    float(rU).mul(inv255),
    float(gU).mul(inv255),
    float(bU).mul(inv255),
  );
  const opacityNode = div(float(aU), 255.0);
  return { colorNode, opacityNode };
}

export type InstancedSplatNodes = {
  positionNode: Node; // vec4
  colorNode: Node; // vec3
  opacityNode: Node; // float
};

export type InstancedSplatCutoffMode =
  | "fixed" // GS3D-style: vPosition scaled by sqrt8, cutoffA fixed (default 8)
  | "opacity"; // PLY-style: cutoffA derived from opacity, vPosition scaled by sqrt(cutoffA)

/**
 * Instanced Gaussian splats driven by storage buffers.
 *
 * Uses:
 * - `createGaussianSplatVertexStage` (vertex position)
 * - `createGaussianSplatFragmentStage` (fragment RGBA w/ discard)
 *
 * Buffers follow the same packing as the legacy splat-quad PLY path:
 * - `centers`: vec3[N]
 * - `cov`: vec3[2N]  (covA at 2*i, covB at 2*i+1)
 * - `rgba`: u32[N] packed RGBA8
 */
export function instancedSplat({
  centers,
  cov,
  rgba,
  shCoeffsL1,
  shCoeffsL2,
  shCoeffsL2Scale = 1,
  shDegree = 0,
  sortedIndices,
  kernel2DSize = DEFAULT_KERNEL_2D_SIZE,
  splatScale = DEFAULT_SPLAT_SCALE,
  maxScreenSpaceSplatSize = DEFAULT_MAX_SCREEN_SPACE_SPLAT_SIZE,
  inverseFocalAdjustment = 1.0,
  cutoffA = DEFAULT_GAUSSIAN_CUTOFF_A,
  cutoffMode = "opacity",
  opacityMultiplier = 1.0,
  encodeLinear = true,
  shDebugColorOnly = false,
  debugWorldViewDir = false,
  enableSh = true,
}: {
  centers: StorageBufferNode;
  cov: StorageBufferNode;
  rgba: StorageBufferNode;
  shCoeffsL1?: StorageBufferNode | null;
  shCoeffsL2?: StorageBufferNode | null;
  shCoeffsL2Scale?: number;
  /** Max SH degree present (0/1/2/3). */
  shDegree?: number;
  sortedIndices?: StorageBufferNode | null;
  kernel2DSize?: number;
  splatScale?: number;
  maxScreenSpaceSplatSize?: number;
  inverseFocalAdjustment?: number;
  cutoffA?: number;
  /**
   * How to compute gaussian cutoff in fragment space.
   *
   * - "fixed": GS3D-style (A > 8, vPosition scaled by sqrt8)
   * - "opacity": PLY-style (cutoff derived from opacity, vPosition scaled by sqrt(cutoffA))
   */
  cutoffMode?: InstancedSplatCutoffMode;
  /** Only used when cutoffMode === "opacity". */
  opacityMultiplier?: number;
  /**
   * Spark-style `encodeLinear`:
   * if true, treat packed RGB as sRGB and convert to linear via `pow(rgb, 2.2)`.
   */
  encodeLinear?: boolean;
  /** Debug: output only SH contribution (no base color). */
  shDebugColorOnly?: boolean;
  /** Debug: output worldViewDir as RGB. */
  debugWorldViewDir?: boolean;
  /** Enable SH contribution. */
  enableSh?: boolean;
}): InstancedSplatNodes {
  // Compute focal length in PIXELS directly in shader (GaussianSplats3D-style):
  // fxPx = P00 * (W/2), fyPx = P11 * (H/2) where screenSize is drawing-buffer size (physical px).
  const focalPx = vec2(
    float(cameraProjectionMatrix[0].x).mul(screenSize.x).mul(0.5),
    float(cameraProjectionMatrix[1].y).mul(screenSize.y).mul(0.5),
  );

  const splatIndex = sortedIndices
    ? sortedIndices.element(instanceIndex)
    : instanceIndex;

  const center = centers.element(splatIndex);

  // cov entries live at indices (2*i) and (2*i+1)
  const covBase = splatIndex.mul(2);
  const covA3 = cov.element(covBase);
  const covB3 = cov.element(add(covBase, 1));

  // 3D covariance matrix (symmetric)
  const Vrk = mat3(
    vec3(covA3.x, covA3.y, covA3.z),
    vec3(covA3.y, covB3.x, covB3.y),
    vec3(covA3.z, covB3.y, covB3.z),
  );

  const rgbaPacked = rgba.element(splatIndex);
  const { colorNode: instanceColor0, opacityNode: instanceOpacity } =
    unpackRGBA8UintToColorOpacity(rgbaPacked);
  const instanceColor = vec3(instanceColor0);

  const worldCenter4 = modelWorldMatrix.mul(vec4(center, 1.0));
  const worldViewDir = vec3(worldCenter4.xyz)
    .sub(vec3(cameraPosition))
    .normalize();
  const shContribution = enableSh
    ? buildShContributionFromBuffers({
        dir: worldViewDir,
        shCoeffsL1,
        shCoeffsL2,
        shCoeffsL2Scale,
        shDegree,
        splatIndex,
      })
    : vec3(0.0);
  const worldViewDirColor = add(worldViewDir.mul(0.5), vec3(0.5));
  const shadedColor = debugWorldViewDir
    ? worldViewDirColor
    : shDebugColorOnly
      ? shContribution
      : add(instanceColor, shContribution);
  const shadedColorLinear = encodeLinear
    ? encodeColorLinear(shadedColor)
    : shadedColor;

  // Varyings (created here, used by fragment stage)
  const corner = vec2(positionLocal.x, positionLocal.y);
  const sqrt8 = 2.8284271247461903;

  let vPosition: Node;
  let cutoffNode: Node;

  if (cutoffMode === "opacity") {
    // Match `createInstancedSplatQuadPlyNodes` logic:
    // cutoffA = max(0, 2 * ln(255 * opacity)), vPosition = corner * sqrt(cutoffA)
    const opacityForCut = max(
      float(instanceOpacity).mul(float(opacityMultiplier)),
      1e-6,
    );
    const cutoffAFromOpacity = max(0.0, log(opacityForCut.mul(255.0)).mul(2.0));
    const radius = sqrt(max(cutoffAFromOpacity, 1e-8));
    vPosition = corner.mul(radius).toVarying("vPosition");
    cutoffNode = cutoffAFromOpacity.toVarying("vCutoffA");
  } else {
    // GS3D-style fixed cutoff.
    vPosition = corner.mul(sqrt8).toVarying("vPosition");
    cutoffNode = float(cutoffA);
  }

  const vColor = vec4(shadedColorLinear, instanceOpacity).toVarying("vColor");

  const positionNode = createGaussianSplatVertexStage({
    center,
    Vrk,
    focalPx,
    kernel2DSize: float(kernel2DSize),
    splatScale: float(splatScale),
    maxScreenSpaceSplatSize: float(maxScreenSpaceSplatSize),
    inverseFocalAdjustment: float(inverseFocalAdjustment),
  });

  const rgbaOut = createGaussianSplatFragmentStage({
    vPosition,
    vColor,
    cutoffA: cutoffNode,
  });

  const colorNode = vec3(rgbaOut.x, rgbaOut.y, rgbaOut.z).clamp(0.0, 1.0);
  const opacityNode = float(rgbaOut.w);

  return { positionNode, colorNode, opacityNode };
}
