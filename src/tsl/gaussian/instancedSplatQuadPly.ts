import type { Node, StorageBufferNode } from "three/webgpu";
import { Vector3 } from "three/webgpu";
import {
  abs,
  cameraProjectionMatrix,
  cameraViewMatrix,
  div,
  float,
  instanceIndex,
  mat3,
  max,
  min,
  modelWorldMatrix,
  mul,
  positionLocal,
  screenSize,
  sqrt,
  uniform,
  uint,
  vec2,
  vec3,
} from "three/tsl";
import { unpackCovariance3D } from "./gaussianCommon";
import { add, bitAnd, exp, shiftRight, vec4 } from "three/tsl";

function unpackRGBA8UintToColorOpacity(rgbaPacked: Node): {
  colorNode: Node;
  opacityNode: Node;
} {
  const inv255 = 1.0 / 255.0;
  const rU = bitAnd(rgbaPacked, uint(0x000000ff));
  const gU = bitAnd(shiftRight(rgbaPacked, uint(8)), uint(0x000000ff));
  const bU = bitAnd(shiftRight(rgbaPacked, uint(16)), uint(0x000000ff));
  const aU = bitAnd(shiftRight(rgbaPacked, uint(24)), uint(0x000000ff));

  const colorNode = vec3(
    float(rU).mul(inv255),
    float(gU).mul(inv255),
    float(bU).mul(inv255)
  );
  const opacityNode = div(float(aU), 255.0);
  return { colorNode, opacityNode };
}

export type InstancedSplatQuadPlyNodes = {
  nodes: {
    vertexNode: Node;
    colorNode: Node;
    opacityNode: Node;
  };
  uniforms: {
    /** Screen-space scale multiplier for splats (default 1). */
    uSplatScale: ReturnType<typeof uniform<number>>;
    /** Gaussian blur kernel added to cov2D diagonal in pixel units (default 0.3 like GaussianSplats3D). */
    uKernel2DSize: ReturnType<typeof uniform<number>>;
    /** Clamp basis vectors to this many pixels in screen space (default 2048 like GaussianSplats3D). */
    uMaxScreenSpaceSplatSize: ReturnType<typeof uniform<number>>;
    /** If 1: apply alpha compensation sqrt(detOrig/detBlur) (default 1). */
    uAntialiasCompensation: ReturnType<typeof uniform<number>>;
    /**
     * Packed debug params:
     * x = opacityMultiplier
     * y = showQuadBg (0/1)
     * z = quadBgAlpha
     */
    uParams: ReturnType<typeof uniform<Vector3>>;
  };
  buffers: {
    centers: StorageBufferNode;
    cov: StorageBufferNode;
    rgba: StorageBufferNode;
    sortedIndices?: StorageBufferNode | null;
  };
};

/**
 * Instanced Gaussian splat quads driven by the PLY ellipsoid buffers:
 * - centers: vec3[N]
 * - cov: vec3[2N]  (covA/covB per instance)
 * - rgba: uint[N] packed RGBA8
 *
 * If `sortedIndices` is provided, instance i fetches splatIndex = sortedIndices[i].
 */
export function createInstancedSplatQuadPlyNodes(
  centers: StorageBufferNode,
  cov: StorageBufferNode,
  rgba: StorageBufferNode,
  sortedIndices?: StorageBufferNode | null
): InstancedSplatQuadPlyNodes {
  const uSplatScale = uniform(1.0).setName("uSplatScale");
  const uKernel2DSize = uniform(0.3).setName("uKernel2DSize");
  const uMaxScreenSpaceSplatSize = uniform(2048.0).setName("uMaxScreenSpaceSplatSize");
  const uAntialiasCompensation = uniform(1.0).setName("uAntialiasCompensation");
  const uParams = uniform(new Vector3(1.0, 0.0, 0.12)).setName("uParams");

  const splatIndex = sortedIndices
    ? sortedIndices.element(instanceIndex)
    : instanceIndex;

  const center = centers.element(splatIndex);
  const rgbaPacked = rgba.element(splatIndex);
  const { colorNode: instanceColor, opacityNode: instanceOpacity } =
    unpackRGBA8UintToColorOpacity(rgbaPacked);

  // cov entries live at indices (2*i) and (2*i+1)
  const covBase = splatIndex.mul(2);
  const covA3 = cov.element(covBase);
  const covB3 = cov.element(covBase.add(1));

  // PLY data lives in object-local space; apply model transform so scaling/rotation (incl. scale=[1,-1,1])
  // affects both center and covariance consistently with the ellipsoid path.
  const centerWorld = modelWorldMatrix.mul(vec4(center, 1.0)).xyz;

  const Vlocal = unpackCovariance3D({
    covA: vec4(covA3, 0.0),
    covB: vec4(covB3, 0.0),
  });

  // Linear part of modelWorldMatrix (column-major 3x3)
  const M3 = mat3(
    vec3(modelWorldMatrix[0].x, modelWorldMatrix[0].y, modelWorldMatrix[0].z),
    vec3(modelWorldMatrix[1].x, modelWorldMatrix[1].y, modelWorldMatrix[1].z),
    vec3(modelWorldMatrix[2].x, modelWorldMatrix[2].y, modelWorldMatrix[2].z)
  );

  // Covariance transforms as: V_world = M * V_local * M^T
  const Vrk = M3.mul(Vlocal).mul(M3.transpose());

  // GaussianSplats3D convention: use sqrt(8) sigma cutoff.
  const sqrt8 = 2.8284271247461903;

  // quad corners from geometry [-1..1]
  const corner = vec2(positionLocal.x, positionLocal.y);
  // vPosition is the fragment-space gaussian coordinate (scaled by sqrt8)
  const vPosition = corner.mul(sqrt8).toVarying("vPosition");

  // center in view/clip/ndc
  const viewCenter4 = cameraViewMatrix.mul(vec4(centerWorld, 1.0));
  const clipCenter4 = cameraProjectionMatrix.mul(viewCenter4);
  const ndcCenter = div(clipCenter4.xyz, clipCenter4.w);

  // Jacobian of perspective projection at viewCenter (affine approximation), in PIXEL units.
  // GaussianSplats3D uses focal lengths in pixels.
  const fxPx = float(cameraProjectionMatrix[0].x).mul(screenSize.x).mul(0.5);
  const fyPx = float(cameraProjectionMatrix[1].y).mul(screenSize.y).mul(0.5);

  const z = float(viewCenter4.z);
  const x = float(viewCenter4.x);
  const y = float(viewCenter4.y);
  const invZ2 = div(1.0, z.mul(z));

  const J = mat3(
    vec3(div(fxPx, z), 0.0, 0.0),
    vec3(0.0, div(fyPx, z), 0.0),
    vec3(
      mul(mul(mul(-1.0, fxPx), x), invZ2),
      mul(mul(mul(-1.0, fyPx), y), invZ2),
      0.0
    )
  );

  // W = transpose(mat3(viewMatrix))
  const W = mat3(cameraViewMatrix).transpose();
  const T = W.mul(J);

  // cov2Dm = transpose(T) * Vrk * T
  const cov2Dm = T.transpose().mul(Vrk).mul(T);

  // 2x2 symmetric block (XY): a=cov00, b=cov01, d=cov11
  const a0 = float(cov2Dm[0].x);
  const b0 = float(cov2Dm[0].y);
  const d0 = float(cov2Dm[1].y);

  // kernel2DSize is in pixel units, and cov2D entries are in pixel^2, so add directly (as in GaussianSplats3D).
  // Optional alpha compensation like GaussianSplats3D: alpha *= sqrt(detOrig / detBlur)
  const detOrig = a0.mul(d0).sub(b0.mul(b0));

  const a = a0.add(float(uKernel2DSize));
  const b = b0;
  const d = d0.add(float(uKernel2DSize));

  const detBlur = a.mul(d).sub(b.mul(b));
  const comp = sqrt(max(div(detOrig, detBlur), 0.0));
  const opacityComp = float(uAntialiasCompensation).greaterThan(0.5).select(comp, 1.0);
  const vOpacity = float(instanceOpacity).mul(opacityComp).toVarying("vOpacity");

  // Eigen decomposition (matches GaussianSplats3D)
  const D = a.mul(d).sub(b.mul(b));
  const trace = a.add(d);
  const traceOver2 = trace.mul(0.5);
  const term2 = sqrt(max(0.1, traceOver2.mul(traceOver2).sub(D)));
  const eigenValue1 = traceOver2.add(term2);
  const eigenValue2 = max(traceOver2.sub(term2), 1e-8);

  const ev1Raw = vec2(b, eigenValue1.sub(a));
  const ev1Len = abs(ev1Raw.x).add(abs(ev1Raw.y));
  const eigenVector1 = ev1Len.lessThan(1e-10).select(vec2(1.0, 0.0), ev1Raw.normalize());
  const eigenVector2 = vec2(eigenVector1.y, eigenVector1.x.mul(-1.0));

  // Basis vectors are in pixel units (since eigenvalues are pixel^2).
  const basisLen1 = min(
    float(sqrt8).mul(sqrt(eigenValue1)),
    float(uMaxScreenSpaceSplatSize)
  );
  const basisLen2 = min(
    float(sqrt8).mul(sqrt(eigenValue2)),
    float(uMaxScreenSpaceSplatSize)
  );

  const basisVector1Px = eigenVector1.mul(basisLen1).mul(float(uSplatScale));
  const basisVector2Px = eigenVector2.mul(basisLen2).mul(float(uSplatScale));

  // pixelOffset = corner.x * basisVector1 + corner.y * basisVector2
  const pixelOffset = add(basisVector1Px.mul(corner.x), basisVector2Px.mul(corner.y));

  // Convert pixels -> NDC: ndcPerPx = 2 / screenSize
  const ndcOffset = pixelOffset.mul(vec2(div(2.0, screenSize.x), div(2.0, screenSize.y)));
  const ndcPos = add(ndcCenter.xy, ndcOffset);
  const posNDC = vec4(vec3(ndcPos, ndcCenter.z), 1.0);

  const opacityMultiplier = float(uParams.x);
  const showQuadBg = float(uParams.y);
  const quadBgAlpha = float(uParams.z);

  // Fragment: fixed sqrt8 cutoff (A > 8 discard), opacity = exp(-0.5*A) * (opacity*comp) * opacityMultiplier
  const A = vec2(vPosition).dot(vec2(vPosition));
  float(A).greaterThan(8.0).discard();
  const gaussianAlpha = exp(float(A).mul(-0.5))
    .mul(float(vOpacity))
    .mul(float(opacityMultiplier));
  const baseAlpha = float(quadBgAlpha).mul(float(showQuadBg));
  const outAlpha = max(baseAlpha, gaussianAlpha);
  const rgbaOut = vec4(vec3(instanceColor), outAlpha);

  return {
    nodes: {
      vertexNode: posNDC,
      colorNode: rgbaOut.xyz,
      opacityNode: rgbaOut.w,
    },
    uniforms: {
      uSplatScale,
      uKernel2DSize,
      uMaxScreenSpaceSplatSize,
      uAntialiasCompensation,
      uParams,
    },
    buffers: { centers, cov, rgba, sortedIndices: sortedIndices ?? null },
  };
}


