import { Vector3, type Node } from "three/webgpu";
import {
  cameraProjectionMatrix,
  cameraViewMatrix,
  modelNormalMatrix,
  modelViewMatrix,
  normalLocal,
  positionLocal,
  uniform,
  vec3,
  vec4,
} from "three/tsl";
import { cholesky3DFromCov, sqrtCutoff } from "./gaussian/gaussianCommon";

export type CovarianceEllipsoidNodes = {
  nodes: {
    /** Clip-space position for the vertex shader. */
    vertexNode: Node;
    /** View/fragment normal for lighting. */
    normalNode: Node;
  };
  uniforms: {
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
  };
  buffers: Record<string, never>;
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
  const { L, invLT } = cholesky3DFromCov({ covA: uCovA, covB: uCovB });
  const { radius } = sqrtCutoff(uCutoff);

  const center = vec3(uCenter.x, uCenter.y, uCenter.z);
  const p = vec3(positionLocal.x, positionLocal.y, positionLocal.z).mul(radius);
  const localPos = L.mul(p).add(center);

  const vertexNode = cameraProjectionMatrix
    .mul(modelViewMatrix)
    .mul(vec4(localPos, 1.0));

  // `MeshStandardNodeMaterial.normalNode` is expected to be in view-space.
  const normalLocalEllipsoid = invLT
    .mul(vec3(normalLocal.x, normalLocal.y, normalLocal.z))
    .normalize();
  const normalNode = cameraViewMatrix
    .transformDirection(modelNormalMatrix.mul(normalLocalEllipsoid))
    .normalize();

  return {
    nodes: { vertexNode, normalNode },
    uniforms: { uCenter, uCovA, uCovB, uCutoff },
    buffers: {},
  };
}
