import type { Node, StorageBufferNode } from "three/webgpu";
import { Vector3 } from "three/webgpu";
import {
  add,
  cameraProjectionMatrix,
  cameraViewMatrix,
  clamp,
  div,
  exp,
  float,
  instanceIndex,
  mat3,
  max,
  mix,
  mul,
  positionLocal,
  uniform,
  vec2,
  vec3,
  vec4,
} from "three/tsl";
import { cholesky2D, sqrtCutoff, unpackCovariance3D } from "./gaussianCommon";

export type InstancedSplatQuadNodes = {
  nodes: {
    vertexNode: Node;
    colorNode: Node;
    opacityNode: Node;
  };
  uniforms: {
    /** Global cutoff (same meaning as in GaussianSplats3D: default 8). */
    uCutoff: ReturnType<typeof uniform<number>>;
    /**
     * Packed debug params:
     * x = opacityMultiplier
     * y = showQuadBg (0/1)
     * z = quadBgAlpha
     */
    uParams: ReturnType<typeof uniform<Vector3>>;
  };
  buffers: {
    splats: StorageBufferNode;
  };
};

export function createInstancedSplatQuadNodes(
  splats: StorageBufferNode
): InstancedSplatQuadNodes {
  const uCutoff = uniform(8.0).setName("uCutoff");
  const uParams = uniform(new Vector3(1.0, 1.0, 0.12)).setName("uParams");

  const s = splats.element(instanceIndex);

  const center4 = s.get("center");
  const covA4 = s.get("covA");
  const covB4 = s.get("covB");
  const colorOpacity4 = s.get("colorOpacity");

  const centerWorld = vec3(center4.x, center4.y, center4.z);
  const Vrk = unpackCovariance3D({ covA: covA4, covB: covB4 });

  // center in view/clip/ndc
  const viewCenter4 = cameraViewMatrix.mul(vec4(centerWorld, 1.0));
  const clipCenter4 = cameraProjectionMatrix.mul(viewCenter4);
  const ndcCenter = div(clipCenter4.xyz, clipCenter4.w);

  // Jacobian of perspective projection at viewCenter
  const fx = float(cameraProjectionMatrix[0].x);
  const fy = float(cameraProjectionMatrix[1].y);

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
  const W = mat3(cameraViewMatrix).transpose();
  const T = W.mul(J);

  // cov2Dm = transpose(T) * Vrk * T
  const cov2Dm = T.transpose().mul(Vrk).mul(T);

  // 2x2 symmetric block (XY): a=cov00, b=cov01, d=cov11
  const a = float(cov2Dm[0].x);
  const b = float(cov2Dm[0].y);
  const d = float(cov2Dm[1].y);

  const { l11, l21, l22 } = cholesky2D(a, b, d);

  // quad corners from geometry [-1..1]
  const corner = vec2(positionLocal.x, positionLocal.y);
  const { cutoff, radius } = sqrtCutoff(uCutoff);
  const vPosition = corner.mul(radius).toVarying("vPosition");

  // offset = L2 * vPosition
  const offset = add(
    mul(vPosition.x, vec2(l11, l21)),
    mul(vPosition.y, vec2(0.0, l22))
  );

  const ndcPos = add(ndcCenter.xy, offset);
  const vertexNode = vec4(vec3(ndcPos, ndcCenter.z), 1.0);

  // Fragment gaussian
  const A = vPosition.dot(vPosition);
  A.greaterThan(cutoff).discard();

  const instanceOpacity = float(colorOpacity4.w);
  const opacityMultiplier = float(uParams.x);
  const showQuadBg = float(uParams.y);
  const quadBgAlpha = float(uParams.z);

  const gaussianAlpha = exp(A.mul(-0.5))
    .mul(instanceOpacity)
    .mul(opacityMultiplier);
  const baseAlpha = quadBgAlpha.mul(showQuadBg);
  const opacityNode = max(baseAlpha, gaussianAlpha);

  // Color (instance rgb packed into colorOpacity.xyz)
  const instanceColor = vec3(colorOpacity4.x, colorOpacity4.y, colorOpacity4.z);
  const bgColor = vec3(0.12, 0.12, 0.12);
  const mask = clamp(
    div(gaussianAlpha, max(instanceOpacity.mul(opacityMultiplier), 1e-6)),
    0.0,
    1.0
  );
  const colorNode = mix(bgColor, instanceColor, mask);

  return {
    nodes: { vertexNode, colorNode, opacityNode },
    uniforms: { uCutoff, uParams },
    buffers: { splats },
  };
}
