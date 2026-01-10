import { Color, type Node, Vector3, Vector4 } from "three/webgpu";
import { clamp, float, mul, positionLocal, sub, uniform, vec3 } from "three/tsl";
import { evalSH_L1 } from "../gaussian/helpers";

export type ShSphereL1Nodes = {
  nodes: {
    colorNode: Node;
    opacityNode: Node;
  };
  uniforms: {
    uC0: ReturnType<typeof uniform<Color>>;
    uC1: ReturnType<typeof uniform<Color>>;
    uC2: ReturnType<typeof uniform<Color>>;
    uC3: ReturnType<typeof uniform<Color>>;
    uAmp: ReturnType<typeof uniform<Vector4>>;
    /** Camera direction (from origin to camera), normalized. */
    uCameraDir: ReturnType<typeof uniform<Vector3>>;
    /** Camera distance from origin in the same space as the sphere radius. */
    uCameraDist: ReturnType<typeof uniform<number>>;
    /** 1 => visualize signed values by mapping [-1..1] -> [0..1]. 0 => raw clamp with bias. */
    uSignedView: ReturnType<typeof uniform<number>>;
    uBias: ReturnType<typeof uniform<number>>;
    uScale: ReturnType<typeof uniform<number>>;
  };
  buffers: Record<string, never>;
};

export function createShSphereL1Nodes(): ShSphereL1Nodes {
  // Coefficient "colors" (really RGB vectors).
  const uC0 = uniform(new Color("#ffffff")).setName("uC0");
  // axis-like defaults so terms are visually distinct
  const uC1 = uniform(new Color("#00ff66")).setName("uC1"); // y
  const uC2 = uniform(new Color("#3399ff")).setName("uC2"); // z
  const uC3 = uniform(new Color("#ff3344")).setName("uC3"); // x

  // Amplitudes for each term: (dc, y, z, x)
  const uAmp = uniform(new Vector4(1.0, 1.0, 1.0, 1.0)).setName("uAmp");

  const uCameraDir = uniform(new Vector3(1, 0, 0)).setName("uCameraDir");
  const uCameraDist = uniform(2.0).setName("uCameraDist");
  const uSignedView = uniform(0.0).setName("uSignedView");

  // Display remap: rgb01 = clamp(rgb * scale + bias, 0..1)
  const uScale = uniform(1.0).setName("uScale");
  const uBias = uniform(0.0).setName("uBias");

  // Per-fragment view direction from surface point to camera position.
  // cameraPos = cameraDir * cameraDist (cameraDir comes from the widget).
  const vPos = vec3(positionLocal.x, positionLocal.y, positionLocal.z).toVarying(
    "vPosLocal"
  );
  const cameraPos = vec3(uCameraDir).mul(float(uCameraDist));
  const dir = sub(cameraPos, vec3(vPos)).normalize();

  const c0 = vec3(uC0).mul(uAmp.x);
  const c1 = vec3(uC1).mul(uAmp.y);
  const c2 = vec3(uC2).mul(uAmp.z);
  const c3 = vec3(uC3).mul(uAmp.w);

  const rgb = evalSH_L1({ dir, c0, c1, c2, c3 });
  const rgbRaw = clamp(mul(rgb, float(uScale)).add(float(uBias)), 0.0, 1.0);
  const rgbSigned = clamp(
    mul(rgb, float(uScale)).mul(0.5).add(0.5),
    0.0,
    1.0
  );
  const rgb01 = float(uSignedView).greaterThan(0.5).select(rgbSigned, rgbRaw);

  return {
    nodes: {
      colorNode: rgb01,
      opacityNode: float(1.0),
    },
    uniforms: {
      uC0,
      uC1,
      uC2,
      uC3,
      uAmp,
      uCameraDir,
      uCameraDist,
      uSignedView,
      uBias,
      uScale,
    },
    buffers: {},
  };
}

