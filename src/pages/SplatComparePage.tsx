import { OrbitControls } from "@react-three/drei";
import { useControls } from "leva";
import { useEffect, useMemo, useState } from "react";
import {
  Color,
  DoubleSide,
  MeshBasicNodeMaterial,
  MeshStandardNodeMaterial,
  Vector3,
} from "three/webgpu";
import { createCovarianceEllipsoidNodes } from "../tsl/covarianceEllipsoid";
import {
  cameraProjectionMatrix,
  float,
  max,
  positionLocal,
  screenSize,
  sqrt,
  uniform,
  vec2,
  vec3,
  vec4,
} from "three/tsl";
import {
  DEFAULT_KERNEL_2D_SIZE,
  DEFAULT_MAX_SCREEN_SPACE_SPLAT_SIZE,
  DEFAULT_SPLAT_SCALE,
  createGaussianSplatFragmentStage,
  createGaussianSplatVertexStage,
} from "../tsl/gaussian/helpers";
import { unpackCovariance3D } from "../tsl/gaussian/covarianceMath";
import { WebGPUCanvasFrame } from "../webgpu/WebGPUCanvasFrame";

export function SplatComparePage() {
  const {
    // shared gaussian params
    centerX,
    centerY,
    centerZ,
    m11,
    m12,
    m13,
    m22,
    m23,
    m33,
    cutoff,

    // sprite look
    color,
    opacity,
    showQuadBg,
    quadBgAlpha,
  } = useControls("Compare (ellipsoid + sprite)", {
    centerX: { value: 0, min: -2, max: 2, step: 0.01 },
    centerY: { value: 0, min: -2, max: 2, step: 0.01 },
    centerZ: { value: 0, min: -2, max: 2, step: 0.01 },

    m11: { value: 0.02, min: 0.000001, max: 9, step: 0.000001 },
    m12: { value: 0, min: -4, max: 4, step: 0.000001 },
    m13: { value: 0, min: -4, max: 4, step: 0.000001 },
    m22: { value: 0.02, min: 0.000001, max: 9, step: 0.000001 },
    m23: { value: 0, min: -4, max: 4, step: 0.000001 },
    m33: { value: 0.02, min: 0.000001, max: 9, step: 0.000001 },

    cutoff: { value: 8.0, min: 0.1, max: 25, step: 0.01 },

    color: { value: "#ff8a3d" },
    opacity: { value: 1.0, min: 0, max: 1, step: 0.001 },
    showQuadBg: { value: true },
    quadBgAlpha: { value: 0.15, min: 0, max: 0.6, step: 0.01 },
  });

  const [ellipsoid] = useState(() => createCovarianceEllipsoidNodes());
  const [splat] = useState(() => {
    const uCenter = uniform(new Vector3()).setName("uCenter");
    const uCovA = uniform(new Vector3(1, 0, 0)).setName("uCovA");
    const uCovB = uniform(new Vector3(1, 0, 1)).setName("uCovB");
    const uColor = uniform(new Color("#ff8a3d")).setName("uColor");
    const uOpacity = uniform(1.0).setName("uOpacity");
    const uCutoff = uniform(8.0).setName("uCutoff");
    const uShowQuadBg = uniform(1.0).setName("uShowQuadBg");
    const uQuadBgAlpha = uniform(0.15).setName("uQuadBgAlpha");

    const focalPx = vec2(
      float(cameraProjectionMatrix[0].x).mul(screenSize.x).mul(0.5),
      float(cameraProjectionMatrix[1].y).mul(screenSize.y).mul(0.5)
    );

    const center = vec3(uCenter.x, uCenter.y, uCenter.z);
    const Vrk = unpackCovariance3D({ covA: uCovA, covB: uCovB });

    const corner = vec2(positionLocal.x, positionLocal.y);
    const cutoffA = float(uCutoff);
    const vPosition = corner.mul(sqrt(max(cutoffA, 1e-8))).toVarying("vPosition");
    const vColor = vec4(vec3(uColor), float(uOpacity)).toVarying("vColor");

    const vertexNode = createGaussianSplatVertexStage({
      center,
      Vrk,
      focalPx,
      kernel2DSize: float(DEFAULT_KERNEL_2D_SIZE),
      splatScale: float(DEFAULT_SPLAT_SCALE),
      maxScreenSpaceSplatSize: float(DEFAULT_MAX_SCREEN_SPACE_SPLAT_SIZE),
      inverseFocalAdjustment: 1.0,
    });

    const rgba = createGaussianSplatFragmentStage({
      vPosition,
      vColor,
      cutoffA,
    });

    const bgColor = vec3(0.12, 0.12, 0.12);
    const baseAlpha = float(uQuadBgAlpha).mul(float(uShowQuadBg));
    const outAlpha = max(baseAlpha, float(rgba.w));
    const mask = float(uShowQuadBg).select(1.0, float(rgba.w).div(max(float(uOpacity), 1e-6)));
    const outColor = bgColor.mix(vec3(uColor), mask.clamp());

    return {
      nodes: { vertexNode, colorNode: outColor, opacityNode: outAlpha },
      uniforms: {
        uCenter,
        uCovA,
        uCovB,
        uColor,
        uOpacity,
        uCutoff,
        uShowQuadBg,
        uQuadBgAlpha,
      },
    };
  });

  const ellipsoidMaterial = useMemo(() => {
    const m = new MeshStandardNodeMaterial({
      side: DoubleSide,
      roughness: 0.75,
      metalness: 0.0,
    });
    m.vertexNode = ellipsoid.nodes.vertexNode;
    m.normalNode = ellipsoid.nodes.normalNode;
    return m;
  }, [ellipsoid]);

  const splatMaterial = useMemo(() => {
    const m = new MeshBasicNodeMaterial({ side: DoubleSide });
    m.transparent = true;
    m.depthTest = false;
    m.depthWrite = false;
    m.vertexNode = splat.nodes.vertexNode;
    m.colorNode = splat.nodes.colorNode;
    m.opacityNode = splat.nodes.opacityNode as never;
    return m;
  }, [splat]);

  useEffect(() => {
    // shared symmetric covariance:
    // [ m11  m12  m13 ]
    // [ m12  m22  m23 ]
    // [ m13  m23  m33 ]
    ellipsoid.uniforms.uCenter.value.set(centerX, centerY, centerZ);
    ellipsoid.uniforms.uCovA.value.set(m11, m12, m13);
    ellipsoid.uniforms.uCovB.value.set(m22, m23, m33);
    ellipsoid.uniforms.uCutoff.value = cutoff;

    splat.uniforms.uCenter.value.set(centerX, centerY, centerZ);
    splat.uniforms.uCovA.value.set(m11, m12, m13);
    splat.uniforms.uCovB.value.set(m22, m23, m33);
    splat.uniforms.uColor.value.set(color);
    splat.uniforms.uOpacity.value = opacity;
    splat.uniforms.uCutoff.value = cutoff;
    splat.uniforms.uShowQuadBg.value = showQuadBg ? 1.0 : 0.0;
    splat.uniforms.uQuadBgAlpha.value = quadBgAlpha;
  }, [
    ellipsoid,
    splat,
    centerX,
    centerY,
    centerZ,
    m11,
    m12,
    m13,
    m22,
    m23,
    m33,
    cutoff,
    color,
    opacity,
    showQuadBg,
    quadBgAlpha,
  ]);

  return (
    <div className="page">
      <div className="pageHeader">
        <h1>Compare: ellipsoid + splat sprite</h1>
        <p className="muted">
          Same center + same 3D covariance drive both: a deformed sphere (3D
          surface) and a projected gaussian sprite (2D). Sprite renders on top
          (no depth test).
        </p>
        <div className="muted">
          File: <code>src/pages/SplatComparePage.tsx</code>
        </div>
      </div>

      <WebGPUCanvasFrame camera={{ position: [3, 2.2, 3], fov: 50 }}>
        <OrbitControls makeDefault enableDamping />
        <ambientLight intensity={0.35} />
        <hemisphereLight
          args={["#dfe8ff", "#1a1a1a", 0.45]}
          position={[0, 1, 0]}
        />
        <directionalLight position={[4, 6, 3]} intensity={1.2} />
        <gridHelper args={[10, 10]} />

        <mesh>
          <sphereGeometry args={[1, 64, 64]} />
          <primitive object={ellipsoidMaterial} attach="material" />
        </mesh>

        <mesh renderOrder={10}>
          <planeGeometry args={[2, 2]} />
          <primitive object={splatMaterial} attach="material" />
        </mesh>
      </WebGPUCanvasFrame>
    </div>
  );
}


