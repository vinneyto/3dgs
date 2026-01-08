import { OrbitControls } from "@react-three/drei";
import { useControls } from "leva";
import { useEffect, useMemo, useState } from "react";
import { Color, DoubleSide, MeshBasicNodeMaterial, Vector3 } from "three/webgpu";
import {
  cameraProjectionMatrix,
  exp,
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

export function SplatQuadPage() {
  const {
    centerX,
    centerY,
    centerZ,
    m11,
    m12,
    m13,
    m22,
    m23,
    m33,
    color,
    opacity,
    cutoff,
    showQuadBg,
    quadBgAlpha,
  } = useControls("Splat (quad)", {
    centerX: { value: 0, min: -2, max: 2, step: 0.01 },
    centerY: { value: 0, min: -2, max: 2, step: 0.01 },
    centerZ: { value: 0, min: -2, max: 2, step: 0.01 },

    m11: { value: 0.02, min: 0.000001, max: 1, step: 0.000001 },
    m12: { value: 0, min: -0.5, max: 0.5, step: 0.000001 },
    m13: { value: 0, min: -0.5, max: 0.5, step: 0.000001 },
    m22: { value: 0.02, min: 0.000001, max: 1, step: 0.000001 },
    m23: { value: 0, min: -0.5, max: 0.5, step: 0.000001 },
    m33: { value: 0.02, min: 0.000001, max: 1, step: 0.000001 },

    color: { value: "#ff8a3d" },
    opacity: { value: 1.0, min: 0, max: 1, step: 0.001 },
    cutoff: { value: 8.0, min: 0.1, max: 25, step: 0.01 },

    showQuadBg: { value: true },
    quadBgAlpha: { value: 0.15, min: 0, max: 0.6, step: 0.01 },
  });

  const [demo] = useState(() => {
    const uCenter = uniform(new Vector3()).setName("uCenter");
    const uCovA = uniform(new Vector3(1, 0, 0)).setName("uCovA");
    const uCovB = uniform(new Vector3(1, 0, 1)).setName("uCovB");
    const uColor = uniform(new Color("#ff8a3d")).setName("uColor");
    const uOpacity = uniform(1.0).setName("uOpacity");
    const uCutoff = uniform(8.0).setName("uCutoff");
    const uShowQuadBg = uniform(1.0).setName("uShowQuadBg");
    const uQuadBgAlpha = uniform(0.15).setName("uQuadBgAlpha");

    // focal length in pixels
    const focalPx = vec2(
      float(cameraProjectionMatrix[0].x).mul(screenSize.x).mul(0.5),
      float(cameraProjectionMatrix[1].y).mul(screenSize.y).mul(0.5)
    );

    const center = vec3(uCenter.x, uCenter.y, uCenter.z);
    const Vrk = unpackCovariance3D({ covA: uCovA, covB: uCovB });

    // vPosition is in "gaussian space": corner * sqrt(cutoffA)
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

    // Optional debug background to see quad bounds (keeps existing UI semantics).
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
  const material = useMemo(() => {
    const m = new MeshBasicNodeMaterial({ side: DoubleSide });
    m.transparent = true;
    m.depthWrite = false;
    m.vertexNode = demo.nodes.vertexNode;
    m.colorNode = demo.nodes.colorNode;
    m.opacityNode = demo.nodes.opacityNode as never;
    return m;
  }, [demo]);

  useEffect(() => {
    demo.uniforms.uCenter.value.set(centerX, centerY, centerZ);
    demo.uniforms.uCovA.value.set(m11, m12, m13);
    demo.uniforms.uCovB.value.set(m22, m23, m33);
    demo.uniforms.uColor.value.set(color);
    demo.uniforms.uOpacity.value = opacity;
    demo.uniforms.uCutoff.value = cutoff;
    demo.uniforms.uShowQuadBg.value = showQuadBg ? 1.0 : 0.0;
    demo.uniforms.uQuadBgAlpha.value = quadBgAlpha;
  }, [
    demo,
    centerX,
    centerY,
    centerZ,
    m11,
    m12,
    m13,
    m22,
    m23,
    m33,
    color,
    opacity,
    cutoff,
    showQuadBg,
    quadBgAlpha,
  ]);

  return (
    <div className="page">
      <div className="pageHeader">
        <h1>Splat quad (debug)</h1>
        <p className="muted">
          Solid-colored quad, but vertex shader already does: cov3D → cov2D →
          eigen basis → quad offset around projected center.
        </p>
        <div className="muted">
          File: <code>src/pages/SplatQuadPage.tsx</code>
        </div>
      </div>

      <WebGPUCanvasFrame camera={{ position: [2.5, 2.0, 2.5], fov: 50 }}>
        <OrbitControls makeDefault enableDamping />
        <gridHelper args={[10, 10]} />
        <mesh>
          <planeGeometry args={[2, 2]} />
          <primitive object={material} attach="material" />
        </mesh>
      </WebGPUCanvasFrame>
    </div>
  );
}
