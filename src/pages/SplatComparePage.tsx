import { OrbitControls } from "@react-three/drei";
import { Leva, useControls } from "leva";
import { useEffect, useMemo, useState } from "react";
import {
  DoubleSide,
  MeshBasicNodeMaterial,
  MeshStandardNodeMaterial,
} from "three/webgpu";
import { createCovarianceEllipsoidNodes } from "../tsl/covarianceEllipsoid";
import { createSplatQuadNodes } from "../tsl/splatQuad";
import { WebGPUCanvas } from "../webgpu/WebGPUCanvas";

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
  const [splat] = useState(() => createSplatQuadNodes());

  const ellipsoidMaterial = useMemo(() => {
    const m = new MeshStandardNodeMaterial({
      side: DoubleSide,
      roughness: 0.75,
      metalness: 0.0,
    });
    m.vertexNode = ellipsoid.vertexNode;
    m.normalNode = ellipsoid.normalNode;
    return m;
  }, [ellipsoid]);

  const splatMaterial = useMemo(() => {
    const m = new MeshBasicNodeMaterial({ side: DoubleSide });
    m.transparent = true;
    m.depthTest = false;
    m.depthWrite = false;
    m.vertexNode = splat.vertexNode;
    m.colorNode = splat.colorNode;
    m.opacityNode = splat.opacityNode as never;
    return m;
  }, [splat]);

  useEffect(() => {
    // shared symmetric covariance:
    // [ m11  m12  m13 ]
    // [ m12  m22  m23 ]
    // [ m13  m23  m33 ]
    ellipsoid.uCenter.value.set(centerX, centerY, centerZ);
    ellipsoid.uCovA.value.set(m11, m12, m13);
    ellipsoid.uCovB.value.set(m22, m23, m33);
    ellipsoid.uCutoff.value = cutoff;

    splat.uCenter.value.set(centerX, centerY, centerZ);
    splat.uCovA.value.set(m11, m12, m13);
    splat.uCovB.value.set(m22, m23, m33);
    splat.uCutoff.value = cutoff;

    splat.uColor.value.set(color);
    splat.uParams.value.set(opacity, showQuadBg ? 1.0 : 0.0, quadBgAlpha);
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
      <Leva collapsed={false} />
      <div className="pageHeader">
        <h1>Compare: ellipsoid + splat sprite</h1>
        <p className="muted">
          Same center + same 3D covariance drive both: a deformed sphere (3D
          surface) and a projected gaussian sprite (2D). Sprite renders on top
          (no depth test).
        </p>
      </div>

      <WebGPUCanvas className="canvasWrap" camera={{ position: [3, 2.2, 3], fov: 50 }}>
        <OrbitControls makeDefault enableDamping />
        <ambientLight intensity={0.25} />
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
      </WebGPUCanvas>
    </div>
  );
}


