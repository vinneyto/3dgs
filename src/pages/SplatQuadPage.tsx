import { OrbitControls } from "@react-three/drei";
import { useControls } from "leva";
import { useEffect, useMemo, useState } from "react";
import { DoubleSide, MeshBasicNodeMaterial } from "three/webgpu";
import { createSplatQuadNodes } from "../tsl/splatQuad";
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

  const [demo] = useState(() => createSplatQuadNodes());
  const material = useMemo(() => {
    const m = new MeshBasicNodeMaterial({ side: DoubleSide });
    m.transparent = true;
    m.depthWrite = false;
    m.vertexNode = demo.vertexNode;
    m.colorNode = demo.colorNode;
    m.opacityNode = demo.opacityNode as never;
    return m;
  }, [demo]);

  useEffect(() => {
    demo.uCenter.value.set(centerX, centerY, centerZ);
    demo.uCovA.value.set(m11, m12, m13);
    demo.uCovB.value.set(m22, m23, m33);
    demo.uColor.value.set(color);
    demo.uCutoff.value = cutoff;
    demo.uParams.value.set(opacity, showQuadBg ? 1.0 : 0.0, quadBgAlpha);
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
