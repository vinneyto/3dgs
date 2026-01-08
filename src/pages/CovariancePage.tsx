import { OrbitControls } from "@react-three/drei";
import { useControls } from "leva";
import { useEffect, useMemo, useState } from "react";
import { DoubleSide, MeshStandardNodeMaterial } from "three/webgpu";
import { createCovarianceEllipsoidNodes } from "../tsl/covarianceEllipsoid";
import { WebGPUCanvasFrame } from "../webgpu/WebGPUCanvasFrame";

export function CovariancePage() {
  const { centerX, centerY, centerZ, m11, m12, m13, m22, m23, m33 } =
    useControls("Covariance (3D)", {
      centerX: { value: 0, min: -2, max: 2, step: 0.01 },
      centerY: { value: 0, min: -2, max: 2, step: 0.01 },
      centerZ: { value: 0, min: -2, max: 2, step: 0.01 },

      m11: { value: 1, min: 0.0001, max: 9, step: 0.0001 },
      m12: { value: 0, min: -4, max: 4, step: 0.0001 },
      m13: { value: 0, min: -4, max: 4, step: 0.0001 },
      m22: { value: 1, min: 0.0001, max: 9, step: 0.0001 },
      m23: { value: 0, min: -4, max: 4, step: 0.0001 },
      m33: { value: 1, min: 0.0001, max: 9, step: 0.0001 },
    });

  const [demo] = useState(() => createCovarianceEllipsoidNodes());
  const material = useMemo(() => {
    const m = new MeshStandardNodeMaterial({
      side: DoubleSide,
      roughness: 0.75,
      metalness: 0.0,
    });
    m.vertexNode = demo.nodes.vertexNode;
    m.normalNode = demo.nodes.normalNode;
    return m;
  }, [demo]);

  useEffect(() => {
    // center
    demo.uniforms.uCenter.value.set(centerX, centerY, centerZ);

    // symmetric covariance:
    // [ m11  m12  m13 ]
    // [ m12  m22  m23 ]
    // [ m13  m23  m33 ]
    demo.uniforms.uCovA.value.set(m11, m12, m13);
    demo.uniforms.uCovB.value.set(m22, m23, m33);
  }, [demo, centerX, centerY, centerZ, m11, m12, m13, m22, m23, m33]);

  return (
    <div className="page">
      <div className="pageHeader">
        <h1>Covariance → Ellipsoid (debug)</h1>
        <p className="muted">
          We send 3 center coords + 6 covariance numbers as uniforms. Vertex
          shader deforms a unit sphere using a Cholesky factorization (matrix
          square-root).
        </p>
        <div className="muted">
          File: <code>src/pages/CovariancePage.tsx</code>
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
          <primitive object={material} attach="material" />
        </mesh>
      </WebGPUCanvasFrame>
    </div>
  );
}
