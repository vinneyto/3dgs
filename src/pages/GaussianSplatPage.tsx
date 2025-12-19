import { useFrame } from "@react-three/fiber";
import { useRef } from "react";
import type { Mesh } from "three";
import { WebGPUCanvas } from "../webgpu/WebGPUCanvas";

function RotatingCube() {
  const ref = useRef<Mesh>(null);

  useFrame((state) => {
    if (!ref.current) return;
    const t = state.clock.getElapsedTime();
    ref.current.rotation.y = t * 0.8;
    ref.current.rotation.x = t * 0.35;
  });

  return (
    <mesh ref={ref}>
      <boxGeometry />
      <meshBasicNodeMaterial color={0xff8a3d} />
    </mesh>
  );
}

export function GaussianSplatPage() {
  return (
    <div className="page">
      <div className="pageHeader">
        <h1>Gaussian Splat</h1>
        <p className="muted">Step 0: render a cube (WebGPU).</p>
      </div>

      <WebGPUCanvas
        className="canvasWrap"
        camera={{ position: [2.5, 2.0, 2.5], fov: 50 }}
      >
        <RotatingCube />
      </WebGPUCanvas>
    </div>
  );
}
