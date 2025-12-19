import { useFrame } from "@react-three/fiber";
import { Leva, useControls } from "leva";
import { useRef } from "react";
import type { Mesh } from "three";
import { WebGPUCanvas } from "../webgpu/WebGPUCanvas";

function RotatingCube(props: {
  color: string;
  speedX: number;
  speedY: number;
}) {
  const ref = useRef<Mesh>(null);

  useFrame((state) => {
    if (!ref.current) return;
    const t = state.clock.getElapsedTime();
    ref.current.rotation.y = t * props.speedY;
    ref.current.rotation.x = t * props.speedX;
  });

  return (
    <mesh ref={ref}>
      <boxGeometry />
      <meshBasicNodeMaterial color={props.color} />
    </mesh>
  );
}

export function GaussianSplatPage() {
  const { color, speedX, speedY } = useControls("Cube", {
    color: { value: "#ff8a3d" },
    speedX: { value: 0.35, min: 0, max: 5, step: 0.01 },
    speedY: { value: 0.8, min: 0, max: 5, step: 0.01 },
  });

  return (
    <div className="page">
      <Leva collapsed={false} />
      <div className="pageHeader">
        <h1>Gaussian Splat</h1>
        <p className="muted">Step 0: render a cube (WebGPU).</p>
      </div>

      <WebGPUCanvas
        className="canvasWrap"
        camera={{ position: [2.5, 2.0, 2.5], fov: 50 }}
      >
        <RotatingCube color={color} speedX={speedX} speedY={speedY} />
      </WebGPUCanvas>
    </div>
  );
}
