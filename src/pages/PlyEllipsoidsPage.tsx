import { OrbitControls } from "@react-three/drei";
import { useControls } from "leva";
import { useRef } from "react";
import type { InstancedMesh } from "three";
import { useDepthKeyCompute } from "../hooks/useDepthKeyCompute";
import { useInstancedEllipsoidPlyShader } from "../hooks/useInstancedEllipsoidPlyShader";
import { usePlyEllipsoidsMaterial } from "../hooks/usePlyEllipsoidsMaterial";
import { usePlyEllipsoidBuffersFromData } from "../hooks/usePlyEllipsoidBuffers";
import { usePlyPacked, type PlyPacked } from "../hooks/usePlyPacked";
import { WebGPUCanvasFrame } from "../webgpu/WebGPUCanvasFrame";

const PLY_URL = "/cactus_splat3_30kSteps_142k_splats.ply";

function PlyEllipsoidsScene({ data }: { data: PlyPacked }) {
  const {
    cutoff,
    metalness,
    roughness,
    useDepth,
    computeDepthKeys,
    debugDepth,
  } = useControls("PLY ellipsoids", {
    cutoff: { value: 1.0, min: 0.05, max: 8.0, step: 0.01 },
    roughness: { value: 0.8, min: 0, max: 1, step: 0.01 },
    metalness: { value: 0.0, min: 0, max: 1, step: 0.01 },
    useDepth: { value: true },
    computeDepthKeys: { value: true },
    debugDepth: { value: false },
  });

  const { centersBuf, covBuf, rgbaBuf } = usePlyEllipsoidBuffersFromData(data);

  const shader = useInstancedEllipsoidPlyShader(centersBuf, covBuf, rgbaBuf);

  const meshRef = useRef<InstancedMesh | null>(null);

  const depthKeysBuf = useDepthKeyCompute({
    enabled: computeDepthKeys,
    centersBuf: shader.buffers.centers,
    count: data.count,
    meshRef,
  });

  const material = usePlyEllipsoidsMaterial({
    shader,
    useDepth,
    debugDepth,
    depthKeysBuf,
    cutoff,
    roughness,
    metalness,
  });

  return (
    <>
      <OrbitControls makeDefault enableDamping />
      <ambientLight intensity={0.25} />
      <directionalLight position={[4, 6, 3]} intensity={1.2} />
      <gridHelper args={[10, 10]} />

      <instancedMesh
        args={[undefined, undefined, data.count]}
        frustumCulled={false}
        scale={[1, -1, 1]}
        ref={meshRef}
      >
        <sphereGeometry args={[1, 24, 24]} />
        <primitive object={material} attach="material" />
      </instancedMesh>
    </>
  );
}

export function PlyEllipsoidsPage() {
  const { status, data } = usePlyPacked(PLY_URL);

  return (
    <div className="page">
      <div className="pageHeader">
        <h1>PLY ellipsoids (instanced)</h1>
        <p className="muted">
          Renders ellipsoids from PLY using three separate storage buffers:
          center, covariance, rgba.
        </p>
        <div className="muted">
          Status: <code>{status}</code>
        </div>
      </div>

      {data ? (
        <WebGPUCanvasFrame
          camera={{ position: [4, 3, 4], fov: 50, near: 0.1, far: 100 }}
          gl={{ antialias: false }}
        >
          <PlyEllipsoidsScene data={data} />
        </WebGPUCanvasFrame>
      ) : null}
    </div>
  );
}
