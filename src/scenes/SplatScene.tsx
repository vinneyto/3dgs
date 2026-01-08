import { OrbitControls } from "@react-three/drei";
import { useControls } from "leva";
import { useRef } from "react";
import type { InstancedMesh } from "three";
import { useDepthKeyCompute } from "../hooks/useDepthKeyCompute";
import { useInstancedEllipsoidPlyShader } from "../hooks/useInstancedEllipsoidPlyShader";
import { usePlyEllipsoidsMaterial } from "../hooks/usePlyEllipsoidsMaterial";
import { usePlyEllipsoidBuffersFromData } from "../hooks/usePlyEllipsoidBuffers";
import { useRadixSortDepthIndices } from "../hooks/useRadixSortDepthIndices";
import type { PlyPacked } from "../hooks/usePlyPacked";

export function SplatScene({
  data,
  controlsGroup = "Splats",
  ellipsoidSphereGeometryArgs = [1, 18, 14],
}: {
  data: PlyPacked;
  controlsGroup?: string;
  ellipsoidSphereGeometryArgs?: [number, number, number];
}) {
  const {
    cutoff,
    alphaDiscard,
    metalness,
    roughness,
    useDepth,
    computeDepthKeys,
    sortByDepth,
    debugDepth,
  } = useControls(controlsGroup, {
    cutoff: { value: 1.0, min: 0.05, max: 8.0, step: 0.01 },
    alphaDiscard: { value: 2.0 / 255.0, min: 0.0, max: 0.1, step: 0.0005 },
    roughness: { value: 0.8, min: 0, max: 1, step: 0.01 },
    metalness: { value: 0.0, min: 0, max: 1, step: 0.01 },
    useDepth: { value: true },
    computeDepthKeys: { value: true },
    sortByDepth: { value: true },
    debugDepth: { value: false },
  });

  const { centersBuf, covBuf, rgbaBuf } = usePlyEllipsoidBuffersFromData(data);

  const meshRef = useRef<InstancedMesh | null>(null);

  const depthKeysBuf = useDepthKeyCompute({
    enabled: computeDepthKeys,
    centersBuf,
    count: data.count,
    meshRef,
  });

  const sortedIndicesBuf = useRadixSortDepthIndices({
    enabled: sortByDepth && computeDepthKeys,
    depthKeysBuf,
    count: data.count,
    descending: true,
  });

  const ellipsoidShader = useInstancedEllipsoidPlyShader(
    centersBuf,
    covBuf,
    rgbaBuf,
    sortedIndicesBuf
  );

  const ellipsoidMaterial = usePlyEllipsoidsMaterial({
    shader: ellipsoidShader,
    useDepth,
    debugDepth,
    depthKeysBuf,
    cutoff,
    alphaDiscard,
    roughness,
    metalness,
  });

  return (
    <>
      <OrbitControls makeDefault enableDamping />
      <ambientLight intensity={0.35} />
      <hemisphereLight
        args={["#dfe8ff", "#1a1a1a", 0.45]}
        position={[0, 1, 0]}
      />
      <directionalLight position={[4, 6, 3]} intensity={1.2} />
      <gridHelper args={[10, 10]} />

      <instancedMesh
        args={[undefined, undefined, data.count]}
        frustumCulled={false}
        scale={[1, -1, 1]}
        ref={meshRef}
      >
        <sphereGeometry args={ellipsoidSphereGeometryArgs} />
        <primitive object={ellipsoidMaterial} attach="material" />
      </instancedMesh>
    </>
  );
}
