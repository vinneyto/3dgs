import { OrbitControls } from "@react-three/drei";
import { useControls } from "leva";
import { useRef } from "react";
import type { InstancedMesh } from "three";
import { useDepthKeyCompute } from "../hooks/useDepthKeyCompute";
import { useInstancedEllipsoidPlyShader } from "../hooks/useInstancedEllipsoidPlyShader";
import { useInstancedSplatQuadPlyShader } from "../hooks/useInstancedSplatQuadPlyShader";
import { usePlyEllipsoidsMaterial } from "../hooks/usePlyEllipsoidsMaterial";
import { usePlyEllipsoidBuffersFromData } from "../hooks/usePlyEllipsoidBuffers";
import { usePlySplatQuadsMaterial } from "../hooks/usePlySplatQuadsMaterial";
import { useRadixSortDepthIndices } from "../hooks/useRadixSortDepthIndices";
import type { PlyPacked } from "../hooks/usePlyPacked";

export function SplatScene({
  data,
  controlsGroup = "Splats",
}: {
  data: PlyPacked;
  controlsGroup?: string;
}) {
  const {
    renderAs,
    cutoff,
    alphaDiscard,
    metalness,
    roughness,
    useDepth,
    computeDepthKeys,
    sortByDepth,
    debugDepth,
    // gaussian look
    splatScale,
    maxScreenSpaceSplatSize,
    antialiasCompensation,
    opacityMultiplier,
    showQuadBg,
    quadBgAlpha,
  } = useControls(controlsGroup, {
    renderAs: {
      value: "ellipsoids",
      options: {
        Ellipsoids: "ellipsoids",
        "Gaussian quads": "gaussian",
      },
    },
    cutoff: { value: 1.0, min: 0.05, max: 8.0, step: 0.01 },
    alphaDiscard: { value: 2.0 / 255.0, min: 0.0, max: 0.1, step: 0.0005 },
    roughness: { value: 0.8, min: 0, max: 1, step: 0.01 },
    metalness: { value: 0.0, min: 0, max: 1, step: 0.01 },
    useDepth: { value: true },
    computeDepthKeys: { value: true },
    sortByDepth: { value: true },
    debugDepth: { value: false },

    // Gaussian quad params (used when renderAs === "gaussian")
    splatScale: { value: 1.0, min: 0.1, max: 4.0, step: 0.01 },
    maxScreenSpaceSplatSize: { value: 2048, min: 64, max: 4096, step: 1 },
    antialiasCompensation: { value: true },
    opacityMultiplier: { value: 1.0, min: 0, max: 2, step: 0.01 },
    showQuadBg: { value: false },
    quadBgAlpha: { value: 0.12, min: 0, max: 0.6, step: 0.01 },
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

  const splatShader = useInstancedSplatQuadPlyShader(
    centersBuf,
    covBuf,
    rgbaBuf,
    sortedIndicesBuf
  );

  const splatMaterial = usePlySplatQuadsMaterial({
    shader: splatShader,
    useDepth,
    debugDepth,
    depthKeysBuf,
    splatScale,
    maxScreenSpaceSplatSize,
    antialiasCompensation,
    opacityMultiplier,
    showQuadBg,
    quadBgAlpha,
  });

  const isGaussian = renderAs === "gaussian";

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
        renderOrder={isGaussian ? 10 : 0}
      >
        {isGaussian ? (
          <planeGeometry args={[2, 2]} />
        ) : (
          <sphereGeometry args={[1, 18, 14]} />
        )}
        <primitive
          object={isGaussian ? splatMaterial : ellipsoidMaterial}
          attach="material"
        />
      </instancedMesh>
    </>
  );
}
