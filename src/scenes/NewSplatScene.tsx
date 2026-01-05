import { OrbitControls } from "@react-three/drei";
import { useControls } from "leva";
import { useMemo, useRef } from "react";
import type { InstancedMesh } from "three";
import { DoubleSide, MeshBasicNodeMaterial } from "three/webgpu";
import { useDepthKeyCompute } from "../hooks/useDepthKeyCompute";
import { usePlyEllipsoidBuffersFromData } from "../hooks/usePlyEllipsoidBuffers";
import { useRadixSortDepthIndices } from "../hooks/useRadixSortDepthIndices";
import type { PlyPacked } from "../hooks/usePlyPacked";
import { instancedSplat } from "../tsl/gaussian/instancedSplat";

export function NewSplatScene({
  data,
  controlsGroup = "Splats",
}: {
  data: PlyPacked;
  controlsGroup?: string;
}) {
  const {
    computeDepthKeys,
    sortByDepth,
    renderSplats,
    useDepth,
    // params (compile-time for now: changing them rebuilds nodes/material)
    kernel2DSize,
    splatScale,
    maxScreenSpaceSplatSize,
    inverseFocalAdjustment,
  } = useControls(controlsGroup, {
    computeDepthKeys: { value: true },
    sortByDepth: { value: true },
    renderSplats: { value: true },
    useDepth: { value: true },
    kernel2DSize: { value: 0.3, min: 0.0, max: 2.0, step: 0.01 },
    splatScale: { value: 1.0, min: 0.1, max: 4.0, step: 0.01 },
    maxScreenSpaceSplatSize: { value: 2048, min: 64, max: 4096, step: 1 },
    inverseFocalAdjustment: { value: 1.0, min: 0.25, max: 4.0, step: 0.01 },
  });

  // Buffers (same as SplatScene)
  const { centersBuf, covBuf, rgbaBuf } = usePlyEllipsoidBuffersFromData(data);

  // Keep a mesh ref so depth computation can match the same transform path as the rendering scene.
  const meshRef = useRef<InstancedMesh | null>(null);

  // Depth keys + sort (copied from SplatScene)
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

  const shader = useMemo(
    () =>
      instancedSplat({
        centers: centersBuf,
        cov: covBuf,
        rgba: rgbaBuf,
        sortedIndices: sortedIndicesBuf,
        kernel2DSize,
        splatScale,
        maxScreenSpaceSplatSize,
        inverseFocalAdjustment,
      }),
    [
      centersBuf,
      covBuf,
      rgbaBuf,
      sortedIndicesBuf,
      kernel2DSize,
      splatScale,
      maxScreenSpaceSplatSize,
      inverseFocalAdjustment,
    ]
  );

  const material = useMemo(() => {
    const isSorted = !!sortedIndicesBuf;
    const enableDepth = useDepth && !isSorted;
    const m = new MeshBasicNodeMaterial({ side: DoubleSide });
    m.transparent = true;
    m.depthTest = enableDepth;
    m.depthWrite = enableDepth;
    m.vertexNode = shader.positionNode;
    m.colorNode = shader.colorNode;
    m.opacityNode = shader.opacityNode as never;
    return m;
  }, [shader, useDepth, sortedIndicesBuf]);

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
        visible={renderSplats}
        renderOrder={10}
      >
        <planeGeometry args={[2, 2]} />
        <primitive object={material} attach="material" />
      </instancedMesh>
    </>
  );
}
