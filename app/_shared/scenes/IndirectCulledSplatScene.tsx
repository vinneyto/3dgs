import { OrbitControls } from "@react-three/drei";
import { button, useControls } from "leva";
import { useCallback, useEffect, useMemo, useRef } from "react";
import { useThree } from "@react-three/fiber";
import type { InstancedMesh } from "three";
import {
  DoubleSide,
  MeshBasicNodeMaterial,
  NormalBlending,
} from "three/webgpu";
import type { PlyPacked } from "@/app/_shared/hooks/usePlyPacked";
import { usePlyEllipsoidBuffersFromData } from "@/app/_shared/hooks/usePlyEllipsoidBuffers";
import { useSplatIndirectCullSort } from "@/app/_shared/hooks/useSplatIndirectCullSort";
import {
  instancedSplat,
  type InstancedSplatCutoffMode,
} from "@/app/_shared/tsl/gaussian/instancedSplat";

export function IndirectCulledSplatScene({
  data,
  controlsGroup = "GPU Culling",
  meshScale = [1, -1, 1],
  disableShBuffers = false,
}: {
  data: PlyPacked;
  controlsGroup?: string;
  meshScale?: [number, number, number];
  disableShBuffers?: boolean;
}) {
  const camera = useThree((s) => s.camera);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const orbitRef = useRef<any>(null);

  const cameraZRef = useRef(-7);
  const orbitTargetZRef = useRef(-10);

  const applyCamera = useCallback(() => {
    const targetZ = orbitTargetZRef.current;
    const camZ = cameraZRef.current;
    const target = { x: 0, y: 0, z: targetZ };

    camera.position.set(0, 0, camZ);
    camera.lookAt(target.x, target.y, target.z);

    const controls = orbitRef.current;
    if (controls?.target?.set) {
      controls.target.set(target.x, target.y, target.z);
      controls.update?.();
    }
  }, [camera]);

  const {
    fov,
    cameraZ,
    orbitTargetZ,
    enableCulling,
    enableSorting,
    renderSplats,
    enableSh,
    // shading params (compile-time-ish)
    kernel2DSize,
    splatScale,
    maxScreenSpaceSplatSize,
    inverseFocalAdjustment,
    cutoffMode,
    opacityMultiplier,
    encodeLinear,
    shDebugColorOnly,
    debugWorldViewDir,
  } = useControls(controlsGroup, {
    fov: { value: 50, min: 10, max: 120, step: 1 },
    cameraZ: { value: -7, min: -100, max: 100, step: 0.1 },
    orbitTargetZ: { value: -10, min: -200, max: 200, step: 0.1 },
    applyCameraBtn: button(() => applyCamera()),
    enableCulling: { value: true },
    enableSorting: { value: true },
    renderSplats: { value: true },
    enableSh: { value: !disableShBuffers },
    kernel2DSize: { value: 0.3, min: 0.0, max: 2.0, step: 0.01 },
    splatScale: { value: 1.0, min: 0.1, max: 4.0, step: 0.01 },
    maxScreenSpaceSplatSize: { value: 2048, min: 64, max: 4096, step: 1 },
    inverseFocalAdjustment: { value: 1.0, min: 0.25, max: 4.0, step: 0.01 },
    cutoffMode: {
      value: "opacity",
      options: { Fixed8: "fixed", "Opacity-derived": "opacity" },
    },
    opacityMultiplier: { value: 1.0, min: 0.0, max: 2.0, step: 0.01 },
    encodeLinear: { value: true },
    shDebugColorOnly: { value: false },
    debugWorldViewDir: { value: false },
  });

  useEffect(() => {
    cameraZRef.current = cameraZ;
  }, [cameraZ]);

  useEffect(() => {
    orbitTargetZRef.current = orbitTargetZ;
  }, [orbitTargetZ]);

  useEffect(() => {
    if ("isPerspectiveCamera" in camera && camera.isPerspectiveCamera) {
      camera.fov = fov;
      camera.updateProjectionMatrix();
    }
  }, [camera, fov]);

  const splatData = disableShBuffers
    ? {
        ...data,
        shCoeffsL1: undefined,
        shCoeffsL2Packed: undefined,
        shCoeffsL2Scale: undefined,
        shCoeffsL3Packed: undefined,
        shCoeffsL3Scale: undefined,
        shDegree: 0,
      }
    : data;

  const {
    centersBuf,
    covBuf,
    rgbaBuf,
    shCoeffsL1Buf,
    shCoeffsL2Buf,
    shCoeffsL2Scale,
    shDegree,
  } = usePlyEllipsoidBuffersFromData(splatData);

  const meshRef = useRef<InstancedMesh | null>(null);

  const { sortedSplatIndices } = useSplatIndirectCullSort({
    enabled: enableCulling && enableSorting,
    centersBuf,
    count: data.count,
    meshRef,
  });

  const shader = useMemo(
    () =>
      instancedSplat({
        centers: centersBuf,
        cov: covBuf,
        rgba: rgbaBuf,
        shCoeffsL1: shCoeffsL1Buf,
        shCoeffsL2: shCoeffsL2Buf,
        shCoeffsL2Scale,
        shDegree,
        enableSh,
        shDebugColorOnly,
        debugWorldViewDir,
        // IMPORTANT: this buffer maps instanceIndex -> original splat index.
        sortedIndices: sortedSplatIndices,
        kernel2DSize,
        splatScale,
        maxScreenSpaceSplatSize,
        inverseFocalAdjustment,
        cutoffMode: cutoffMode as InstancedSplatCutoffMode,
        opacityMultiplier,
        encodeLinear,
      }),
    [
      centersBuf,
      covBuf,
      rgbaBuf,
      shCoeffsL1Buf,
      shCoeffsL2Buf,
      shCoeffsL2Scale,
      shDegree,
      enableSh,
      shDebugColorOnly,
      debugWorldViewDir,
      sortedSplatIndices,
      kernel2DSize,
      splatScale,
      maxScreenSpaceSplatSize,
      inverseFocalAdjustment,
      cutoffMode,
      opacityMultiplier,
      encodeLinear,
    ],
  );

  const material = useMemo(() => {
    const m = new MeshBasicNodeMaterial({ side: DoubleSide });
    m.transparent = true;
    // With explicit back-to-front sorting, don't use depth buffer to avoid popping/z-fighting.
    m.depthTest = false;
    m.depthWrite = false;
    m.blending = NormalBlending;
    m.vertexNode = shader.positionNode;
    m.colorNode = shader.colorNode;
    m.opacityNode = shader.opacityNode;
    return m;
  }, [shader]);

  return (
    <>
      <OrbitControls ref={orbitRef} makeDefault enableDamping />
      <ambientLight intensity={0.25} />
      <directionalLight position={[4, 6, 3]} intensity={1.2} />
      <gridHelper args={[10, 10]} />

      <instancedMesh
        args={[undefined, undefined, data.count]}
        frustumCulled={false}
        scale={meshScale}
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

