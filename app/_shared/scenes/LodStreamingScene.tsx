import { OrbitControls } from "@react-three/drei";
import { useFrame, useThree } from "@react-three/fiber";
import { useControls } from "leva";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { InstancedMesh } from "three";
import { DoubleSide, MeshBasicNodeMaterial, NormalBlending } from "three/webgpu";
import { usePlyEllipsoidBuffers } from "@/app/_shared/hooks/usePlyEllipsoidBuffers";
import type { PlyPacked } from "@/app/_shared/hooks/usePlyPacked";
import { instancedSplat, type InstancedSplatCutoffMode } from "@/app/_shared/tsl/gaussian/instancedSplat";
import type { ChunkParseFormat } from "@/app/_shared/lod/ChunkParsePool";
import { LodStreamingController } from "@/app/_shared/lod/LodStreamingController";
import { appendChunkToBuffers } from "@/app/_shared/lod/splatBufferWriter";
import type { LodMeta } from "@/app/_shared/lod/types";

export function LodStreamingScene({
  meta,
  metaJson,
  metaUrl,
  capacity,
  lodIndex = 0,
  parseFormat = "auto",
  workerCount = 3,
  meshScale = [1, 1, 1],
}: {
  meta: LodMeta;
  metaJson: string;
  metaUrl: string;
  capacity: number;
  lodIndex?: number;
  parseFormat?: ChunkParseFormat;
  workerCount?: number;
  meshScale?: [number, number, number];
}) {
  const camera = useThree((s) => s.camera);
  const meshRef = useRef<InstancedMesh | null>(null);
  const controllerRef = useRef<LodStreamingController | null>(null);
  const loadedCountRef = useRef(0);
  const [loadedCount, setLoadedCount] = useState(0);

  const { centersBuf, covBuf, rgbaBuf } = usePlyEllipsoidBuffers(capacity);

  const {
    renderSplats,
    kernel2DSize,
    splatScale,
    maxScreenSpaceSplatSize,
    inverseFocalAdjustment,
    cutoffMode,
    opacityMultiplier,
    encodeLinear,
  } = useControls("LOD streaming", {
    renderSplats: { value: true },
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
  });

  const shader = useMemo(
    () =>
      instancedSplat({
        centers: centersBuf,
        cov: covBuf,
        rgba: rgbaBuf,
        enableSh: false,
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
    m.depthTest = true;
    m.depthWrite = false;
    m.blending = NormalBlending;
    m.vertexNode = shader.positionNode;
    m.colorNode = shader.colorNode;
    m.opacityNode = shader.opacityNode;
    return m;
  }, [shader]);

  const appendChunk = useCallback(
    ({ chunk }: { chunk: PlyPacked }) => {
      const result = appendChunkToBuffers(
        {
          centers: centersBuf,
          covariances: covBuf,
          rgba: rgbaBuf,
        },
        chunk,
        loadedCountRef.current,
        capacity,
      );
      if (!result) {
        console.warn("[lod] capacity exceeded, skipping chunk");
        return;
      }
      loadedCountRef.current = result.nextBase;
      setLoadedCount(result.nextBase);
    },
    [capacity, centersBuf, covBuf, rgbaBuf],
  );

  useEffect(() => {
    const controller = new LodStreamingController({
      meta,
      metaJson,
      metaUrl,
      lodIndex,
      parseFormat,
      workerCount,
    });
    loadedCountRef.current = 0;
    setLoadedCount(0);
    controllerRef.current = controller;
    return () => {
      controller.dispose();
      controllerRef.current = null;
    };
  }, [meta, metaJson, metaUrl, lodIndex, parseFormat, workerCount]);

  useFrame(() => {
    const controller = controllerRef.current;
    if (!controller) return;
    controller.tick(camera, appendChunk);
  });

  useEffect(() => {
    if (meshRef.current) {
      meshRef.current.count = loadedCount;
    }
  }, [loadedCount]);

  return (
    <>
      <OrbitControls makeDefault enableDamping />
      <ambientLight intensity={0.25} />
      <directionalLight position={[4, 6, 3]} intensity={1.2} />
      <gridHelper args={[10, 10]} />

      <instancedMesh
        args={[undefined, undefined, capacity]}
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
