import { OrbitControls } from "@react-three/drei";
import { button, useControls } from "leva";
import { useEffect, useMemo, useState } from "react";
import { DoubleSide, MeshBasicNodeMaterial, PlaneGeometry } from "three/webgpu";
import { instancedArray } from "three/tsl";
import { mulberry32, randRange } from "../lib/random";
import {
  BillboardParticleStruct,
  createStorageBillboardCircleNodes,
} from "../tsl/billboard/storageBillboardCircle";
import { WebGPUCanvasFrame } from "../webgpu/WebGPUCanvasFrame";

function fillRandomParticles(array: Float32Array, count: number, seed: number) {
  const r = mulberry32(seed | 0);
  let o = 0;

  for (let i = 0; i < count; i++) {
    // center vec4
    array[o++] = randRange(r, -3, 3);
    array[o++] = randRange(r, -0.5, 2.0);
    array[o++] = randRange(r, -3, 3);
    array[o++] = 1.0;

    // color vec4
    array[o++] = randRange(r, 0.15, 1.0);
    array[o++] = randRange(r, 0.15, 1.0);
    array[o++] = randRange(r, 0.15, 1.0);
    array[o++] = 1.0;
  }
}

export function BillboardCircleStoragePage() {
  const [regenTick, setRegenTick] = useState(0);

  const { count, seed, radiusWorld, minSizePx, maxSizePx } = useControls(
    "Billboard circles (storage buffer)",
    {
      count: { value: 512, min: 1, max: 4096, step: 1 },
      seed: { value: 1, min: 1, max: 9999, step: 1 },
      radiusWorld: { value: 0.06, min: 0.001, max: 0.5, step: 0.001 },
      minSizePx: { value: 3, min: 1, max: 64, step: 1 },
      maxSizePx: { value: 24, min: 1, max: 128, step: 1 },
      regenerate: button(() => setRegenTick((x) => x + 1)),
    }
  );

  const particles = useMemo(
    () => instancedArray(count, BillboardParticleStruct),
    [count]
  );

  const shader = useMemo(
    () => createStorageBillboardCircleNodes(particles),
    [particles]
  );

  const material = useMemo(() => {
    const m = new MeshBasicNodeMaterial({ side: DoubleSide });
    m.transparent = false;
    m.depthTest = true;
    m.depthWrite = true;
    m.vertexNode = shader.nodes.vertexNode;
    m.colorNode = shader.nodes.colorNode;
    m.opacityNode = shader.nodes.opacityNode as never;
    return m;
  }, [shader]);

  useEffect(() => {
    shader.uniforms.uMinSizePx.value = minSizePx;
    shader.uniforms.uMaxSizePx.value = maxSizePx;
    shader.uniforms.uRadiusWorld.value = radiusWorld;
  }, [shader, minSizePx, maxSizePx, radiusWorld]);

  useEffect(() => {
    const attr = particles.value;
    const array = attr.array as Float32Array;
    fillRandomParticles(array, count, seed + regenTick * 101);
    attr.needsUpdate = true;
  }, [particles, count, seed, regenTick]);

  const geometry = useMemo(() => new PlaneGeometry(2, 2), []);

  return (
    <div className="page">
      <div className="pageHeader">
        <h1>Billboard circles (storage buffer)</h1>
        <p className="muted">
          Same shader nodes as the attribute-based demo, but data comes from a
          storage buffer (`instancedArray`) indexed by `instanceIndex`.
        </p>
        <div className="muted">
          File: <code>src/pages/BillboardCircleStoragePage.tsx</code>
        </div>
      </div>

      <WebGPUCanvasFrame camera={{ position: [4, 3, 4], fov: 50 }}>
        <OrbitControls makeDefault enableDamping />
        <ambientLight intensity={0.25} />
        <gridHelper args={[10, 10]} />

        <instancedMesh
          args={[geometry, undefined, count]}
          frustumCulled={false}
        >
          <primitive object={material} attach="material" />
        </instancedMesh>
      </WebGPUCanvasFrame>
    </div>
  );
}
