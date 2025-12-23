import { OrbitControls } from "@react-three/drei";
import { button, useControls } from "leva";
import { useEffect, useMemo, useState } from "react";
import {
  DoubleSide,
  DynamicDrawUsage,
  InstancedBufferAttribute,
  MeshBasicNodeMaterial,
  PlaneGeometry,
} from "three/webgpu";
import { instancedBufferAttribute } from "three/tsl";
import { mulberry32, randRange } from "../lib/random";
import { createInstancedBillboardCircleNodes } from "../tsl/billboard/instancedBillboardCircle";
import { WebGPUCanvasFrame } from "../webgpu/WebGPUCanvasFrame";

export function BillboardCirclePage() {
  const [regenTick, setRegenTick] = useState(0);

  const { count, seed, radiusWorld, minSizePx, maxSizePx } = useControls(
    "Billboard circles (instanced)",
    {
      count: { value: 512, min: 1, max: 4096, step: 1 },
      seed: { value: 1, min: 1, max: 9999, step: 1 },
      radiusWorld: { value: 0.06, min: 0.001, max: 0.5, step: 0.001 },
      minSizePx: { value: 3, min: 1, max: 64, step: 1 },
      maxSizePx: { value: 24, min: 1, max: 128, step: 1 },
      regenerate: button(() => setRegenTick((x) => x + 1)),
    }
  );

  const geometry = useMemo(() => {
    const g = new PlaneGeometry(2, 2);
    // Per-instance center (world) and color (rgb)
    const centerAttr = new InstancedBufferAttribute(
      new Float32Array(count * 3),
      3
    );
    centerAttr.setUsage(DynamicDrawUsage);
    g.setAttribute("iCenter", centerAttr);

    const colorAttr = new InstancedBufferAttribute(
      new Float32Array(count * 3),
      3
    );
    colorAttr.setUsage(DynamicDrawUsage);
    g.setAttribute("iColor", colorAttr);
    return g;
  }, [count]);

  const iCenter = useMemo(
    () => geometry.getAttribute("iCenter") as InstancedBufferAttribute,
    [geometry]
  );

  const iColor = useMemo(
    () => geometry.getAttribute("iColor") as InstancedBufferAttribute,
    [geometry]
  );

  const centerWorld3 = useMemo(
    () => instancedBufferAttribute(iCenter),
    [iCenter]
  );
  const color3 = useMemo(() => instancedBufferAttribute(iColor), [iColor]);

  const shader = useMemo(
    () =>
      createInstancedBillboardCircleNodes({
        centerWorld3,
        color3,
        minSizePx,
        maxSizePx,
        radiusWorld,
      }),
    [centerWorld3, color3, minSizePx, maxSizePx, radiusWorld]
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
    // Update uniforms
    shader.uniforms.uMinSizePx.value = minSizePx;
    shader.uniforms.uMaxSizePx.value = maxSizePx;
    shader.uniforms.uRadiusWorld.value = radiusWorld;
  }, [shader, minSizePx, maxSizePx, radiusWorld]);

  useEffect(() => {
    const r = mulberry32((seed + regenTick * 101) | 0);
    const centers = iCenter.array as Float32Array;
    const colors = iColor.array as Float32Array;

    for (let i = 0; i < count; i++) {
      const oc = i * 3;
      centers[oc + 0] = randRange(r, -3, 3);
      centers[oc + 1] = randRange(r, -0.5, 2.0);
      centers[oc + 2] = randRange(r, -3, 3);

      // random per-instance color (rgb)
      colors[oc + 0] = randRange(r, 0.15, 1.0);
      colors[oc + 1] = randRange(r, 0.15, 1.0);
      colors[oc + 2] = randRange(r, 0.15, 1.0);
    }

    iCenter.needsUpdate = true;
    iColor.needsUpdate = true;
  }, [iCenter, iColor, count, seed, regenTick]);

  return (
    <div className="page">
      <div className="pageHeader">
        <h1>Billboard circles (TSL)</h1>
        <p className="muted">
          Vertex: world radius → perspective pixel size (clamped) → expand plane
          in NDC. Fragment: UV circle with discard.
        </p>
      </div>

      <WebGPUCanvasFrame forceWebGL camera={{ position: [4, 3, 4], fov: 50 }}>
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
