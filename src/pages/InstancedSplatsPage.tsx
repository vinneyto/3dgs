import { OrbitControls } from "@react-three/drei";
import { button, useControls } from "leva";
import { useEffect, useMemo, useState } from "react";
import {
  DoubleSide,
  MeshBasicNodeMaterial,
  MeshStandardNodeMaterial,
} from "three/webgpu";
import { instancedArray } from "three/tsl";
import { mulberry32, randRange } from "../lib/random";
import { SplatInstanceStruct } from "../tsl/gaussian/gaussianCommon";
import { createInstancedEllipsoidNodes } from "../tsl/gaussian/instancedEllipsoid";
import { createInstancedSplatQuadNodes } from "../tsl/gaussian/instancedSplatQuad";
import { WebGPUCanvasFrame } from "../webgpu/WebGPUCanvasFrame";

/**
 * Fill storage buffer with random SPD covariance via L*L^T.
 * Layout per instance: center(vec4), covA(vec4), covB(vec4), colorOpacity(vec4).
 */
function fillRandomSplats(array: Float32Array, count: number, seed: number) {
  const r = mulberry32(seed | 0);
  let o = 0;

  for (let i = 0; i < count; i++) {
    const cx = randRange(r, -2, 2);
    const cy = randRange(r, -0.2, 1.8);
    const cz = randRange(r, -2, 2);

    // Lower-triangular L with positive diag (controls size) and small off-diagonals (tilt)
    const l11 = randRange(r, 0.03, 0.18);
    const l22 = randRange(r, 0.03, 0.18);
    const l33 = randRange(r, 0.03, 0.18);
    const l21 = randRange(r, -0.08, 0.08);
    const l31 = randRange(r, -0.08, 0.08);
    const l32 = randRange(r, -0.08, 0.08);

    // Cov = L * L^T
    const m11 = l11 * l11;
    const m12 = l11 * l21;
    const m13 = l11 * l31;
    const m22 = l21 * l21 + l22 * l22;
    const m23 = l21 * l31 + l22 * l32;
    const m33 = l31 * l31 + l32 * l32 + l33 * l33;

    const cr = randRange(r, 0.15, 1.0);
    const cg = randRange(r, 0.15, 1.0);
    const cb = randRange(r, 0.15, 1.0);
    const opacity = randRange(r, 0.35, 1.0);

    // center vec4
    array[o++] = cx;
    array[o++] = cy;
    array[o++] = cz;
    array[o++] = 1.0;

    // covA vec4: (m11,m12,m13,_)
    array[o++] = m11;
    array[o++] = m12;
    array[o++] = m13;
    array[o++] = 0.0;

    // covB vec4: (m22,m23,m33,_)
    array[o++] = m22;
    array[o++] = m23;
    array[o++] = m33;
    array[o++] = 0.0;

    // colorOpacity vec4: (r,g,b,a)
    array[o++] = cr;
    array[o++] = cg;
    array[o++] = cb;
    array[o++] = opacity;
  }
}

export function InstancedSplatsPage() {
  const [regenTick, setRegenTick] = useState(0);

  const {
    count,
    seed,
    cutoff,
    opacityMultiplier,
    showQuadBg,
    quadBgAlpha,
    showMeshes,
    showQuads,
  } = useControls("Instanced splats (storage buffer)", {
    count: { value: 256, min: 1, max: 2048, step: 1 },
    seed: { value: 1, min: 1, max: 9999, step: 1 },
    cutoff: { value: 8.0, min: 0.25, max: 25, step: 0.01 },
    opacityMultiplier: { value: 1.0, min: 0, max: 2, step: 0.01 },
    showQuadBg: { value: false },
    quadBgAlpha: { value: 0.12, min: 0, max: 0.6, step: 0.01 },
    showMeshes: { value: true },
    showQuads: { value: true },
    regenerate: button(() => setRegenTick((x) => x + 1)),
  });

  const splats = useMemo(
    () => instancedArray(count, SplatInstanceStruct),
    [count]
  );

  const ellipsoid = useMemo(
    () => createInstancedEllipsoidNodes(splats),
    [splats]
  );
  const quad = useMemo(() => createInstancedSplatQuadNodes(splats), [splats]);

  const ellipsoidMaterial = useMemo(() => {
    const m = new MeshStandardNodeMaterial({
      side: DoubleSide,
      roughness: 0.8,
      metalness: 0.0,
    });
    m.vertexNode = ellipsoid.nodes.vertexNode;
    m.normalNode = ellipsoid.nodes.normalNode;
    return m;
  }, [ellipsoid]);

  const quadMaterial = useMemo(() => {
    const m = new MeshBasicNodeMaterial({ side: DoubleSide });
    m.transparent = true;
    m.depthTest = false;
    m.depthWrite = false;
    m.vertexNode = quad.nodes.vertexNode;
    m.colorNode = quad.nodes.colorNode;
    m.opacityNode = quad.nodes.opacityNode;
    return m;
  }, [quad]);

  useEffect(() => {
    ellipsoid.uniforms.uCutoff.value = cutoff;
    quad.uniforms.uCutoff.value = cutoff;
    quad.uniforms.uParams.value.set(
      opacityMultiplier,
      showQuadBg ? 1.0 : 0.0,
      quadBgAlpha
    );
  }, [ellipsoid, quad, cutoff, opacityMultiplier, showQuadBg, quadBgAlpha]);

  useEffect(() => {
    const attr = splats.value;
    const array = attr.array as Float32Array;
    fillRandomSplats(array, count, seed + regenTick * 101);
    attr.needsUpdate = true;
  }, [splats, count, seed, regenTick]);

  return (
    <div className="page">
      <div className="pageHeader">
        <h1>Instanced splats (storage buffer)</h1>
        <p className="muted">
          Per-instance data comes from a WebGPU storage buffer. The same node
          logic is used for all instances; only the data source changes.
        </p>
      </div>

      <WebGPUCanvasFrame camera={{ position: [4, 3, 4], fov: 50 }}>
        <OrbitControls makeDefault enableDamping />
        <ambientLight intensity={0.35} />
        <hemisphereLight
          args={["#dfe8ff", "#1a1a1a", 0.45]}
          position={[0, 1, 0]}
        />
        <directionalLight position={[4, 6, 3]} intensity={1.2} />
        <gridHelper args={[10, 10]} />

        {showMeshes ? (
          <instancedMesh
            args={[undefined, undefined, count]}
            frustumCulled={false}
          >
            <sphereGeometry args={[1, 18, 14]} />
            <primitive object={ellipsoidMaterial} attach="material" />
          </instancedMesh>
        ) : null}

        {showQuads ? (
          <instancedMesh
            args={[undefined, undefined, count]}
            frustumCulled={false}
            renderOrder={10}
          >
            <planeGeometry args={[2, 2]} />
            <primitive object={quadMaterial} attach="material" />
          </instancedMesh>
        ) : null}
      </WebGPUCanvasFrame>
    </div>
  );
}
