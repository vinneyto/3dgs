import { OrbitControls } from "@react-three/drei";
import { useControls } from "leva";
import { useEffect, useMemo, useState } from "react";
import { DoubleSide, MeshStandardNodeMaterial } from "three/webgpu";
import { instancedArray } from "three/tsl";
import { parseSplatPly } from "../loaders/ply";
import { createInstancedEllipsoidPlyNodes } from "../tsl/gaussian/instancedEllipsoidPly";
import { WebGPUCanvasFrame } from "../webgpu/WebGPUCanvasFrame";

const PLY_URL = "/cactus_splat3_30kSteps_142k_splats.ply";

type PlyPacked = {
  count: number;
  center: Float32Array; // 3N
  covariance: Float32Array; // 6N (two vec3 per splat)
  rgba: Uint32Array; // N packed RGBA8
};

export function PlyEllipsoidsPage() {
  const { cutoff, metalness, roughness } = useControls("PLY ellipsoids", {
    cutoff: { value: 1.0, min: 0.05, max: 8.0, step: 0.01 },
    roughness: { value: 0.8, min: 0, max: 1, step: 0.01 },
    metalness: { value: 0.0, min: 0, max: 1, step: 0.01 },
  });

  const [status, setStatus] = useState("idle");
  const [data, setData] = useState<PlyPacked | null>(null);

  useEffect(() => {
    const ac = new AbortController();
    (async () => {
      try {
        setStatus(`fetching: ${PLY_URL}`);
        const res = await fetch(PLY_URL, { signal: ac.signal });
        if (!res.ok)
          throw new Error(`fetch failed: ${res.status} ${res.statusText}`);

        setStatus("reading arrayBuffer…");
        const buf = await res.arrayBuffer();
        const bytes = new Uint8Array(buf);

        setStatus("parsing PLY…");
        const splat = parseSplatPly(bytes);

        const count = splat.count;
        // No repacking for center/covariance: they are already Float32Array(3N) and Float32Array(6N).
        // rgba is already packed Uint32Array(N): r|(g<<8)|(b<<16)|(a<<24)

        console.log("[PLY ellipsoids buffers]", {
          count,
          centerLen: splat.center.length,
          covarianceLen: splat.covariance.length,
          rgbaLen: splat.rgba.length,
        });

        setData({
          count,
          center: splat.center,
          covariance: splat.covariance,
          rgba: splat.rgba,
        });
        setStatus("ready");
      } catch (e) {
        if ((e as any)?.name === "AbortError") return;
        console.error(e);
        setStatus(`error: ${(e as Error)?.message ?? String(e)}`);
      }
    })();

    return () => ac.abort();
  }, []);

  const centersBuf = useMemo(() => {
    if (!data) return null;
    return instancedArray(data.count, "vec3");
  }, [data]);
  const covBuf = useMemo(() => {
    if (!data) return null;
    // 2 vec3 entries per splat => 2N elements
    return instancedArray(data.count * 2, "vec3");
  }, [data]);
  const rgbaBuf = useMemo(() => {
    if (!data) return null;
    return instancedArray(data.count, "uint");
  }, [data]);

  const shader = useMemo(() => {
    if (!centersBuf || !covBuf || !rgbaBuf) return null;
    return createInstancedEllipsoidPlyNodes(centersBuf, covBuf, rgbaBuf);
  }, [centersBuf, covBuf, rgbaBuf]);

  const material = useMemo(() => {
    if (!shader) return null;
    const m = new MeshStandardNodeMaterial({ side: DoubleSide });
    // PERF: disable depth test + blending as requested (fastest, but visually incorrect for overlap).
    m.depthTest = true;
    m.depthWrite = true;
    // m.transparent = false;
    // m.blending = NoBlending;
    m.vertexNode = shader.nodes.vertexNode;
    m.normalNode = shader.nodes.normalNode;
    m.colorNode = shader.nodes.colorNode;
    m.opacityNode = shader.nodes.opacityNode as never;
    return m;
  }, [shader]);

  useEffect(() => {
    if (!data || !centersBuf || !covBuf || !rgbaBuf) return;
    (centersBuf.value.array as Float32Array).set(data.center);
    (covBuf.value.array as Float32Array).set(data.covariance);
    (rgbaBuf.value.array as Uint32Array).set(data.rgba);
    centersBuf.value.needsUpdate = true;
    covBuf.value.needsUpdate = true;
    rgbaBuf.value.needsUpdate = true;
  }, [data, centersBuf, covBuf, rgbaBuf]);

  useEffect(() => {
    if (!shader || !material) return;
    shader.uniforms.uCutoff.value = cutoff;
    material.roughness = roughness;
    material.metalness = metalness;
  }, [shader, material, cutoff, roughness, metalness]);

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

      <WebGPUCanvasFrame
        camera={{ position: [4, 3, 4], fov: 50 }}
        gl={{ antialias: false }}
      >
        <OrbitControls makeDefault enableDamping />
        <ambientLight intensity={0.25} />
        <directionalLight position={[4, 6, 3]} intensity={1.2} />
        <gridHelper args={[10, 10]} />

        {data && material ? (
          <instancedMesh
            args={[undefined, undefined, data.count]}
            frustumCulled={false}
            scale={[1, -1, 1]}
          >
            <sphereGeometry args={[1, 24, 24]} />
            <primitive object={material} attach="material" />
          </instancedMesh>
        ) : null}
      </WebGPUCanvasFrame>
    </div>
  );
}
