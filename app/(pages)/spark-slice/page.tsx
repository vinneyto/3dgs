"use client";

import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import { OrbitControls } from "@react-three/drei";
import { Canvas, useThree } from "@react-three/fiber";
import { Leva, useControls } from "leva";
import { useEffect, useMemo, useRef, useState } from "react";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import {
  Color,
  DoubleSide,
  Euler,
  MathUtils,
  Quaternion,
  Vector3,
  WebGLRenderer,
} from "three";
import { useIsMobile } from "@/app/_shared/hooks/useIsMobile";
import { patchSparkSliceShader } from "@/app/_shared/lib/patchSparkSliceShader";

const PLY_URL = "/cactus_splat3_30kSteps_142k_splats.ply";
const CAMERA_DIRECTION = new Vector3(1.3, 0.85, 1.8).normalize();
const DEFAULT_CENTER = new Vector3(0, 0, 0);
const DEFAULT_PLANE_NORMAL = new Vector3(0, 0, 1);

type SparkSliceControls = {
  planeOffset: number;
  planeTiltX: number;
  planeTiltY: number;
  planeTiltZ: number;
  highlightWidth: number;
  highlightStrength: number;
  highlightColor: string;
  showPlane: boolean;
  planeOpacity: number;
  planeColor: string;
  maxStdDev: number;
  blurAmount: number;
  falloff: number;
  focalAdjustment: number;
  encodeLinear: boolean;
};

type BoundsInfo = {
  center: Vector3;
  size: Vector3;
};

function SparkSliceScene({
  controls,
  onStatusChange,
}: {
  controls: SparkSliceControls;
  onStatusChange: (status: string) => void;
}) {
  const gl = useThree((state) => state.gl);
  const camera = useThree((state) => state.camera);
  const orbitRef = useRef<OrbitControlsImpl | null>(null);
  const [bounds, setBounds] = useState<BoundsInfo | null>(null);

  const { spark, sliceUniforms } = useMemo(() => {
    const sparkRenderer = new SparkRenderer({
      renderer: gl as WebGLRenderer,
      maxStdDev: Math.sqrt(8),
      blurAmount: 0.3,
      falloff: 1.0,
      focalAdjustment: 1.0,
    });

    sparkRenderer.renderOrder = 10;

    return {
      spark: sparkRenderer,
      sliceUniforms: patchSparkSliceShader(sparkRenderer),
    };
  }, [gl]);

  const splats = useMemo(() => new SplatMesh({ url: PLY_URL }), []);

  useEffect(() => {
    let cancelled = false;
    onStatusChange(`loading Spark scene from ${PLY_URL}…`);

    void splats.initialized
      .then((mesh) => {
        if (cancelled) return;

        const box = mesh.getBoundingBox();
        const center = box.getCenter(new Vector3());
        const size = box.getSize(new Vector3());
        const splatCount = mesh.packedSplats.numSplats.toLocaleString();

        setBounds({ center, size });
        onStatusChange(`ready • ${splatCount} splats`);
      })
      .catch((error) => {
        if (cancelled) return;
        const message =
          error instanceof Error ? error.message : String(error);
        onStatusChange(`error • ${message}`);
      });

    return () => {
      cancelled = true;
      splats.dispose();
    };
  }, [onStatusChange, splats]);

  useEffect(() => {
    if (!bounds) return;

    const target = bounds.center;
    const radius = Math.max(bounds.size.length() * 0.5, 1.2);
    const cameraPosition = target
      .clone()
      .add(CAMERA_DIRECTION.clone().multiplyScalar(radius * 1.9));

    camera.position.copy(cameraPosition);

    if ("isPerspectiveCamera" in camera && camera.isPerspectiveCamera) {
      camera.near = Math.max(radius / 200, 0.01);
      camera.far = radius * 20;
      camera.updateProjectionMatrix();
    }

    orbitRef.current?.target.copy(target);
    orbitRef.current?.update();
  }, [bounds, camera]);

  const planeNormal = useMemo(
    () =>
      DEFAULT_PLANE_NORMAL.clone()
        .applyEuler(
          new Euler(
            MathUtils.degToRad(controls.planeTiltX),
            MathUtils.degToRad(controls.planeTiltY),
            MathUtils.degToRad(controls.planeTiltZ),
            "XYZ",
          ),
        )
        .normalize(),
    [
      controls.planeTiltX,
      controls.planeTiltY,
      controls.planeTiltZ,
    ],
  );

  const planeCenter = useMemo(() => {
    const baseCenter = bounds?.center ?? DEFAULT_CENTER;
    return baseCenter
      .clone()
      .addScaledVector(planeNormal, controls.planeOffset);
  }, [bounds, controls.planeOffset, planeNormal]);

  const planeQuaternion = useMemo(
    () =>
      new Quaternion().setFromUnitVectors(
        DEFAULT_PLANE_NORMAL,
        planeNormal,
      ),
    [planeNormal],
  );

  const planeSize = useMemo(() => {
    const size = bounds?.size;
    if (!size) return 4;
    return Math.max(size.x, size.y, size.z) * 1.7;
  }, [bounds]);

  useEffect(() => {
    spark.maxStdDev = controls.maxStdDev;
    spark.blurAmount = controls.blurAmount;
    spark.falloff = controls.falloff;
    spark.focalAdjustment = controls.focalAdjustment;
    spark.viewpoint.encodeLinear = controls.encodeLinear;

    sliceUniforms.sliceHighlightEnabled.value = true;
    sliceUniforms.slicePlaneOrigin.value.copy(planeCenter);
    sliceUniforms.slicePlaneNormal.value.copy(planeNormal);
    sliceUniforms.sliceHighlightWidth.value = controls.highlightWidth;
    sliceUniforms.sliceHighlightStrength.value = controls.highlightStrength;
    sliceUniforms.sliceHighlightColor.value
      .set(new Color(controls.highlightColor))
      .convertSRGBToLinear();
  }, [
    controls.blurAmount,
    controls.encodeLinear,
    controls.falloff,
    controls.focalAdjustment,
    controls.highlightColor,
    controls.highlightStrength,
    controls.highlightWidth,
    controls.maxStdDev,
    planeCenter,
    planeNormal,
    sliceUniforms,
    spark,
  ]);

  return (
    <>
      <color attach="background" args={["#0b0d12"]} />
      <OrbitControls ref={orbitRef} makeDefault enableDamping />
      <gridHelper args={[10, 10, "#2b3342", "#1b2230"]} />
      <axesHelper args={[1.25]} />

      {controls.showPlane ? (
        <mesh
          position={planeCenter}
          quaternion={planeQuaternion}
          renderOrder={3}
        >
          <planeGeometry args={[planeSize, planeSize]} />
          <meshBasicMaterial
            color={controls.planeColor}
            transparent
            opacity={controls.planeOpacity}
            side={DoubleSide}
            depthWrite={false}
            depthTest={false}
          />
        </mesh>
      ) : null}

      <primitive object={spark} />
      <primitive object={splats} />
    </>
  );
}

export default function SparkSlicePage() {
  const isMobile = useIsMobile(820);
  const [status, setStatus] = useState("booting Spark demo…");

  const controls = useControls("Spark slice", {
    planeOffset: { value: 0.0, min: -2.0, max: 2.0, step: 0.01 },
    planeTiltX: { value: -10, min: -90, max: 90, step: 1 },
    planeTiltY: { value: 18, min: -90, max: 90, step: 1 },
    planeTiltZ: { value: 0, min: -180, max: 180, step: 1 },
    highlightWidth: { value: 0.08, min: 0.01, max: 0.4, step: 0.005 },
    highlightStrength: { value: 2.25, min: 0.0, max: 6.0, step: 0.05 },
    highlightColor: "#ff8a00",
    showPlane: true,
    planeOpacity: { value: 0.16, min: 0.0, max: 0.6, step: 0.01 },
    planeColor: "#f6ad55",
    maxStdDev: {
      value: Math.sqrt(8),
      min: Math.sqrt(4),
      max: Math.sqrt(10),
      step: 0.01,
    },
    blurAmount: { value: 0.3, min: 0.0, max: 1.0, step: 0.01 },
    falloff: { value: 1.0, min: 0.0, max: 1.0, step: 0.01 },
    focalAdjustment: { value: 1.0, min: 0.5, max: 2.0, step: 0.01 },
    encodeLinear: true,
  }) as SparkSliceControls;

  return (
    <div className="page">
      <div className="pageHeader">
        <h1>Spark slice highlight (cactus)</h1>
        <p className="muted">
          Spark renders the cactus PLY from <code>public/</code>, and its splat
          shader is locally patched so each sprite computes a plane distance and
          adds a brighter highlight near the slicing plane.
        </p>
        <div className="muted">
          File: <code>app/(pages)/spark-slice/page.tsx</code>
        </div>
        <div className="muted">
          Status: <code>{status}</code>
        </div>
      </div>

      <div className="canvasWrap">
        <Canvas
          camera={{ position: [3.5, 2.5, 4.5], fov: 45, near: 0.01, far: 100 }}
          gl={{ antialias: false, powerPreference: "high-performance" }}
          dpr={[1, 1.5]}
          onCreated={({ gl }) => {
            gl.setClearColor("#0b0d12", 1);
          }}
        >
          <SparkSliceScene controls={controls} onStatusChange={setStatus} />
        </Canvas>

        <div
          style={{
            position: "absolute",
            left: 12,
            bottom: 12,
            zIndex: 5,
            padding: "8px 10px",
            borderRadius: 10,
            background: "rgba(11, 13, 18, 0.72)",
            border: "1px solid rgba(255, 255, 255, 0.08)",
            color: "rgba(255, 255, 255, 0.88)",
            fontSize: 12,
            lineHeight: 1.45,
            maxWidth: 360,
            pointerEvents: "none",
          }}
        >
          Move the plane with <code>planeOffset</code> and rotate it via{" "}
          <code>planeTilt*</code>. The orange band is computed in Spark&apos;s
          fragment shader from a per-fragment plane distance estimate.
        </div>

        <div className="levaDock" suppressHydrationWarning>
          <Leva
            collapsed={isMobile}
            titleBar={{ title: "Spark slice controls" }}
          />
        </div>
      </div>
    </div>
  );
}
