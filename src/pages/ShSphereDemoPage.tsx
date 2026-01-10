import { OrbitControls } from "@react-three/drei";
import { useThree } from "@react-three/fiber";
import { useControls } from "leva";
import { useCallback, useEffect, useMemo, useState } from "react";
import { ArrowHelper, Material, Vector3 } from "three";
import { DoubleSide, MeshBasicNodeMaterial } from "three/webgpu";
import { DirectionSphereWidget } from "../components/DirectionSphereWidget";
import { useDirectionSphereWidgetDrag } from "../hooks/useDirectionSphereWidgetDrag";
import { createShSphereL1Nodes } from "../tsl/sh/shSphereL1";
import { WebGPUCanvasFrame } from "../webgpu/WebGPUCanvasFrame";

function SceneContent({
  dir,
  onChange,
  mainSphereMaterial,
}: {
  dir: Vector3;
  onChange: (d: Vector3) => void;
  mainSphereMaterial: MeshBasicNodeMaterial;
}) {
  const { camera } = useThree();

  const widgetDrag = useDirectionSphereWidgetDrag({
    radius: 2.0,
    direction: dir,
    onChange,
    getCameraDirection: () => {
      const v = new Vector3();
      camera.getWorldDirection(v);
      return v;
    },
  });

  return (
    <>
      <OrbitControls
        makeDefault
        enableDamping
        enabled={!widgetDrag.isDragging}
      />
      <ambientLight intensity={0.35} />
      <directionalLight position={[4, 6, 3]} intensity={1.2} />
      <gridHelper args={[10, 10]} />
      <axesHelper args={[2]} />

      {/* Main sphere: SH (L1) visualized on the surface */}
      <mesh>
        <sphereGeometry args={[1, 64, 48]} />
        <primitive object={mainSphereMaterial} attach="material" />
      </mesh>

      {/* Satellite widget: constrained to radius > main sphere */}
      <DirectionSphereWidget
        radius={2.0}
        satelliteRadius={0.12}
        direction={dir}
        showOrbitSphere={false}
        satellitePointerHandlers={widgetDrag.bind}
      />
    </>
  );
}

export function ShSphereDemoPage() {
  const {
    c0,
    cY,
    cZ,
    cX,
    amp0,
    ampY,
    ampZ,
    ampX,
    signedView,
    scale,
    bias,
    cameraDist,
  } = useControls("SH (L1)", {
    c0: { value: "#ffffff" }, // DC
    cY: { value: "#00ff66" }, // Y term
    cZ: { value: "#3399ff" }, // Z term
    cX: { value: "#ff3344" }, // X term

    amp0: { value: 1.0, min: -3, max: 3, step: 0.01 },
    ampY: { value: 1.0, min: -3, max: 3, step: 0.01 },
    ampZ: { value: 1.0, min: -3, max: 3, step: 0.01 },
    ampX: { value: 1.0, min: -3, max: 3, step: 0.01 },

    signedView: { value: false },
    scale: { value: 1.0, min: 0.0, max: 5.0, step: 0.01 },
    bias: { value: 0.0, min: -1.0, max: 1.0, step: 0.01 },

    cameraDist: { value: 2.0, min: 1.1, max: 10.0, step: 0.01 },
  });

  const [dir, setDir] = useState(() => new Vector3(1, 0, 0));

  const onChange = useCallback((d: Vector3) => {
    // store a clone so React state changes
    setDir(d.clone());
  }, []);

  const [demo] = useState(() => createShSphereL1Nodes());
  const mainSphereMaterial = useMemo(() => {
    const m = new MeshBasicNodeMaterial({ side: DoubleSide });
    m.transparent = false;
    m.depthTest = true;
    m.depthWrite = true;
    m.colorNode = demo.nodes.colorNode;
    m.opacityNode = demo.nodes.opacityNode as never;
    return m;
  }, [demo]);

  useEffect(() => {
    demo.uniforms.uC0.value.set(c0);
    demo.uniforms.uC1.value.set(cY);
    demo.uniforms.uC2.value.set(cZ);
    demo.uniforms.uC3.value.set(cX);

    demo.uniforms.uAmp.value.set(amp0, ampY, ampZ, ampX);
    demo.uniforms.uCameraDir.value.copy(dir.clone().normalize());
    demo.uniforms.uCameraDist.value = cameraDist;
    demo.uniforms.uSignedView.value = signedView ? 1.0 : 0.0;
    demo.uniforms.uScale.value = scale;
    demo.uniforms.uBias.value = bias;
  }, [
    demo,
    c0,
    cY,
    cZ,
    cX,
    amp0,
    ampY,
    ampZ,
    ampX,
    dir,
    cameraDist,
    signedView,
    scale,
    bias,
  ]);

  const arrow = useMemo(() => {
    const a = new ArrowHelper(
      new Vector3(1, 0, 0),
      new Vector3(0, 0, 0),
      2.0,
      0x66ccff,
      0.15,
      0.08
    );
    a.renderOrder = 998;
    (a.line.material as Material).depthTest = false;
    (a.cone.material as Material).depthTest = false;
    return a;
  }, []);

  useEffect(() => {
    arrow.setDirection(dir.clone().normalize());
  }, [arrow, dir]);

  return (
    <div className="page">
      <div className="pageHeader">
        <h1>SH on sphere (WIP)</h1>
        <p className="muted">
          Step 1: drag the satellite along the orbit sphere to define a
          direction. Next step: visualize spherical harmonic lobes on the main
          sphere using that direction.
        </p>
      </div>

      <WebGPUCanvasFrame
        forceWebGL
        camera={{ position: [3.2, 2.2, 3.2], fov: 50 }}
      >
        <SceneContent
          dir={dir}
          onChange={onChange}
          mainSphereMaterial={mainSphereMaterial}
        />

        {/* Extra direction indicator (arrow), drawn on top */}
        <primitive object={arrow} />
      </WebGPUCanvasFrame>
    </div>
  );
}
