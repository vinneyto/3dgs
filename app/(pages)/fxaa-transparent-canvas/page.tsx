"use client";

import { Canvas, type RootState } from "@react-three/fiber";
import { Leva, useControls } from "leva";
import { useEffect, useRef } from "react";
import { FxaaTransparentDemo } from "@/app/_shared/lib/fxaaTransparentDemo";

export default function FxaaTransparentCanvasPage() {
  const demoRef = useRef<FxaaTransparentDemo | null>(null);

  const { fxaaEnabled } = useControls("FXAA demo", {
    fxaaEnabled: { value: true, label: "Enable FXAA" },
  });

  useEffect(() => {
    return () => {
      demoRef.current?.detach();
      demoRef.current = null;
    };
  }, []);

  const handleCreated = (state: RootState) => {
    const container = state.gl.domElement.parentElement;
    if (!container) return;

    demoRef.current?.detach();

    state.gl.setClearColor(0x000000, 0);

    const demo = new FxaaTransparentDemo(state.gl);
    demo.setFxaaEnabled(fxaaEnabled);
    demo.attach(container);
    demoRef.current = demo;
  };

  useEffect(() => {
    demoRef.current?.setFxaaEnabled(fxaaEnabled);
  }, [fxaaEnabled]);

  return (
    <div className="page">
      <div className="pageHeader">
        <h1>Canvas transparency + FXAA (three.js only scene/passes)</h1>
        <p className="muted">
          Scene + composer + FXAA are implemented in plain three.js TS code.
          React Three Fiber is used only for &lt;Canvas /&gt; mount lifecycle.
        </p>
        <div className="muted">
          File: <code>app/(pages)/fxaa-transparent-canvas/page.tsx</code>
        </div>
      </div>

      <div className="canvasWrap checkerboardUnderlay">
        <Canvas
          gl={{
            antialias: false,
            alpha: true,
            powerPreference: "high-performance",
            premultipliedAlpha: true,
          }}
          frameloop="never"
          onCreated={handleCreated}
          onPointerMissed={() => {
            // no-op: keeps pointer handling explicit for this bare-canvas demo
          }}
        />
        <div className="levaDock">
          <Leva
            fill
            flat
            titleBar={{
              drag: true,
              filter: false,
            }}
          />
        </div>
      </div>
    </div>
  );
}
