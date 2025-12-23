import type { CSSProperties, ReactNode } from "react";
import { Leva } from "leva";
import type { CanvasProps } from "@react-three/fiber";
import { WebGPUCanvas } from "./WebGPUCanvas";

type WebGPUCanvasFrameProps = Omit<CanvasProps, "className" | "style"> & {
  className?: string;
  style?: CSSProperties;
  clearColor?: number;
  /** Force WebGL backend (WebGPURenderer fallback) for testing/compat. */
  forceWebGL?: boolean;
  /** Show Leva panel docked inside the canvas container (top-right). */
  leva?: boolean;
  levaTitle?: ReactNode;
};

export function WebGPUCanvasFrame({
  className = "canvasWrap",
  style,
  clearColor,
  forceWebGL,
  leva = true,
  levaTitle,
  ...canvasProps
}: WebGPUCanvasFrameProps) {
  return (
    <div className={className} style={style}>
      <WebGPUCanvas
        {...canvasProps}
        clearColor={clearColor}
        forceWebGL={forceWebGL}
        style={{
          width: "100%",
          height: "100%",
        }}
      />
      {leva ? (
        <div className="levaDock">
          <Leva
            collapsed={false}
            titleBar={levaTitle ? { title: levaTitle } : undefined}
          />
        </div>
      ) : null}
    </div>
  );
}
