"use client";

import { Stats } from "@react-three/drei";
import { Leva } from "leva";
import type { CSSProperties, ReactNode } from "react";
import { useRef, type RefObject } from "react";
import type { CanvasProps } from "@react-three/fiber";
import { useIsMobile } from "@/app/_shared/hooks/useIsMobile";
import { WebGPUCanvas } from "./WebGPUCanvas";

type WebGPUCanvasFrameProps = Omit<CanvasProps, "className" | "style"> & {
  className?: string;
  style?: CSSProperties;
  clearColor?: number;
  /** Force WebGL backend (WebGPURenderer fallback) for testing/compat. */
  forceWebGL?: boolean;
  /** Show a small FPS counter overlay (inside the canvas). */
  fps?: boolean;
  /** Show Leva panel docked inside the canvas container (top-right). */
  leva?: boolean;
  levaTitle?: ReactNode;
};

export function WebGPUCanvasFrame({
  className = "canvasWrap",
  style,
  clearColor,
  forceWebGL,
  fps = true,
  leva = true,
  levaTitle,
  children,
  ...canvasProps
}: WebGPUCanvasFrameProps) {
  const isMobile = useIsMobile(820);
  const statsParent = useRef<HTMLDivElement>(null!);

  return (
    <div className={className} style={style}>
      <div ref={statsParent} className="statsDock" />
      <WebGPUCanvas
        {...canvasProps}
        clearColor={clearColor}
        forceWebGL={forceWebGL}
        style={{
          width: "100%",
          height: "100%",
        }}
      >
        {children}
        {fps && !isMobile ? (
          <Stats parent={statsParent as unknown as RefObject<HTMLElement>} />
        ) : null}
      </WebGPUCanvas>
      {leva ? (
        <div className="levaDock" suppressHydrationWarning>
          <Leva
            collapsed={isMobile}
            titleBar={levaTitle ? { title: levaTitle } : undefined}
          />
        </div>
      ) : null}
    </div>
  );
}
