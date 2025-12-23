import {
  Canvas,
  extend,
  type CanvasProps,
  type ThreeToJSXElements,
} from "@react-three/fiber";
import * as THREE from "three/webgpu";

declare module "@react-three/fiber" {
  // eslint-disable-next-line @typescript-eslint/no-empty-object-type
  interface ThreeElements extends ThreeToJSXElements<typeof THREE> {}
}

// Hide R3F WebGPU boilerplate here so demos/pages stay clean.
// Ref: https://r3f.docs.pmnd.rs/tutorials/v9-migration-guide#webgpu
// eslint-disable-next-line @typescript-eslint/no-explicit-any
extend(THREE as any);

type WebGPUCanvasProps = Omit<CanvasProps, "gl"> & {
  clearColor?: number;
  /** Force WebGL backend (WebGPURenderer fallback) for testing/compat. */
  forceWebGL?: boolean;
};

export function WebGPUCanvas({
  clearColor = 0x0b0d12,
  forceWebGL,
  ...props
}: WebGPUCanvasProps) {
  return (
    <Canvas
      {...props}
      gl={async (glProps) => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const renderer = new THREE.WebGPURenderer({
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          ...(glProps as any),
          forceWebGL,
        });
        await renderer.init();
        renderer.setClearColor?.(clearColor, 1);
        return renderer;
      }}
    />
  );
}
