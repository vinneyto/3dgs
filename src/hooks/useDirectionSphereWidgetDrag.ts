import type { ThreeEvent } from "@react-three/fiber";
import { useCallback, useRef, useState } from "react";
import { Plane, Vector3 } from "three";

export type UseDirectionSphereWidgetDragOptions = {
  /** Orbit radius of the widget (satellite is clamped onto this sphere). */
  radius: number;
  /** Current direction (will be normalized). */
  direction: Vector3;
  /** Called with normalized direction whenever it changes. */
  onChange: (dir: Vector3) => void;
  /** World-space camera forward direction provider (normalized). */
  getCameraDirection: () => Vector3;
};

export function useDirectionSphereWidgetDrag({
  radius,
  direction,
  onChange,
  getCameraDirection,
}: UseDirectionSphereWidgetDragOptions) {
  const [isDragging, setIsDragging] = useState(false);
  const pointerIdRef = useRef<number | null>(null);
  const planeRef = useRef<Plane>(new Plane());
  const tmp = useRef(new Vector3());

  const beginDrag = useCallback(
    (e: ThreeEvent<PointerEvent>) => {
      e.stopPropagation();
      pointerIdRef.current = e.pointerId;
      setIsDragging(true);

      // Build a plane perpendicular to the camera direction, passing through the current satellite point.
      const camDir = getCameraDirection().normalize();
      const satPos = direction.clone().normalize().multiplyScalar(radius);
      planeRef.current.setFromNormalAndCoplanarPoint(camDir, satPos);

      // Try to capture pointer so we keep receiving events
      const el = e.target as unknown as Element & {
        setPointerCapture?: (id: number) => void;
      };
      el.setPointerCapture?.(e.pointerId);
    },
    [direction, getCameraDirection, radius]
  );

  const updateDrag = useCallback(
    (e: ThreeEvent<PointerEvent>) => {
      if (!isDragging) return;
      if (pointerIdRef.current != null && e.pointerId !== pointerIdRef.current) return;
      e.stopPropagation();

      const hit = e.ray.intersectPlane(planeRef.current, tmp.current);
      if (!hit) return;
      const dir = hit.clone().normalize();
      onChange(dir);
    },
    [isDragging, onChange]
  );

  const endDrag = useCallback((e: ThreeEvent<PointerEvent>) => {
    if (pointerIdRef.current != null && e.pointerId !== pointerIdRef.current) return;
    e.stopPropagation();
    pointerIdRef.current = null;
    setIsDragging(false);

    const el = e.target as unknown as Element & {
      releasePointerCapture?: (id: number) => void;
    };
    el.releasePointerCapture?.(e.pointerId);
  }, []);

  return {
    isDragging,
    bind: {
      onPointerDown: beginDrag,
      onPointerMove: updateDrag,
      onPointerUp: endDrag,
      onPointerCancel: endDrag,
    },
  };
}

