import { useCallback, useEffect, useMemo, useRef } from "react";
import {
  BufferAttribute,
  BufferGeometry,
  Line as ThreeLine,
  LineBasicMaterial,
  Quaternion,
  Vector3,
} from "three";
import type { Group } from "three";
import type { ThreeEvent } from "@react-three/fiber";

export type DirectionSphereWidgetProps = {
  /** Radius of the (invisible) sphere that the satellite is constrained to. */
  radius: number;
  /** Satellite sphere radius (visual only). */
  satelliteRadius?: number;
  /** Direction (will be normalized). */
  direction: Vector3 | [number, number, number];
  /** Called whenever direction changes (normalized). */
  onChange?: (dir: Vector3) => void;
  /** Visual helpers */
  showOrbitSphere?: boolean;
  /** Pointer handlers for the satellite mesh (drag logic lives outside). */
  satellitePointerHandlers?: {
    onPointerDown?: (e: ThreeEvent<PointerEvent>) => void;
    onPointerMove?: (e: ThreeEvent<PointerEvent>) => void;
    onPointerUp?: (e: ThreeEvent<PointerEvent>) => void;
    onPointerCancel?: (e: ThreeEvent<PointerEvent>) => void;
  };
};

function toVec3(v: Vector3 | [number, number, number]) {
  return Array.isArray(v) ? new Vector3(v[0], v[1], v[2]) : v.clone();
}

export function DirectionSphereWidget({
  radius,
  satelliteRadius = 0.12,
  direction,
  onChange,
  showOrbitSphere = false,
  satellitePointerHandlers,
}: DirectionSphereWidgetProps) {
  const rotGroupRef = useRef<Group>(null);

  const applyDirection = useCallback(
    (dir: Vector3) => {
      const d = dir.clone().normalize();
      const q = new Quaternion().setFromUnitVectors(new Vector3(1, 0, 0), d);
      rotGroupRef.current?.quaternion.copy(q);
      onChange?.(d);
    },
    [onChange]
  );

  useEffect(() => {
    applyDirection(toVec3(direction));
  }, [applyDirection, direction]);

  const lineGeometry = useMemo(() => {
    const g = new BufferGeometry();
    const positions = new Float32Array([
      0,
      0,
      0, // center
      radius,
      0,
      0, // satellite at +X in local space
    ]);
    g.setAttribute("position", new BufferAttribute(positions, 3));
    return g;
  }, [radius]);

  const lineObj = useMemo(() => {
    const m = new LineBasicMaterial({
      color: 0xffffff,
      depthTest: false,
      depthWrite: false,
    });
    const l = new ThreeLine(lineGeometry, m);
    l.frustumCulled = false;
    l.renderOrder = 999;
    return l;
  }, [lineGeometry]);

  return (
    <group>
      {showOrbitSphere ? (
        <mesh>
          <sphereGeometry args={[radius, 48, 32]} />
          <meshBasicMaterial
            color="#ffffff"
            wireframe
            transparent
            opacity={0.15}
            depthWrite={false}
          />
        </mesh>
      ) : null}

      {/* Rotated “direction” group: line + satellite */}
      <group ref={rotGroupRef}>
        {/* Line from center → satellite, rendered on top */}
        <primitive object={lineObj} />

        <mesh position={[radius, 0, 0]} {...satellitePointerHandlers}>
          <sphereGeometry args={[satelliteRadius, 32, 24]} />
          <meshStandardMaterial
            color="#ff8a3d"
            roughness={0.4}
            metalness={0.0}
            emissive="#000000"
          />
        </mesh>
      </group>
    </group>
  );
}
