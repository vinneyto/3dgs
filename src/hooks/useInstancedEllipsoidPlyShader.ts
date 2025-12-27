import { useMemo } from "react";
import type { StorageBufferNode } from "three/webgpu";
import {
  createInstancedEllipsoidPlyNodes,
  type InstancedEllipsoidPlyNodes,
} from "../tsl/gaussian/instancedEllipsoidPly";

export function useInstancedEllipsoidPlyShader(
  centersBuf: StorageBufferNode,
  covBuf: StorageBufferNode,
  rgbaBuf: StorageBufferNode
): InstancedEllipsoidPlyNodes {
  return useMemo(
    () => createInstancedEllipsoidPlyNodes(centersBuf, covBuf, rgbaBuf),
    [centersBuf, covBuf, rgbaBuf]
  );
}


