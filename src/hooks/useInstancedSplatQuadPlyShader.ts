import { useMemo } from "react";
import type { StorageBufferNode } from "three/webgpu";
import {
  createInstancedSplatQuadPlyNodes,
  type InstancedSplatQuadPlyNodes,
} from "../tsl/gaussian/instancedSplatQuadPly";

export function useInstancedSplatQuadPlyShader(
  centersBuf: StorageBufferNode,
  covBuf: StorageBufferNode,
  rgbaBuf: StorageBufferNode,
  sortedIndicesBuf?: StorageBufferNode | null
): InstancedSplatQuadPlyNodes {
  return useMemo(
    () =>
      createInstancedSplatQuadPlyNodes(
        centersBuf,
        covBuf,
        rgbaBuf,
        sortedIndicesBuf
      ),
    [centersBuf, covBuf, rgbaBuf, sortedIndicesBuf]
  );
}


