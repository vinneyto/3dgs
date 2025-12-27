import type { Node, StorageBufferNode } from "three/webgpu";
import {
  bitAnd,
  bitXor,
  div,
  float,
  instanceIndex,
  uint,
  vec3,
} from "three/tsl";

export function createDepthDebugColorOpacity(depthKeys: StorageBufferNode): {
  colorNode: Node;
  opacityNode: Node;
} {
  // depthKeys stores GaussianSplats3D-style sortable keys:
  // key = uint(int(z * 4096)) XOR 0x80000000
  const keyU = depthKeys.element(instanceIndex);
  const signedBits = bitXor(keyU, uint(0x80000000));

  // For visualization we use the low 16 bits (fine variation of z*4096) mapped to 0..1.
  // This is not "physical depth", but a high-contrast view that confirms keys are changing.
  const lo16 = bitAnd(signedBits, uint(0x0000ffff));
  const v = div(float(lo16), 65535.0);
  const colorNode = vec3(v, v, v);
  const opacityNode = float(1.0);

  return { colorNode, opacityNode };
}
