import { Box3, Frustum, Matrix4, Vector3 } from "three";
import type { Camera } from "three";
import type { LodChunkNode } from "./types";

export enum FileState {
  Unrequested = 0,
  Requested = 1,
  Loaded = 2,
}

export type LodChunkManagerOptions = {
  maxRequestsPerTick?: number;
};

type NodeEntry = LodChunkNode & {
  box: Box3;
};

export class LodChunkManager {
  private nodes: NodeEntry[];
  private fileStates: Uint8Array;
  private pendingRequests: number[] = [];
  private frustum = new Frustum();
  private viewProj = new Matrix4();
  private maxRequestsPerTick: number;

  constructor(nodes: LodChunkNode[], fileCount: number, options: LodChunkManagerOptions = {}) {
    this.nodes = nodes.map((node) => ({
      ...node,
      box: new Box3(
        new Vector3(node.min[0], node.min[1], node.min[2]),
        new Vector3(node.max[0], node.max[1], node.max[2]),
      ),
    }));
    this.fileStates = new Uint8Array(fileCount);
    this.maxRequestsPerTick = options.maxRequestsPerTick ?? 32;
  }

  updateCamera(camera: Camera) {
    this.viewProj.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    this.frustum.setFromProjectionMatrix(this.viewProj);

    let emitted = 0;
    for (const node of this.nodes) {
      if (emitted >= this.maxRequestsPerTick) {
        break;
      }
      if (!this.frustum.intersectsBox(node.box)) {
        continue;
      }
      const state = this.fileStates[node.fileIndex];
      if (state !== FileState.Unrequested) {
        continue;
      }
      this.fileStates[node.fileIndex] = FileState.Requested;
      this.pendingRequests.push(node.fileIndex);
      emitted += 1;
    }
  }

  drainRequests(): number[] {
    if (this.pendingRequests.length === 0) {
      return [];
    }
    const requests = this.pendingRequests.slice();
    this.pendingRequests.length = 0;
    return requests;
  }

  markLoaded(fileIndex: number) {
    this.fileStates[fileIndex] = FileState.Loaded;
  }

  markUnrequested(fileIndex: number) {
    this.fileStates[fileIndex] = FileState.Unrequested;
  }

  getFileState(fileIndex: number): FileState {
    return this.fileStates[fileIndex] ?? FileState.Unrequested;
  }
}
