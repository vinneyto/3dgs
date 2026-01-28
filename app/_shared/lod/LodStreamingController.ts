import { Matrix4 } from "three";
import type { Camera } from "three";
import type { PlyPacked } from "../hooks/usePlyPacked";
import { getRustWasm } from "../lib/rustWasm";
import { ChunkParsePool, type ChunkParseFormat } from "./ChunkParsePool";
import { resolveLodFileUrls } from "./lodMeta";
import type { LodMeta } from "./types";

export type LodStreamingControllerOptions = {
  meta: LodMeta;
  metaJson: string;
  metaUrl: string;
  lodIndex?: number;
  parseFormat?: ChunkParseFormat;
  workerCount?: number;
  maxRequestsPerTick?: number;
  fetcher?: (url: string) => Promise<ArrayBuffer>;
};

export type LodChunkLoaded = {
  fileIndex: number;
  chunk: PlyPacked;
};

export class LodStreamingController {
  private manager: {
    update_view_proj: (viewProj: Float32Array) => void;
    drain_requests: () => Uint32Array;
    mark_loaded: (fileIndex: number) => void;
    mark_unrequested: (fileIndex: number) => void;
  } | null = null;
  private parserPool: ChunkParsePool;
  private fileUrls: string[];
  private inflightFetches = new Set<number>();
  private fetcher: (url: string) => Promise<ArrayBuffer>;
  private parseFormat: ChunkParseFormat;
  private viewProj = new Matrix4();
  private viewProjArray = new Float32Array(16);

  constructor(options: LodStreamingControllerOptions) {
    const lodIndex = options.lodIndex ?? 0;
    this.fileUrls = resolveLodFileUrls(options.meta, options.metaUrl);

    getRustWasm()
      .then((mod) => {
      const modAny = mod as unknown as {
        LodChunkManager?: new (
          metaJson: string,
          lodIndex: number,
          maxRequestsPerTick: number,
        ) => {
          update_view_proj: (viewProj: Float32Array) => void;
          drain_requests: () => Uint32Array;
          mark_loaded: (fileIndex: number) => void;
          mark_unrequested: (fileIndex: number) => void;
        };
      };
        if (!modAny.LodChunkManager) {
          throw new Error("LodChunkManager wasm export is missing");
        }
        this.manager = new modAny.LodChunkManager(
          options.metaJson,
          lodIndex,
          options.maxRequestsPerTick ?? 32,
        );
      })
      .catch((err) => {
        console.error("[lod] failed to init wasm manager", err);
      });
    this.parserPool = new ChunkParsePool({
      size: options.workerCount,
      format: options.parseFormat,
    });
    this.fetcher = options.fetcher ?? ((url) => fetch(url).then((res) => res.arrayBuffer()));
    this.parseFormat = options.parseFormat ?? "auto";
  }

  tick(camera: Camera, onChunkLoaded: (payload: LodChunkLoaded) => void) {
    if (!this.manager) {
      return;
    }

    this.viewProj.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    this.viewProjArray.set(this.viewProj.elements);
    try {
      this.manager.update_view_proj(this.viewProjArray);
    } catch (err) {
      console.error("[lod] update_view_proj failed", err);
      return;
    }

    const requests = this.manager.drain_requests();
    for (let i = 0; i < requests.length; i += 1) {
      const fileIndex = requests[i];
      if (this.inflightFetches.has(fileIndex)) {
        continue;
      }
      const url = this.fileUrls[fileIndex];
      if (!url) {
        this.manager.mark_unrequested(fileIndex);
        continue;
      }
      this.inflightFetches.add(fileIndex);
      this.fetcher(url)
        .then((buffer) => {
          const bytes = new Uint8Array(buffer);
          this.parserPool.submit(fileIndex, bytes, this.parseFormat);
        })
        .catch((err) => {
          console.error("[lod] fetch failed", url, err);
          this.inflightFetches.delete(fileIndex);
          this.manager?.mark_unrequested(fileIndex);
        });
    }

    for (const result of this.parserPool.drainResults()) {
      this.inflightFetches.delete(result.fileIndex);
      this.manager?.mark_loaded(result.fileIndex);
      onChunkLoaded({ fileIndex: result.fileIndex, chunk: result.chunk });
    }

    for (const error of this.parserPool.drainErrors()) {
      console.error("[lod] parse failed", error);
      this.inflightFetches.delete(error.fileIndex);
      this.manager?.mark_unrequested(error.fileIndex);
    }
  }

  dispose() {
    this.parserPool.dispose();
  }
}
