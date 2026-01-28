import type { Camera } from "three";
import type { PlyPacked } from "../hooks/usePlyPacked";
import { ChunkParsePool, type ChunkParseFormat } from "./ChunkParsePool";
import { LodChunkManager } from "./LodChunkManager";
import { flattenLodMeta, resolveLodFileUrls, selectLodNodes } from "./lodMeta";
import type { LodMeta } from "./types";

export type LodStreamingControllerOptions = {
  meta: LodMeta;
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
  private manager: LodChunkManager;
  private parserPool: ChunkParsePool;
  private fileUrls: string[];
  private inflightFetches = new Set<number>();
  private fetcher: (url: string) => Promise<ArrayBuffer>;
  private parseFormat: ChunkParseFormat;

  constructor(options: LodStreamingControllerOptions) {
    const lodIndex = options.lodIndex ?? 0;
    const leaves = flattenLodMeta(options.meta);
    const nodes = selectLodNodes(leaves, lodIndex);
    this.fileUrls = resolveLodFileUrls(options.meta, options.metaUrl);

    this.manager = new LodChunkManager(nodes, this.fileUrls.length, {
      maxRequestsPerTick: options.maxRequestsPerTick,
    });
    this.parserPool = new ChunkParsePool({
      size: options.workerCount,
      format: options.parseFormat,
    });
    this.fetcher = options.fetcher ?? ((url) => fetch(url).then((res) => res.arrayBuffer()));
    this.parseFormat = options.parseFormat ?? "auto";
  }

  tick(camera: Camera, onChunkLoaded: (payload: LodChunkLoaded) => void) {
    this.manager.updateCamera(camera);

    const requests = this.manager.drainRequests();
    for (const fileIndex of requests) {
      if (this.inflightFetches.has(fileIndex)) {
        continue;
      }
      const url = this.fileUrls[fileIndex];
      if (!url) {
        this.manager.markUnrequested(fileIndex);
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
          this.manager.markUnrequested(fileIndex);
        });
    }

    for (const result of this.parserPool.drainResults()) {
      this.inflightFetches.delete(result.fileIndex);
      this.manager.markLoaded(result.fileIndex);
      onChunkLoaded({ fileIndex: result.fileIndex, chunk: result.chunk });
    }

    for (const error of this.parserPool.drainErrors()) {
      console.error("[lod] parse failed", error);
      this.inflightFetches.delete(error.fileIndex);
      this.manager.markUnrequested(error.fileIndex);
    }
  }

  dispose() {
    this.parserPool.dispose();
  }
}
