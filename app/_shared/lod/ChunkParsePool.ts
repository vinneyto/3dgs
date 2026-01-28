import type { PlyPacked } from "../hooks/usePlyPacked";

export type ChunkParseFormat = "auto" | "ply" | "sog";

export type ChunkParseJob = {
  jobId: number;
  fileIndex: number;
  bytes: Uint8Array;
  format: ChunkParseFormat;
};

export type ChunkParseResult = {
  jobId: number;
  fileIndex: number;
  chunk: PlyPacked;
};

export type ChunkParseError = {
  jobId: number;
  fileIndex: number;
  error: string;
};

type WorkerMessage =
  | { type: "result"; payload: ChunkParseResult }
  | { type: "error"; payload: ChunkParseError };

type WorkerHandle = {
  worker: Worker;
  busy: boolean;
};

export type ChunkParsePoolOptions = {
  size?: number;
  format?: ChunkParseFormat;
};

export class ChunkParsePool {
  private workers: WorkerHandle[] = [];
  private queue: ChunkParseJob[] = [];
  private results: ChunkParseResult[] = [];
  private errors: ChunkParseError[] = [];
  private nextJobId = 1;
  private format: ChunkParseFormat;

  constructor(options: ChunkParsePoolOptions = {}) {
    const concurrency =
      typeof navigator !== "undefined" ? navigator.hardwareConcurrency || 2 : 2;
    const size = options.size ?? Math.max(1, Math.min(4, concurrency));
    this.format = options.format ?? "auto";

    for (let i = 0; i < size; i += 1) {
      const worker = new Worker(
        new URL("./ChunkParseWorker.ts", import.meta.url),
        { type: "module" },
      );
      const handle: WorkerHandle = { worker, busy: false };
      worker.onmessage = (event: MessageEvent<WorkerMessage>) => {
        handle.busy = false;
        if (event.data.type === "result") {
          this.results.push(event.data.payload);
        } else if (event.data.type === "error") {
          this.errors.push(event.data.payload);
        }
        this.pump();
      };
      this.workers.push(handle);
    }
  }

  submit(fileIndex: number, bytes: Uint8Array, format?: ChunkParseFormat) {
    const job: ChunkParseJob = {
      jobId: this.nextJobId++,
      fileIndex,
      bytes,
      format: format ?? this.format,
    };
    this.queue.push(job);
    this.pump();
  }

  drainResults(): ChunkParseResult[] {
    if (this.results.length === 0) {
      return [];
    }
    const out = this.results.slice();
    this.results.length = 0;
    return out;
  }

  drainErrors(): ChunkParseError[] {
    if (this.errors.length === 0) {
      return [];
    }
    const out = this.errors.slice();
    this.errors.length = 0;
    return out;
  }

  dispose() {
    for (const handle of this.workers) {
      handle.worker.terminate();
    }
    this.workers = [];
    this.queue = [];
    this.results = [];
    this.errors = [];
  }

  private pump() {
    if (this.queue.length === 0) {
      return;
    }
    for (const handle of this.workers) {
      if (this.queue.length === 0) {
        break;
      }
      if (handle.busy) {
        continue;
      }
      const job = this.queue.shift();
      if (!job) {
        break;
      }
      handle.busy = true;
      handle.worker.postMessage(
        {
          type: "parse",
          payload: {
            jobId: job.jobId,
            fileIndex: job.fileIndex,
            bytes: job.bytes,
            format: job.format,
          },
        },
        [job.bytes.buffer],
      );
    }
  }
}
