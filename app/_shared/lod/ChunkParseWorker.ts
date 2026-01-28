import { getRustWasm } from "../lib/rustWasm";
import type { PlyPacked } from "../hooks/usePlyPacked";

type ParseRequest = {
  type: "parse";
  payload: {
    jobId: number;
    fileIndex: number;
    bytes: Uint8Array;
    format: "auto" | "ply" | "sog";
  };
};

type ParseResponse =
  | { type: "result"; payload: { jobId: number; fileIndex: number; chunk: PlyPacked } }
  | { type: "error"; payload: { jobId: number; fileIndex: number; error: string } };

async function parseBytes(bytes: Uint8Array, format: "auto" | "ply" | "sog"): Promise<PlyPacked> {
  const mod = await getRustWasm();
  const modAny = mod as unknown as {
    parse_sog?: (input: Uint8Array) => any;
    parse_sog_packed?: (input: Uint8Array) => any;
    parse_splat_ply: (input: Uint8Array) => any;
  };

  const parseSog = modAny.parse_sog_packed ?? modAny.parse_sog;
  const parsePly = modAny.parse_splat_ply;

  let out: any;
  if (format === "sog") {
    if (!parseSog) {
      throw new Error("parse_sog is not available in wasm module");
    }
    out = parseSog(bytes);
  } else if (format === "ply") {
    out = parsePly(bytes);
  } else {
    if (parseSog) {
      try {
        out = parseSog(bytes);
      } catch (err) {
        out = parsePly(bytes);
      }
    } else {
      out = parsePly(bytes);
    }
  }

  const shCoeffsL1 = new Float32Array(out.shCoeffsL1);
  const shCoeffsL2Packed = new Uint32Array(out.shCoeffsL2Packed);
  const shCoeffsL3Packed = new Uint32Array(out.shCoeffsL3Packed);

  const chunk: PlyPacked = {
    count: out.count,
    center: new Float32Array(out.center),
    covariance: new Float32Array(out.covariance),
    rgba: new Uint32Array(out.rgba),
    shCoeffsL1: shCoeffsL1.length > 0 ? shCoeffsL1 : undefined,
    shCoeffsL2Packed: shCoeffsL2Packed.length > 0 ? shCoeffsL2Packed : undefined,
    shCoeffsL2Scale: out.shCoeffsL2Scale,
    shCoeffsL3Packed: shCoeffsL3Packed.length > 0 ? shCoeffsL3Packed : undefined,
    shCoeffsL3Scale: out.shCoeffsL3Scale,
    shDegree: out.shDegree,
  };

  out.free?.();

  return chunk;
}

const workerScope = self as unknown as {
  postMessage: (message: ParseResponse, transfer?: Transferable[]) => void;
  addEventListener: (
    type: "message",
    listener: (event: MessageEvent<ParseRequest>) => void,
  ) => void;
};

workerScope.addEventListener("message", async (event: MessageEvent<ParseRequest>) => {
  const { payload } = event.data;
  const { jobId, fileIndex, bytes, format } = payload;

  try {
    const chunk = await parseBytes(bytes, format);
    const transfer: Transferable[] = [];
    const pushBuffer = (buf: ArrayBufferLike) => {
      if (buf instanceof ArrayBuffer) {
        transfer.push(buf);
      }
    };
    pushBuffer(chunk.center.buffer);
    pushBuffer(chunk.covariance.buffer);
    pushBuffer(chunk.rgba.buffer);
    if (chunk.shCoeffsL1) pushBuffer(chunk.shCoeffsL1.buffer);
    if (chunk.shCoeffsL2Packed) pushBuffer(chunk.shCoeffsL2Packed.buffer);
    if (chunk.shCoeffsL3Packed) pushBuffer(chunk.shCoeffsL3Packed.buffer);

    const response: ParseResponse = {
      type: "result",
      payload: { jobId, fileIndex, chunk },
    };
    workerScope.postMessage(response, transfer);
  } catch (err) {
    const response: ParseResponse = {
      type: "error",
      payload: {
        jobId,
        fileIndex,
        error: err instanceof Error ? err.message : String(err),
      },
    };
    workerScope.postMessage(response);
  }
});
