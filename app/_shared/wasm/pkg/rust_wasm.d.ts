/* tslint:disable */
/* eslint-disable */

export class LodTileQuery {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
}

export class SplatPlyBuffers {
  private constructor();
  free(): void;
  [Symbol.dispose](): void;
  readonly covariance: Float32Array;
  readonly shCoeffsL1: Float32Array;
  readonly shCoeffsL2Scale: number;
  readonly shCoeffsL3Scale: number;
  readonly shCoeffsL2Packed: Uint32Array;
  readonly shCoeffsL3Packed: Uint32Array;
  readonly rgba: Uint32Array;
  readonly count: number;
  readonly center: Float32Array;
  readonly format: string;
  readonly bboxMax: Float32Array;
  readonly bboxMin: Float32Array;
  readonly shDegree: number;
}

export function hamming_distance_u32(a: number, b: number): number;

export function is_bit_set_u32(a: number, k: number): boolean;

export function parse_splat_ply(bytes: Uint8Array): SplatPlyBuffers;

export function parse_splat_ply_with_opts(bytes: Uint8Array, assume_log_scale: boolean, assume_logit_opacity: boolean): SplatPlyBuffers;

export function powers_of_two_u32(a: number): Array<any>;

export function set_bit_u32(a: number, k: number): number;

export function shift_right_report_u32(a: number, shift: number): string;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_lodtilequery_free: (a: number, b: number) => void;
  readonly lodtilequerywasm_new: (a: number, b: number, c: number) => [number, number, number];
  readonly lodtilequerywasm_query_view_proj: (a: number, b: number, c: number) => [number, number, number];
  readonly __wbg_splatplybuffers_free: (a: number, b: number) => void;
  readonly hamming_distance_u32: (a: number, b: number) => number;
  readonly is_bit_set_u32: (a: number, b: number) => number;
  readonly parse_splat_ply: (a: number, b: number) => [number, number, number];
  readonly parse_splat_ply_with_opts: (a: number, b: number, c: number, d: number) => [number, number, number];
  readonly powers_of_two_u32: (a: number) => any;
  readonly set_bit_u32: (a: number, b: number) => number;
  readonly shift_right_report_u32: (a: number, b: number) => [number, number];
  readonly splatplybuffers_bboxMax: (a: number) => any;
  readonly splatplybuffers_bboxMin: (a: number) => any;
  readonly splatplybuffers_center: (a: number) => any;
  readonly splatplybuffers_count: (a: number) => number;
  readonly splatplybuffers_covariance: (a: number) => any;
  readonly splatplybuffers_format: (a: number) => [number, number];
  readonly splatplybuffers_rgba: (a: number) => any;
  readonly splatplybuffers_shCoeffsL1: (a: number) => any;
  readonly splatplybuffers_shCoeffsL2Packed: (a: number) => any;
  readonly splatplybuffers_shCoeffsL2Scale: (a: number) => number;
  readonly splatplybuffers_shCoeffsL3Packed: (a: number) => any;
  readonly splatplybuffers_shCoeffsL3Scale: (a: number) => number;
  readonly splatplybuffers_shDegree: (a: number) => number;
  readonly __wbindgen_externrefs: WebAssembly.Table;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __externref_table_dealloc: (a: number) => void;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
