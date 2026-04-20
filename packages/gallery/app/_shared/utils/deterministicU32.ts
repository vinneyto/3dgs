/**
 * Deterministic u32 "random" generator used to build reproducible datasets.
 *
 * This matches the old WGSL fill kernel we used in the WebGPU demo:
 *
 * - x = (index ^ seed)
 * - x = lcg(x)
 * - x = x ^ (x >> 16)
 */

export function lcgU32(x: number): number {
  // Keep in uint32 domain.
  return (Math.imul(x >>> 0, 1664525) + 1013904223) >>> 0;
}

export function deterministicU32FromIndex(index: number, seed: number): number {
  let x = ((index >>> 0) ^ (seed >>> 0)) >>> 0;
  x = lcgU32(x);
  x = (x ^ (x >>> 16)) >>> 0;
  return x;
}

export function fillDeterministicU32FromIndex(
  out: Uint32Array,
  seed: number,
  startIndex = 0,
): void {
  for (let i = 0; i < out.length; i++) {
    out[i] = deterministicU32FromIndex(startIndex + i, seed);
  }
}

