import type { ComputeNode, StorageBufferNode } from "three/webgpu";
import {
  Fn,
  invocationLocalIndex,
  numWorkgroups,
  uint,
  wgslFn,
  workgroupArray,
  workgroupId,
} from "three/tsl";

export const RADIX01_WORKGROUP_SIZE_X = 256;
export const RADIX01_BLOCK_SIZE = RADIX01_WORKGROUP_SIZE_X * 2; // 512

export const RADIX02_WORKGROUP_SIZE_X = 256;
export const RADIX02_SCAN_BLOCK_LEN = 512;

export const RADIX03_WORKGROUP_SIZE_X = 256;
export const RADIX03_BLOCK_LEN = 512;

export const RADIX04_WORKGROUP_SIZE_X = 256;
export const RADIX04_BLOCK_SIZE = RADIX04_WORKGROUP_SIZE_X * 2; // 512

function ceilDiv(a: number, b: number): number {
  return Math.floor((a + b - 1) / b);
}

function invocationsForGroups(
  groups: number,
  workgroupSizeX: number,
): number {
  return Math.max(1, groups * workgroupSizeX);
}

const radixSort01LocalCountingFn = wgslFn(/* wgsl */ `
fn radixSort01LocalCounting(
  values: ptr<storage, array<u32>, read>,
  counts4_out: ptr<storage, array<u32>, read_write>,
  n: u32,
  shift: u32,
  workgroupsX: u32,
  numBlocks: u32,
  lid: u32,
  wg_id: vec3<u32>,
  tmp: ptr<workgroup, array<vec4<u32>, 256>, read_write>
) -> void {
  let gid = wg_id.x + wg_id.y * workgroupsX;
  if (gid >= numBlocks) { return; }

  let base = gid * 512u;
  let i0 = base + 2u * lid;
  let i1 = i0 + 1u;

  var c = vec4<u32>(0u, 0u, 0u, 0u);
  if (i0 < n) {
    let v0 = (*values)[i0];
    let d0 = (v0 >> shift) & 3u;
    if (d0 == 0u) {
      c.x = c.x + 1u;
    } else if (d0 == 1u) {
      c.y = c.y + 1u;
    } else if (d0 == 2u) {
      c.z = c.z + 1u;
    } else {
      c.w = c.w + 1u;
    }
  }
  if (i1 < n) {
    let v1 = (*values)[i1];
    let d1 = (v1 >> shift) & 3u;
    if (d1 == 0u) {
      c.x = c.x + 1u;
    } else if (d1 == 1u) {
      c.y = c.y + 1u;
    } else if (d1 == 2u) {
      c.z = c.z + 1u;
    } else {
      c.w = c.w + 1u;
    }
  }

  (*tmp)[lid] = c;
  workgroupBarrier();

  var stride = 128u;
  loop {
    if (stride == 0u) { break; }
    if (lid < stride) {
      (*tmp)[lid] = (*tmp)[lid] + (*tmp)[lid + stride];
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }

  if (lid == 0u) {
    let outBase = gid * 4u;
    let t = (*tmp)[0u];
    (*counts4_out)[outBase + 0u] = t.x;
    (*counts4_out)[outBase + 1u] = t.y;
    (*counts4_out)[outBase + 2u] = t.z;
    (*counts4_out)[outBase + 3u] = t.w;
  }
}
`);

const radixSort02ScanSumReductionFn = wgslFn(/* wgsl */ `
fn radixSort02ScanSumReduction(
  input4: ptr<storage, array<u32>, read>,
  output_scan4: ptr<storage, array<u32>, read_write>,
  output_block_sums4: ptr<storage, array<u32>, read_write>,
  len: u32,
  workgroupsX: u32,
  numGroups: u32,
  lid: u32,
  wg_id: vec3<u32>,
  temp: ptr<workgroup, array<vec4<u32>, 512>, read_write>
) -> void {
  let gid = wg_id.x + wg_id.y * workgroupsX;
  if (gid >= numGroups) { return; }

  let base = gid * 512u;
  let idx0 = base + 2u * lid;
  let idx1 = idx0 + 1u;

  var v0 = vec4<u32>(0u, 0u, 0u, 0u);
  var v1 = vec4<u32>(0u, 0u, 0u, 0u);
  if (idx0 < len) {
    let b0 = idx0 * 4u;
    v0 = vec4<u32>(
      (*input4)[b0 + 0u],
      (*input4)[b0 + 1u],
      (*input4)[b0 + 2u],
      (*input4)[b0 + 3u]
    );
  }
  if (idx1 < len) {
    let b1 = idx1 * 4u;
    v1 = vec4<u32>(
      (*input4)[b1 + 0u],
      (*input4)[b1 + 1u],
      (*input4)[b1 + 2u],
      (*input4)[b1 + 3u]
    );
  }

  (*temp)[2u * lid] = v0;
  (*temp)[2u * lid + 1u] = v1;
  workgroupBarrier();

  var offset = 1u;
  loop {
    if (offset >= 512u) { break; }
    let idx = (lid + 1u) * (offset << 1u) - 1u;
    if (idx < 512u) {
      (*temp)[idx] = (*temp)[idx] + (*temp)[idx - offset];
    }
    workgroupBarrier();
    offset = offset << 1u;
  }

  if (lid == 0u) {
    (*temp)[511u] = vec4<u32>(0u, 0u, 0u, 0u);
  }
  workgroupBarrier();

  offset = 256u;
  loop {
    if (offset == 0u) { break; }
    let idx = (lid + 1u) * (offset << 1u) - 1u;
    if (idx < 512u) {
      let t = (*temp)[idx - offset];
      (*temp)[idx - offset] = (*temp)[idx];
      (*temp)[idx] = (*temp)[idx] + t;
    }
    workgroupBarrier();
    offset = offset >> 1u;
  }

  if (idx0 < len) {
    let b0 = idx0 * 4u;
    let s0 = (*temp)[2u * lid] + v0;
    (*output_scan4)[b0 + 0u] = s0.x;
    (*output_scan4)[b0 + 1u] = s0.y;
    (*output_scan4)[b0 + 2u] = s0.z;
    (*output_scan4)[b0 + 3u] = s0.w;
  }
  if (idx1 < len) {
    let b1 = idx1 * 4u;
    let s1 = (*temp)[2u * lid + 1u] + v1;
    (*output_scan4)[b1 + 0u] = s1.x;
    (*output_scan4)[b1 + 1u] = s1.y;
    (*output_scan4)[b1 + 2u] = s1.z;
    (*output_scan4)[b1 + 3u] = s1.w;
  }

  if (lid == 255u) {
    let outBase = gid * 4u;
    let chunk = (*temp)[511u] + v1;
    (*output_block_sums4)[outBase + 0u] = chunk.x;
    (*output_block_sums4)[outBase + 1u] = chunk.y;
    (*output_block_sums4)[outBase + 2u] = chunk.z;
    (*output_block_sums4)[outBase + 3u] = chunk.w;
  }
}
`);

const radixSort03ScanAccumulationFn = wgslFn(/* wgsl */ `
fn radixSort03ScanAccumulation(
  scan_inout4: ptr<storage, array<u32>, read_write>,
  upper_scan4: ptr<storage, array<u32>, read>,
  len: u32,
  workgroupsX: u32,
  numGroups: u32,
  lid: u32,
  wg_id: vec3<u32>
) -> void {
  let gid = wg_id.x + wg_id.y * workgroupsX;
  if (gid >= numGroups) { return; }

  let base = gid * 512u;
  let idx0 = base + 2u * lid;
  let idx1 = idx0 + 1u;

  var offset = vec4<u32>(0u, 0u, 0u, 0u);
  if (gid > 0u) {
    let ob = (gid - 1u) * 4u;
    offset = vec4<u32>(
      (*upper_scan4)[ob + 0u],
      (*upper_scan4)[ob + 1u],
      (*upper_scan4)[ob + 2u],
      (*upper_scan4)[ob + 3u]
    );
  }

  if (idx0 < len) {
    let b0 = idx0 * 4u;
    let v0 = vec4<u32>(
      (*scan_inout4)[b0 + 0u],
      (*scan_inout4)[b0 + 1u],
      (*scan_inout4)[b0 + 2u],
      (*scan_inout4)[b0 + 3u]
    ) + offset;
    (*scan_inout4)[b0 + 0u] = v0.x;
    (*scan_inout4)[b0 + 1u] = v0.y;
    (*scan_inout4)[b0 + 2u] = v0.z;
    (*scan_inout4)[b0 + 3u] = v0.w;
  }

  if (idx1 < len) {
    let b1 = idx1 * 4u;
    let v1 = vec4<u32>(
      (*scan_inout4)[b1 + 0u],
      (*scan_inout4)[b1 + 1u],
      (*scan_inout4)[b1 + 2u],
      (*scan_inout4)[b1 + 3u]
    ) + offset;
    (*scan_inout4)[b1 + 0u] = v1.x;
    (*scan_inout4)[b1 + 1u] = v1.y;
    (*scan_inout4)[b1 + 2u] = v1.z;
    (*scan_inout4)[b1 + 3u] = v1.w;
  }
}
`);

const radixSort04ScatterFn = wgslFn(/* wgsl */ `
fn radixSort04Scatter(
  values_in: ptr<storage, array<u32>, read>,
  scan_counts4: ptr<storage, array<u32>, read>,
  values_out: ptr<storage, array<u32>, read_write>,
  n: u32,
  shift: u32,
  workgroupsX: u32,
  numBlocks: u32,
  lid: u32,
  wg_id: vec3<u32>,
  temp: ptr<workgroup, array<vec4<u32>, 256>, read_write>,
  global_starts_local: ptr<workgroup, array<vec4<u32>, 1>, read_write>
) -> void {
  let gid = wg_id.x + wg_id.y * workgroupsX;
  if (gid >= numBlocks) { return; }

  let base = gid * 512u;
  let i0 = base + 2u * lid;
  let i1 = i0 + 1u;

  if (lid == 0u) {
    var totals = vec4<u32>(0u, 0u, 0u, 0u);
    if (numBlocks > 0u) {
      let tb = (numBlocks - 1u) * 4u;
      totals = vec4<u32>(
        (*scan_counts4)[tb + 0u],
        (*scan_counts4)[tb + 1u],
        (*scan_counts4)[tb + 2u],
        (*scan_counts4)[tb + 3u]
      );
    }
    let base0 = 0u;
    let base1 = totals.x;
    let base2 = totals.x + totals.y;
    let base3 = totals.x + totals.y + totals.z;
    let digit_bases = vec4<u32>(base0, base1, base2, base3);

    var prefix_before = vec4<u32>(0u, 0u, 0u, 0u);
    if (gid > 0u) {
      let pb = (gid - 1u) * 4u;
      prefix_before = vec4<u32>(
        (*scan_counts4)[pb + 0u],
        (*scan_counts4)[pb + 1u],
        (*scan_counts4)[pb + 2u],
        (*scan_counts4)[pb + 3u]
      );
    }
    (*global_starts_local)[0u] = digit_bases + prefix_before;
  }

  var v0 = 0u;
  var v1 = 0u;
  var d0 = 0u;
  var d1 = 0u;
  var valid0 = false;
  var valid1 = false;
  if (i0 < n) {
    valid0 = true;
    v0 = (*values_in)[i0];
    d0 = (v0 >> shift) & 3u;
  }
  if (i1 < n) {
    valid1 = true;
    v1 = (*values_in)[i1];
    d1 = (v1 >> shift) & 3u;
  }

  var c = vec4<u32>(0u, 0u, 0u, 0u);
  if (valid0) {
    if (d0 == 0u) { c.x = c.x + 1u; }
    else if (d0 == 1u) { c.y = c.y + 1u; }
    else if (d0 == 2u) { c.z = c.z + 1u; }
    else { c.w = c.w + 1u; }
  }
  if (valid1) {
    if (d1 == 0u) { c.x = c.x + 1u; }
    else if (d1 == 1u) { c.y = c.y + 1u; }
    else if (d1 == 2u) { c.z = c.z + 1u; }
    else { c.w = c.w + 1u; }
  }

  (*temp)[lid] = c;
  workgroupBarrier();

  var offset = 1u;
  loop {
    if (offset >= 256u) { break; }
    let idx = (lid + 1u) * (offset << 1u) - 1u;
    if (idx < 256u) {
      (*temp)[idx] = (*temp)[idx] + (*temp)[idx - offset];
    }
    workgroupBarrier();
    offset = offset << 1u;
  }

  if (lid == 0u) {
    (*temp)[255u] = vec4<u32>(0u, 0u, 0u, 0u);
  }
  workgroupBarrier();

  offset = 128u;
  loop {
    if (offset == 0u) { break; }
    let idx = (lid + 1u) * (offset << 1u) - 1u;
    if (idx < 256u) {
      let t = (*temp)[idx - offset];
      (*temp)[idx - offset] = (*temp)[idx];
      (*temp)[idx] = (*temp)[idx] + t;
    }
    workgroupBarrier();
    offset = offset >> 1u;
  }

  let thread_prefix = (*temp)[lid];
  let global_starts = (*global_starts_local)[0u];

  if (valid0) {
    var local_rank0 = 0u;
    var dst0 = 0u;
    if (d0 == 0u) { local_rank0 = thread_prefix.x; dst0 = global_starts.x + local_rank0; }
    else if (d0 == 1u) { local_rank0 = thread_prefix.y; dst0 = global_starts.y + local_rank0; }
    else if (d0 == 2u) { local_rank0 = thread_prefix.z; dst0 = global_starts.z + local_rank0; }
    else { local_rank0 = thread_prefix.w; dst0 = global_starts.w + local_rank0; }
    (*values_out)[dst0] = v0;
  }

  if (valid1) {
    let add = select(0u, 1u, valid0 && (d1 == d0));
    var local_rank1 = 0u;
    var dst1 = 0u;
    if (d1 == 0u) { local_rank1 = thread_prefix.x + add; dst1 = global_starts.x + local_rank1; }
    else if (d1 == 1u) { local_rank1 = thread_prefix.y + add; dst1 = global_starts.y + local_rank1; }
    else if (d1 == 2u) { local_rank1 = thread_prefix.z + add; dst1 = global_starts.z + local_rank1; }
    else { local_rank1 = thread_prefix.w + add; dst1 = global_starts.w + local_rank1; }
    (*values_out)[dst1] = v1;
  }
}
`);

export function createRadixSort01LocalCountingCompute({
  values,
  counts4Out,
  n,
  shift,
  name = "radix_sort_01_local_counting_tsl",
}: {
  values: StorageBufferNode;
  counts4Out: StorageBufferNode;
  n: number;
  shift: number;
  name?: string;
}): ComputeNode {
  const numBlocks = ceilDiv(Math.max(0, n | 0), RADIX01_BLOCK_SIZE);
  const invocations = invocationsForGroups(numBlocks, RADIX01_WORKGROUP_SIZE_X);

  return Fn(() => {
    const tmp = workgroupArray("uvec4", RADIX01_WORKGROUP_SIZE_X);
    radixSort01LocalCountingFn({
      values,
      counts4_out: counts4Out,
      n: uint(n >>> 0),
      shift: uint(shift >>> 0),
      workgroupsX: numWorkgroups.x,
      numBlocks: uint(numBlocks >>> 0),
      lid: invocationLocalIndex,
      wg_id: workgroupId,
      tmp,
    });
  })()
    .compute(invocations, [RADIX01_WORKGROUP_SIZE_X, 1, 1])
    .setName(name);
}

export function createRadixSort02ScanSumReductionCompute({
  input4,
  outputScan4,
  outputBlockSums4,
  len,
  name = "radix_sort_02_scan_sum_reduction_tsl",
}: {
  input4: StorageBufferNode;
  outputScan4: StorageBufferNode;
  outputBlockSums4: StorageBufferNode;
  len: number;
  name?: string;
}): ComputeNode {
  const numGroups = ceilDiv(Math.max(0, len | 0), RADIX02_SCAN_BLOCK_LEN);
  const invocations = invocationsForGroups(numGroups, RADIX02_WORKGROUP_SIZE_X);

  return Fn(() => {
    const temp = workgroupArray("uvec4", RADIX02_SCAN_BLOCK_LEN);
    radixSort02ScanSumReductionFn({
      input4,
      output_scan4: outputScan4,
      output_block_sums4: outputBlockSums4,
      len: uint(len >>> 0),
      workgroupsX: numWorkgroups.x,
      numGroups: uint(numGroups >>> 0),
      lid: invocationLocalIndex,
      wg_id: workgroupId,
      temp,
    });
  })()
    .compute(invocations, [RADIX02_WORKGROUP_SIZE_X, 1, 1])
    .setName(name);
}

export function createRadixSort03ScanAccumulationCompute({
  scanInOut4,
  upperScan4,
  len,
  name = "radix_sort_03_scan_accumulation_tsl",
}: {
  scanInOut4: StorageBufferNode;
  upperScan4: StorageBufferNode;
  len: number;
  name?: string;
}): ComputeNode {
  const numGroups = ceilDiv(Math.max(0, len | 0), RADIX03_BLOCK_LEN);
  const invocations = invocationsForGroups(numGroups, RADIX03_WORKGROUP_SIZE_X);

  return Fn(() => {
    radixSort03ScanAccumulationFn({
      scan_inout4: scanInOut4,
      upper_scan4: upperScan4,
      len: uint(len >>> 0),
      workgroupsX: numWorkgroups.x,
      numGroups: uint(numGroups >>> 0),
      lid: invocationLocalIndex,
      wg_id: workgroupId,
    });
  })()
    .compute(invocations, [RADIX03_WORKGROUP_SIZE_X, 1, 1])
    .setName(name);
}

export function createRadixSort04ScatterCompute({
  valuesIn,
  scanCounts4,
  valuesOut,
  n,
  shift,
  name = "radix_sort_04_scatter_tsl",
}: {
  valuesIn: StorageBufferNode;
  scanCounts4: StorageBufferNode;
  valuesOut: StorageBufferNode;
  n: number;
  shift: number;
  name?: string;
}): ComputeNode {
  const numBlocks = ceilDiv(Math.max(0, n | 0), RADIX04_BLOCK_SIZE);
  const invocations = invocationsForGroups(numBlocks, RADIX04_WORKGROUP_SIZE_X);

  return Fn(() => {
    const temp = workgroupArray("uvec4", RADIX04_WORKGROUP_SIZE_X);
    const globalStartsLocal = workgroupArray("uvec4", 1);
    radixSort04ScatterFn({
      values_in: valuesIn,
      scan_counts4: scanCounts4,
      values_out: valuesOut,
      n: uint(n >>> 0),
      shift: uint(shift >>> 0),
      workgroupsX: numWorkgroups.x,
      numBlocks: uint(numBlocks >>> 0),
      lid: invocationLocalIndex,
      wg_id: workgroupId,
      temp,
      global_starts_local: globalStartsLocal,
    });
  })()
    .compute(invocations, [RADIX04_WORKGROUP_SIZE_X, 1, 1])
    .setName(name);
}
