import {
  makeBindGroupLayoutDescriptors,
  makeShaderDataDefinitions,
  makeStructuredView,
} from "webgpu-utils";
import {
  createComputePipelineWithWgslDebug,
  wrapWgslSourceError,
} from "@/app/_shared/webgpu/wgslDebug";

export const RADIX04_WORKGROUP_SIZE_X = 256;
export const RADIX04_BLOCK_SIZE = RADIX04_WORKGROUP_SIZE_X * 2; // 512

function ceilDiv(a: number, b: number): number {
  return Math.floor((a + b - 1) / b);
}

function compute2DDispatch(
  device: GPUDevice,
  numWorkgroupsTotal: number,
): { x: number; y: number } {
  const maxDim = Number(device.limits.maxComputeWorkgroupsPerDimension);
  const x = Math.min(maxDim, numWorkgroupsTotal);
  const y = Math.ceil(numWorkgroupsTotal / x);
  if (y > maxDim) {
    throw new Error(
      `Dispatch too large: workgroupsTotal=${numWorkgroupsTotal}, maxComputeWorkgroupsPerDimension=${maxDim}`,
    );
  }
  return { x, y };
}

export type RadixSort04ScatterKernel = {
  code: string;
  pipeline: GPUComputePipeline;
  paramsBuffer: GPUBuffer;
  encode: (
    encoder: GPUCommandEncoder,
    args: {
      valuesIn: GPUBuffer;
      scanCounts4: GPUBuffer;
      valuesOut: GPUBuffer;
      n: number;
      shift: number;
    },
  ) => { numBlocks: number; dispatch: { x: number; y: number; z: 1 } };
};

/**
 * Stage 04 scatter for RADIX=4 (2-bit digit), modeled after GPGPUTasks2025 OpenCL kernel.
 *
 * Uses:
 * - digit bases from totals = scanCounts4[lastBlock]
 * - global_before_block from scanCounts4[gid-1]
 * - local_rank from exclusive scan of per-thread digit counts (vec4<u32>) within workgroup
 *
 * Writes stable partitioned output:
 * dst = base[d] + global_before_block[d] + local_rank[d]
 */
export function createRadixSort04ScatterKernel({
  device,
  label = "radix_sort_04_scatter",
}: {
  device: GPUDevice;
  label?: string;
}): Promise<RadixSort04ScatterKernel> {
  const code = /* wgsl */ `
struct Params {
  n: u32,
  shift: u32,
  workgroupsX: u32,
  numBlocks: u32,
}

@group(0) @binding(0) var<storage, read> values_in : array<u32>;
@group(0) @binding(1) var<storage, read> scan_counts4 : array<u32>;
@group(0) @binding(2) var<storage, read_write> values_out : array<u32>;
@group(0) @binding(3) var<uniform> params : Params;

var<workgroup> temp : array<vec4<u32>, ${RADIX04_WORKGROUP_SIZE_X}>;
var<workgroup> global_starts_local : vec4<u32>;

fn load4_scan_counts4(idx: u32) -> vec4<u32> {
  let base = idx * 4u;
  return vec4<u32>(
    scan_counts4[base + 0u],
    scan_counts4[base + 1u],
    scan_counts4[base + 2u],
    scan_counts4[base + 3u]
  );
}

fn bumpDigit(c: ptr<function, vec4<u32>>, d: u32) {
  if (d == 0u) {
    (*c).x = (*c).x + 1u;
  } else if (d == 1u) {
    (*c).y = (*c).y + 1u;
  } else if (d == 2u) {
    (*c).z = (*c).z + 1u;
  } else {
    (*c).w = (*c).w + 1u;
  }
}

@compute @workgroup_size(${RADIX04_WORKGROUP_SIZE_X}, 1, 1)
fn main(
  @builtin(local_invocation_id) local_id : vec3<u32>,
  @builtin(workgroup_id) workgroup_id : vec3<u32>,
) {
  let lid = local_id.x;
  let gid = workgroup_id.x + workgroup_id.y * params.workgroupsX;
  if (gid >= params.numBlocks) { return; }

  let base = gid * ${RADIX04_BLOCK_SIZE}u;
  let i0 = base + 2u * lid;
  let i1 = i0 + 1u;

  // Lane0 computes global starts for this block:
  // digit_bases from totals, plus prefix_before for this gid.
  if (lid == 0u) {
    var totals = vec4<u32>(0u, 0u, 0u, 0u);
    if (params.numBlocks > 0u) {
      totals = load4_scan_counts4(params.numBlocks - 1u);
    }
    let base0 = 0u;
    let base1 = totals.x;
    let base2 = totals.x + totals.y;
    let base3 = totals.x + totals.y + totals.z;
    let digit_bases = vec4<u32>(base0, base1, base2, base3);

    var prefix_before = vec4<u32>(0u, 0u, 0u, 0u);
    if (gid > 0u) {
      prefix_before = load4_scan_counts4(gid - 1u);
    }
    global_starts_local = digit_bases + prefix_before;
  }

  // Load digits for two elements (guarded).
  var v0 = 0u;
  var v1 = 0u;
  var d0 = 0u;
  var d1 = 0u;
  var valid0 = false;
  var valid1 = false;
  if (i0 < params.n) {
    valid0 = true;
    v0 = values_in[i0];
    d0 = (v0 >> params.shift) & 3u;
  }
  if (i1 < params.n) {
    valid1 = true;
    v1 = values_in[i1];
    d1 = (v1 >> params.shift) & 3u;
  }

  // Per-thread digit counts for stable local ranks.
  var c = vec4<u32>(0u, 0u, 0u, 0u);
  if (valid0) { bumpDigit(&c, d0); }
  if (valid1) { bumpDigit(&c, d1); }

  // Exclusive scan of c across 256 threads (component-wise via vec4<u32>).
  temp[lid] = c;
  workgroupBarrier(); // also sync global_starts_local

  // up-sweep
  var offset = 1u;
  loop {
    if (offset >= ${RADIX04_WORKGROUP_SIZE_X}u) { break; }
    let idx = (lid + 1u) * (offset << 1u) - 1u;
    if (idx < ${RADIX04_WORKGROUP_SIZE_X}u) {
      temp[idx] = temp[idx] + temp[idx - offset];
    }
    workgroupBarrier();
    offset = offset << 1u;
  }

  // exclusive root
  if (lid == 0u) {
    temp[${RADIX04_WORKGROUP_SIZE_X - 1}u] = vec4<u32>(0u, 0u, 0u, 0u);
  }
  workgroupBarrier();

  // down-sweep
  offset = ${RADIX04_WORKGROUP_SIZE_X / 2}u;
  loop {
    if (offset == 0u) { break; }
    let idx = (lid + 1u) * (offset << 1u) - 1u;
    if (idx < ${RADIX04_WORKGROUP_SIZE_X}u) {
      let t = temp[idx - offset];
      temp[idx - offset] = temp[idx];
      temp[idx] = temp[idx] + t;
    }
    workgroupBarrier();
    offset = offset >> 1u;
  }

  let thread_prefix = temp[lid];
  let global_starts = global_starts_local;

  if (valid0) {
    var local_rank0 = 0u;
    var dst0 = 0u;
    if (d0 == 0u) { local_rank0 = thread_prefix.x; dst0 = global_starts.x + local_rank0; }
    else if (d0 == 1u) { local_rank0 = thread_prefix.y; dst0 = global_starts.y + local_rank0; }
    else if (d0 == 2u) { local_rank0 = thread_prefix.z; dst0 = global_starts.z + local_rank0; }
    else { local_rank0 = thread_prefix.w; dst0 = global_starts.w + local_rank0; }
    values_out[dst0] = v0;
  }

  if (valid1) {
    let add = select(0u, 1u, valid0 && (d1 == d0));
    var local_rank1 = 0u;
    var dst1 = 0u;
    if (d1 == 0u) { local_rank1 = thread_prefix.x + add; dst1 = global_starts.x + local_rank1; }
    else if (d1 == 1u) { local_rank1 = thread_prefix.y + add; dst1 = global_starts.y + local_rank1; }
    else if (d1 == 2u) { local_rank1 = thread_prefix.z + add; dst1 = global_starts.z + local_rank1; }
    else { local_rank1 = thread_prefix.w + add; dst1 = global_starts.w + local_rank1; }
    values_out[dst1] = v1;
  }
}
`;

  let defs: ReturnType<typeof makeShaderDataDefinitions>;
  try {
    defs = makeShaderDataDefinitions(code);
  } catch (e) {
    throw wrapWgslSourceError({
      label,
      stage: "makeShaderDataDefinitions",
      code,
      error: e,
    });
  }
  const paramsView = makeStructuredView(defs.uniforms.params);

  const paramsBuffer = device.createBuffer({
    label: `${label}_params`,
    size: paramsView.arrayBuffer.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  let bglDescs: ReturnType<typeof makeBindGroupLayoutDescriptors>;
  try {
    bglDescs = makeBindGroupLayoutDescriptors(defs, {
      compute: { entryPoint: "main" },
    });
  } catch (e) {
    throw wrapWgslSourceError({
      label,
      stage: "makeBindGroupLayoutDescriptors",
      code,
      error: e,
    });
  }
  const bindGroupLayouts: GPUBindGroupLayout[] = [];
  for (let i = 0; i < bglDescs.length; i++) {
    bindGroupLayouts[i] = device.createBindGroupLayout(bglDescs[i] ?? { entries: [] });
  }
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts });

  return createComputePipelineWithWgslDebug({
    device,
    label,
    code,
    layout: pipelineLayout,
    entryPoint: "main",
  }).then(({ pipeline }) => {
    return {
      code,
      pipeline,
      paramsBuffer,
      encode: (encoder, args) => {
        const n = Math.max(0, args.n | 0);
        const shift = args.shift >>> 0;
        const numBlocks = ceilDiv(n, RADIX04_BLOCK_SIZE);
        const dispatch2d = compute2DDispatch(device, numBlocks);

        paramsView.set({
          n: n >>> 0,
          shift,
          workgroupsX: dispatch2d.x >>> 0,
          numBlocks: numBlocks >>> 0,
        });
        device.queue.writeBuffer(paramsBuffer, 0, paramsView.arrayBuffer);

        const bindGroup = device.createBindGroup({
          label: `${label}_bindGroup0`,
          layout: bindGroupLayouts[0] ?? pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: args.valuesIn } },
            { binding: 1, resource: { buffer: args.scanCounts4 } },
            { binding: 2, resource: { buffer: args.valuesOut } },
            { binding: 3, resource: { buffer: paramsBuffer } },
          ],
        });

        const pass = encoder.beginComputePass({ label });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(dispatch2d.x, dispatch2d.y, 1);
        pass.end();

        return { numBlocks, dispatch: { x: dispatch2d.x, y: dispatch2d.y, z: 1 } };
      },
    };
  });
}

