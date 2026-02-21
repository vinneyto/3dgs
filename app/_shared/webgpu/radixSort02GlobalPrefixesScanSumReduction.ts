import {
  makeBindGroupLayoutDescriptors,
  makeShaderDataDefinitions,
  makeStructuredView,
} from "webgpu-utils";
import {
  createComputePipelineWithWgslDebug,
  wrapWgslSourceError,
} from "@/app/_shared/webgpu/wgslDebug";

export const RADIX02_WORKGROUP_SIZE_X = 256;
export const RADIX02_SCAN_BLOCK_LEN = 512; // elements per workgroup

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

export type RadixSort02ScanSumReductionKernel = {
  code: string;
  pipeline: GPUComputePipeline;
  paramsBuffer: GPUBuffer;
  encode: (
    encoder: GPUCommandEncoder,
    args: {
      /** u32[4*len] */
      input4: GPUBuffer;
      /** u32[4*len] */
      outputScan4: GPUBuffer;
      /** u32[4*ceil(len/512)] */
      outputBlockSums4: GPUBuffer;
      /** Length in packed uint4 elements. */
      len: number;
    },
  ) => { numGroups: number; dispatch: { x: number; y: number; z: 1 } };
};

/**
 * Stage 02 (up-sweep scan + sums) for packed uint4 arrays stored as u32[4*len].
 *
 * Mirrors `GPGPUTasks2025/src/kernels/cl/radix_sort_02_global_prefixes_scan_sum_reduction.cl`:
 * - scans 512 elements per workgroup (2 per thread)
 * - writes inclusive scan to `output_scan4`
 * - writes per-chunk sums to `output_block_sums4` (one uint4 per chunk)
 */
export function createRadixSort02GlobalPrefixesScanSumReductionKernel({
  device,
  label = "radix_sort_02_global_prefixes_scan_sum_reduction",
}: {
  device: GPUDevice;
  label?: string;
}): Promise<RadixSort02ScanSumReductionKernel> {
  const code = /* wgsl */ `
struct Params {
  len: u32,
  workgroupsX: u32,
  numGroups: u32,
  _pad0: u32,
}

@group(0) @binding(0) var<storage, read> input4 : array<u32>;
@group(0) @binding(1) var<storage, read_write> output_scan4 : array<u32>;
@group(0) @binding(2) var<storage, read_write> output_block_sums4 : array<u32>;
@group(0) @binding(3) var<uniform> params : Params;

var<workgroup> temp : array<vec4<u32>, ${RADIX02_SCAN_BLOCK_LEN}>;

fn load4_input4(idx: u32) -> vec4<u32> {
  let base = idx * 4u;
  return vec4<u32>(
    input4[base + 0u],
    input4[base + 1u],
    input4[base + 2u],
    input4[base + 3u]
  );
}

fn store4_output_scan4(idx: u32, v: vec4<u32>) {
  let base = idx * 4u;
  output_scan4[base + 0u] = v.x;
  output_scan4[base + 1u] = v.y;
  output_scan4[base + 2u] = v.z;
  output_scan4[base + 3u] = v.w;
}

fn store4_output_block_sums4(idx: u32, v: vec4<u32>) {
  let base = idx * 4u;
  output_block_sums4[base + 0u] = v.x;
  output_block_sums4[base + 1u] = v.y;
  output_block_sums4[base + 2u] = v.z;
  output_block_sums4[base + 3u] = v.w;
}

@compute @workgroup_size(${RADIX02_WORKGROUP_SIZE_X}, 1, 1)
fn main(
  @builtin(local_invocation_id) local_id : vec3<u32>,
  @builtin(workgroup_id) workgroup_id : vec3<u32>,
) {
  let lid = local_id.x;
  let gid = workgroup_id.x + workgroup_id.y * params.workgroupsX;
  if (gid >= params.numGroups) { return; }

  let base = gid * ${RADIX02_SCAN_BLOCK_LEN}u;
  let idx0 = base + 2u * lid;
  let idx1 = idx0 + 1u;

  var v0 = vec4<u32>(0u, 0u, 0u, 0u);
  var v1 = vec4<u32>(0u, 0u, 0u, 0u);
  if (idx0 < params.len) { v0 = load4_input4(idx0); }
  if (idx1 < params.len) { v1 = load4_input4(idx1); }

  temp[2u * lid] = v0;
  temp[2u * lid + 1u] = v1;
  workgroupBarrier();

  // up-sweep
  var offset = 1u;
  loop {
    if (offset >= ${RADIX02_SCAN_BLOCK_LEN}u) { break; }
    let idx = (lid + 1u) * (offset << 1u) - 1u;
    if (idx < ${RADIX02_SCAN_BLOCK_LEN}u) {
      temp[idx] = temp[idx] + temp[idx - offset];
    }
    workgroupBarrier();
    offset = offset << 1u;
  }

  // exclusive root
  if (lid == 0u) {
    temp[${RADIX02_SCAN_BLOCK_LEN - 1}u] = vec4<u32>(0u, 0u, 0u, 0u);
  }
  workgroupBarrier();

  // down-sweep
  offset = ${RADIX02_SCAN_BLOCK_LEN / 2}u;
  loop {
    if (offset == 0u) { break; }
    let idx = (lid + 1u) * (offset << 1u) - 1u;
    if (idx < ${RADIX02_SCAN_BLOCK_LEN}u) {
      let t = temp[idx - offset];
      temp[idx - offset] = temp[idx];
      temp[idx] = temp[idx] + t;
    }
    workgroupBarrier();
    offset = offset >> 1u;
  }

  // write inclusive scan
  if (idx0 < params.len) { store4_output_scan4(idx0, temp[2u * lid] + v0); }
  if (idx1 < params.len) { store4_output_scan4(idx1, temp[2u * lid + 1u] + v1); }

  // chunk sum (inclusive last element)
  if (lid == ${RADIX02_WORKGROUP_SIZE_X - 1}u) {
    store4_output_block_sums4(gid, temp[${RADIX02_SCAN_BLOCK_LEN - 1}u] + v1);
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
        const len = Math.max(0, args.len | 0);
        const numGroups = ceilDiv(len, RADIX02_SCAN_BLOCK_LEN);
        const dispatch2d = compute2DDispatch(device, numGroups);

        paramsView.set({
          len: len >>> 0,
          workgroupsX: dispatch2d.x >>> 0,
          numGroups: numGroups >>> 0,
        });
        device.queue.writeBuffer(paramsBuffer, 0, paramsView.arrayBuffer);

        const bindGroup = device.createBindGroup({
          label: `${label}_bindGroup0`,
          layout: bindGroupLayouts[0] ?? pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: args.input4 } },
            { binding: 1, resource: { buffer: args.outputScan4 } },
            { binding: 2, resource: { buffer: args.outputBlockSums4 } },
            { binding: 3, resource: { buffer: paramsBuffer } },
          ],
        });

        const pass = encoder.beginComputePass({ label });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(dispatch2d.x, dispatch2d.y, 1);
        pass.end();

        return { numGroups, dispatch: { x: dispatch2d.x, y: dispatch2d.y, z: 1 } };
      },
    };
  });
}

