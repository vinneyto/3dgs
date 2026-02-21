import {
  makeBindGroupLayoutDescriptors,
  makeShaderDataDefinitions,
  makeStructuredView,
} from "webgpu-utils";
import {
  createComputePipelineWithWgslDebug,
  wrapWgslSourceError,
} from "@/app/_shared/webgpu/wgslDebug";

export const RADIX01_WORKGROUP_SIZE_X = 256;
export const RADIX01_BLOCK_SIZE = RADIX01_WORKGROUP_SIZE_X * 2; // 512
export const RADIX01_RADIX = 4; // 2 bits

function ceilDiv(a: number, b: number): number {
  return Math.floor((a + b - 1) / b);
}

export type RadixSort01LocalCountingDispatch = {
  /** Total number of 512-element blocks. */
  numBlocks: number;
  /** Dispatch dims. (2D is used when numBlocks > maxComputeWorkgroupsPerDimension). */
  x: number;
  y: number;
  z: 1;
};

export type RadixSort01LocalCountingKernel = {
  /** WGSL source (useful for debugging/reflection). */
  code: string;
  pipeline: GPUComputePipeline;
  paramsBuffer: GPUBuffer;
  encode: (
    encoder: GPUCommandEncoder,
    args: { values: GPUBuffer; counts4Out: GPUBuffer; n: number; shift: number },
  ) => { numBlocks: number; dispatch: RadixSort01LocalCountingDispatch };
};

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

/**
 * Radix sort stage 01 (local counting), modeled after `GPGPUTasks2025`:
 * - one workgroup processes 512 input values (256 threads × 2 items/thread)
 * - outputs `counts4_out[block]` as packed 4×u32 histogram for digits 0..3
 *
 * Output layout matches OpenCL `vstore4(tmp[0], gid, counts4_out)` style:
 * `counts4_out[(block*4 + d)] = count(d)`.
 */
export function createRadixSort01LocalCountingKernel({
  device,
  label = "radix_sort_01_local_counting",
}: {
  device: GPUDevice;
  label?: string;
}): Promise<RadixSort01LocalCountingKernel> {
  const code = /* wgsl */ `
struct Params {
  n: u32,
  shift: u32,
  workgroupsX: u32,
  numBlocks: u32,
}

@group(0) @binding(0) var<storage, read> values : array<u32>;
@group(0) @binding(1) var<storage, read_write> counts4_out : array<u32>;
@group(0) @binding(2) var<uniform> params : Params;

var<workgroup> tmp : array<vec4<u32>, ${RADIX01_WORKGROUP_SIZE_X}>;

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

@compute @workgroup_size(${RADIX01_WORKGROUP_SIZE_X}, 1, 1)
fn main(
  @builtin(local_invocation_id) local_id : vec3<u32>,
  @builtin(workgroup_id) workgroup_id : vec3<u32>,
) {
  let lid = local_id.x;

  // Flatten 2D dispatch into a single linear group id.
  let gid = workgroup_id.x + workgroup_id.y * params.workgroupsX;
  if (gid >= params.numBlocks) { return; }

  let base = gid * ${RADIX01_BLOCK_SIZE}u;
  let i0 = base + 2u * lid;
  let i1 = i0 + 1u;

  var c = vec4<u32>(0u, 0u, 0u, 0u);
  if (i0 < params.n) {
    let v0 = values[i0];
    let d0 = (v0 >> params.shift) & 3u;
    bumpDigit(&c, d0);
  }
  if (i1 < params.n) {
    let v1 = values[i1];
    let d1 = (v1 >> params.shift) & 3u;
    bumpDigit(&c, d1);
  }

  tmp[lid] = c;
  workgroupBarrier();

  // Reduce 256 -> 1 (component-wise) in workgroup memory.
  var stride = ${RADIX01_WORKGROUP_SIZE_X / 2}u;
  loop {
    if (stride == 0u) { break; }
    if (lid < stride) {
      tmp[lid] = tmp[lid] + tmp[lid + stride];
    }
    workgroupBarrier();
    stride = stride >> 1u;
  }

  if (lid == 0u) {
    let outBase = gid * 4u;
    let t = tmp[0];
    counts4_out[outBase + 0u] = t.x;
    counts4_out[outBase + 1u] = t.y;
    counts4_out[outBase + 2u] = t.z;
    counts4_out[outBase + 3u] = t.w;
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
        const numBlocks = ceilDiv(n, RADIX01_BLOCK_SIZE);
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
            { binding: 0, resource: { buffer: args.values } },
            { binding: 1, resource: { buffer: args.counts4Out } },
            { binding: 2, resource: { buffer: paramsBuffer } },
          ],
        });

        const dispatch: RadixSort01LocalCountingDispatch = {
          numBlocks,
          x: dispatch2d.x,
          y: dispatch2d.y,
          z: 1,
        };

        const pass = encoder.beginComputePass({ label });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(dispatch.x, dispatch.y, 1);
        pass.end();

        return { numBlocks, dispatch };
      },
    };
  });
}

