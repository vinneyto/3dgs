import {
  makeBindGroupLayoutDescriptors,
  makeShaderDataDefinitions,
  makeStructuredView,
} from "webgpu-utils";
import {
  createComputePipelineWithWgslDebug,
  wrapWgslSourceError,
} from "@/app/_shared/webgpu/wgslDebug";

export const RADIX03_WORKGROUP_SIZE_X = 256;
export const RADIX03_BLOCK_LEN = 512;

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

export type RadixSort03ScanAccumulationKernel = {
  code: string;
  pipeline: GPUComputePipeline;
  paramsBuffer: GPUBuffer;
  encode: (
    encoder: GPUCommandEncoder,
    args: { scanInOut4: GPUBuffer; upperScan4: GPUBuffer; len: number },
  ) => { numGroups: number; dispatch: { x: number; y: number; z: 1 } };
};

/**
 * Stage 03 (down-pass accumulation): adds per-chunk offsets from `upper_scan4`
 * to `scan_inout4` in-place.
 *
 * Mirrors `GPGPUTasks2025/src/kernels/cl/radix_sort_03_global_prefixes_scan_accumulation.cl`.
 */
export function createRadixSort03GlobalPrefixesScanAccumulationKernel({
  device,
  label = "radix_sort_03_global_prefixes_scan_accumulation",
}: {
  device: GPUDevice;
  label?: string;
}): Promise<RadixSort03ScanAccumulationKernel> {
  const code = /* wgsl */ `
struct Params {
  len: u32,
  workgroupsX: u32,
  numGroups: u32,
  _pad0: u32,
}

@group(0) @binding(0) var<storage, read_write> scan_inout4 : array<u32>;
@group(0) @binding(1) var<storage, read> upper_scan4 : array<u32>;
@group(0) @binding(2) var<uniform> params : Params;

fn load4_upper_scan4(idx: u32) -> vec4<u32> {
  let base = idx * 4u;
  return vec4<u32>(
    upper_scan4[base + 0u],
    upper_scan4[base + 1u],
    upper_scan4[base + 2u],
    upper_scan4[base + 3u]
  );
}

fn load4_scan_inout4(idx: u32) -> vec4<u32> {
  let base = idx * 4u;
  return vec4<u32>(
    scan_inout4[base + 0u],
    scan_inout4[base + 1u],
    scan_inout4[base + 2u],
    scan_inout4[base + 3u]
  );
}

fn store4_scan_inout4(idx: u32, v: vec4<u32>) {
  let base = idx * 4u;
  scan_inout4[base + 0u] = v.x;
  scan_inout4[base + 1u] = v.y;
  scan_inout4[base + 2u] = v.z;
  scan_inout4[base + 3u] = v.w;
}

@compute @workgroup_size(${RADIX03_WORKGROUP_SIZE_X}, 1, 1)
fn main(
  @builtin(local_invocation_id) local_id : vec3<u32>,
  @builtin(workgroup_id) workgroup_id : vec3<u32>,
) {
  let lid = local_id.x;
  let gid = workgroup_id.x + workgroup_id.y * params.workgroupsX;
  if (gid >= params.numGroups) { return; }

  let base = gid * ${RADIX03_BLOCK_LEN}u;
  let idx0 = base + 2u * lid;
  let idx1 = idx0 + 1u;

  var offset = vec4<u32>(0u, 0u, 0u, 0u);
  if (gid > 0u) {
    offset = load4_upper_scan4(gid - 1u);
  }

  if (idx0 < params.len) {
    let v = load4_scan_inout4(idx0);
    store4_scan_inout4(idx0, v + offset);
  }
  if (idx1 < params.len) {
    let v = load4_scan_inout4(idx1);
    store4_scan_inout4(idx1, v + offset);
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
    bindGroupLayouts[i] = device.createBindGroupLayout(
      bglDescs[i] ?? { entries: [] },
    );
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
        const numGroups = ceilDiv(len, RADIX03_BLOCK_LEN);
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
            { binding: 0, resource: { buffer: args.scanInOut4 } },
            { binding: 1, resource: { buffer: args.upperScan4 } },
            { binding: 2, resource: { buffer: paramsBuffer } },
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
