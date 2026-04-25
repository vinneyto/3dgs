const WG: u32 = 256u;
const BLOCK: u32 = 512u;

struct Params {
  data: vec4<u32>, // x: n, y: grid_x
};

@group(0) @binding(0) var<storage, read_write> out_data: array<u32>;
@group(0) @binding(1) var<storage, read> block_offsets: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn add_block_offsets(@builtin(local_invocation_id) lid: vec3<u32>,
                     @builtin(workgroup_id) wid: vec3<u32>) {
  let tid = lid.x;
  let bid = wid.x + wid.y * params.data.y;
  let base = bid * BLOCK;
  let i0 = base + tid;
  let i1 = base + tid + WG;
  let nblocks = (params.data.x + BLOCK - 1u) / BLOCK;
  if (bid >= nblocks) { return; }

  let off = block_offsets[bid];
  if (i0 < params.data.x) { out_data[i0] = out_data[i0] + off; }
  if (i1 < params.data.x) { out_data[i1] = out_data[i1] + off; }
}
