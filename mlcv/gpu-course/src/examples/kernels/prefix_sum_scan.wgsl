const WG: u32 = 256u;
const BLOCK: u32 = 512u;

struct Params {
  data: vec4<u32>, // x: n, y: grid_x
};

@group(0) @binding(0) var<storage, read> in_data: array<u32>;
@group(0) @binding(1) var<storage, read_write> out_data: array<u32>;
@group(0) @binding(2) var<storage, read_write> block_sums: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> temp: array<u32, 512>;

@compute @workgroup_size(256)
fn scan512_write_sums(@builtin(global_invocation_id) gid: vec3<u32>,
                      @builtin(local_invocation_id) lid: vec3<u32>,
                      @builtin(workgroup_id) wid: vec3<u32>) {
  let tid = lid.x;
  let bid = wid.x + wid.y * params.data.y;
  let base = bid * BLOCK;
  let i0 = base + tid;
  let i1 = base + tid + WG;
  let nblocks = (params.data.x + BLOCK - 1u) / BLOCK;
  if (bid >= nblocks) { return; }

  temp[tid] = select(0u, in_data[i0], i0 < params.data.x);
  temp[tid + WG] = select(0u, in_data[i1], i1 < params.data.x);
  workgroupBarrier();

  var offset: u32 = 1u;
  loop {
    if (offset >= BLOCK) { break; }
    let idx = ((tid + 1u) * offset * 2u) - 1u;
    if (idx < BLOCK) {
      temp[idx] = temp[idx] + temp[idx - offset];
    }
    workgroupBarrier();
    offset = offset << 1u;
  }

  if (tid == 0u) {
    block_sums[bid] = temp[BLOCK - 1u];
    temp[BLOCK - 1u] = 0u;
  }
  workgroupBarrier();

  var off: u32 = BLOCK >> 1u;
  loop {
    if (off == 0u) { break; }
    let idx = ((tid + 1u) * off * 2u) - 1u;
    if (idx < BLOCK) {
      let t = temp[idx - off];
      temp[idx - off] = temp[idx];
      temp[idx] = temp[idx] + t;
    }
    workgroupBarrier();
    off = off >> 1u;
  }

  if (i0 < params.data.x) { out_data[i0] = temp[tid]; }
  if (i1 < params.data.x) { out_data[i1] = temp[tid + WG]; }
}

@compute @workgroup_size(256)
fn scan512_write_sums_add_offsets(@builtin(global_invocation_id) gid: vec3<u32>,
                                  @builtin(local_invocation_id) lid: vec3<u32>,
                                  @builtin(workgroup_id) wid: vec3<u32>) {
  let tid = lid.x;
  let bid = wid.x + wid.y * params.data.y;
  let base = bid * BLOCK;
  let i0 = base + tid;
  let i1 = base + tid + WG;
  let nblocks = (params.data.x + BLOCK - 1u) / BLOCK;
  if (bid >= nblocks) { return; }

  temp[tid] = select(0u, in_data[i0], i0 < params.data.x);
  temp[tid + WG] = select(0u, in_data[i1], i1 < params.data.x);
  workgroupBarrier();

  var offset: u32 = 1u;
  loop {
    if (offset >= BLOCK) { break; }
    let idx = ((tid + 1u) * offset * 2u) - 1u;
    if (idx < BLOCK) {
      temp[idx] = temp[idx] + temp[idx - offset];
    }
    workgroupBarrier();
    offset = offset << 1u;
  }

  if (tid == 0u) {
    temp[BLOCK - 1u] = 0u;
  }
  workgroupBarrier();

  var off: u32 = BLOCK >> 1u;
  loop {
    if (off == 0u) { break; }
    let idx = ((tid + 1u) * off * 2u) - 1u;
    if (idx < BLOCK) {
      let t = temp[idx - off];
      temp[idx - off] = temp[idx];
      temp[idx] = temp[idx] + t;
    }
    workgroupBarrier();
    off = off >> 1u;
  }

  let add = block_sums[bid];
  if (i0 < params.data.x) { out_data[i0] = temp[tid] + add; }
  if (i1 < params.data.x) { out_data[i1] = temp[tid + WG] + add; }
}
