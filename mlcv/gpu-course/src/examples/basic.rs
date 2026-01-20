use bytemuck::{Pod, Zeroable};

use crate::compute::{BindingsBuilder, ComputeContext, ComputeKernel};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    data: [u32; 4],
    scale: [f32; 3],
    _pad0: f32,
}

pub async fn run() -> Result<(), String> {
    let ctx = ComputeContext::new().await?;
    let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let storage = ctx.create_storage_buffer(&data);
    let params = Params {
        data: [data.len() as u32, 0, 0, 0],
        scale: [2.0, 0.0, 0.0],
        _pad0: 0.0,
    };
    let uniform = ctx.create_uniform_buffer(&params);

    let shader = r#"
struct Params {
  data: vec4<u32>,
  scale: vec3<f32>,
  _pad0: f32,
};

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.data.x) { return; }
  data[i] = data[i] * params.scale.x;
}
"#;

    let (bindings, resources) = BindingsBuilder::new()
        .storage(0, &storage, false)
        .uniform(1, &uniform)
        .build();

    let kernel = ComputeKernel::new(&ctx, shader, "main", &bindings, &resources);
    let workgroups = (data.len() as u32 + 63) / 64;
    kernel.dispatch(&ctx, workgroups, 1, 1);

    let out: Vec<f32> = ctx.read_buffer(&storage, data.len());
    println!("out[0..5] = {:?}", &out[..5]);
    Ok(())
}
