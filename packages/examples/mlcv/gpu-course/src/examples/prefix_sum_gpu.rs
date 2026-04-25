use std::time::Instant;

use bytemuck::{Pod, Zeroable};

use crate::compute::{BindingsBuilder, ComputeContext, ComputePipeline};

const WG_SIZE: u32 = 256;
const BLOCK_SIZE: u32 = WG_SIZE * 2;

const SCAN_SHADER: &str = include_str!("kernels/prefix_sum_scan.wgsl");
const ADD_OFFSETS_SHADER: &str = include_str!("kernels/prefix_sum_add_offsets.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    data: [u32; 4],
}

pub async fn run() -> Result<(), String> {
    let n: usize = 100 * 1000 * 1000;
    let mut input = vec![0u32; n];
    let mut total_sum: u64 = 0;
    for i in 0..n {
        let v = (3 * (i as u64 + 5) + 7) % 17;
        input[i] = v as u32;
        total_sum += v;
        assert!(
            total_sum <= u32::MAX as u64,
            "prefix sum overflow at i={}",
            i
        );
    }

    let mut cpu_prefix_sum = vec![0u32; n];
    let start = Instant::now();
    let mut cpu_sum: u64 = 0;
    for i in 0..n {
        cpu_prefix_sum[i] = cpu_sum as u32;
        cpu_sum += input[i] as u64;
    }
    let cpu_elapsed = start.elapsed().as_secs_f64();
    println!("cpu prefix sum time (exclusive, seconds): {}", cpu_elapsed);
    let cpu_memory_gb = (std::mem::size_of::<u32>() as f64 * 2.0 * n as f64)
        / 1024.0
        / 1024.0
        / 1024.0;
    println!(
        "cpu effective bandwidth estimate: {} GB/s",
        cpu_memory_gb / cpu_elapsed
    );

    let ctx = ComputeContext::new().await?;
    let input_buf = ctx.create_storage_buffer(&input);
    let output_buf = ctx.create_storage_buffer(&vec![0u32; n]);
    let dummy_sums_buf = ctx.create_storage_buffer(&[0u32]);

    let params = Params { data: [0, 0, 0, 0] };
    let params_buf = ctx.create_uniform_buffer(&params);

    let ceil_div = |a: usize, b: usize| (a + b - 1) / b;
    let mut sizes = vec![n];
    while *sizes.last().unwrap() > BLOCK_SIZE as usize {
        sizes.push(ceil_div(*sizes.last().unwrap(), BLOCK_SIZE as usize));
    }
    let l = sizes.len().saturating_sub(1);

    let mut sums = Vec::new();
    let mut offs = Vec::new();
    if l > 0 {
        for ell in 0..l {
            sums.push(ctx.create_storage_buffer(&vec![0u32; sizes[ell + 1]]));
            offs.push(ctx.create_storage_buffer(&vec![0u32; sizes[ell + 1]]));
        }
    }

    let scan_bindings = [
        crate::compute::BufferBindingDesc {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        crate::compute::BufferBindingDesc {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        crate::compute::BufferBindingDesc {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        crate::compute::BufferBindingDesc {
            binding: 3,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
    ];
    let add_bindings = [
        crate::compute::BufferBindingDesc {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        crate::compute::BufferBindingDesc {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        crate::compute::BufferBindingDesc {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
    ];
    let scan_pipeline = ComputePipeline::new(&ctx, SCAN_SHADER, "scan512_write_sums", &scan_bindings);
    let scan_add_pipeline =
        ComputePipeline::new(&ctx, SCAN_SHADER, "scan512_write_sums_add_offsets", &scan_bindings);
    let add_offsets_pipeline =
        ComputePipeline::new(&ctx, ADD_OFFSETS_SHADER, "add_block_offsets", &add_bindings);

    let dispatch_scan = |ctx: &ComputeContext,
                         entry: &str,
                         in_buf: &wgpu::Buffer,
                         out_buf: &wgpu::Buffer,
                         extra_buf: &wgpu::Buffer,
                         n: usize| {
        let n_blocks = ceil_div(n, BLOCK_SIZE as usize) as u32;
        let grid_x = n_blocks.min(65535);
        let grid_y = (n_blocks + grid_x - 1) / grid_x;
        let params = Params {
            data: [n as u32, grid_x, 0, 0],
        };
        ctx.update_buffer(&params_buf, &params);
        let (_bindings, resources) = BindingsBuilder::new()
            .storage(0, in_buf, true)
            .storage(1, out_buf, false)
            .storage(2, extra_buf, false)
            .uniform(3, &params_buf)
            .build();
        let bind_group = scan_pipeline.create_bind_group(ctx, &resources);
        if entry == "scan512_write_sums" {
            scan_pipeline.dispatch_with_bind_group(ctx, &bind_group, grid_x.max(1), grid_y.max(1), 1);
        } else {
            scan_add_pipeline.dispatch_with_bind_group(ctx, &bind_group, grid_x.max(1), grid_y.max(1), 1);
        }
    };

    let dispatch_add_offsets =
        |ctx: &ComputeContext, out_buf: &wgpu::Buffer, offsets: &wgpu::Buffer, n: usize| {
            let n_blocks = ceil_div(n, BLOCK_SIZE as usize) as u32;
            let grid_x = n_blocks.min(65535);
            let grid_y = (n_blocks + grid_x - 1) / grid_x;
            let params = Params {
                data: [n as u32, grid_x, 0, 0],
            };
            ctx.update_buffer(&params_buf, &params);
            let (_bindings, resources) = BindingsBuilder::new()
                .storage(0, out_buf, false)
                .storage(1, offsets, true)
                .uniform(2, &params_buf)
                .build();
            let bind_group = add_offsets_pipeline.create_bind_group(ctx, &resources);
            add_offsets_pipeline.dispatch_with_bind_group(ctx, &bind_group, grid_x.max(1), grid_y.max(1), 1);
        };

    // Precompute block offsets once (input is constant across iterations).
    if l > 0 {
        dispatch_scan(
            &ctx,
            "scan512_write_sums",
            &input_buf,
            &output_buf,
            &sums[0],
            n,
        );

        for ell in 0..(l - 1) {
            let nin = sizes[ell + 1];
            dispatch_scan(
                &ctx,
                "scan512_write_sums",
                &sums[ell],
                &offs[ell],
                &sums[ell + 1],
                nin,
            );
        }

        dispatch_scan(
            &ctx,
            "scan512_write_sums",
            &sums[l - 1],
            &offs[l - 1],
            &dummy_sums_buf,
            sizes[l],
        );

        if l >= 2 {
            for ell in (0..(l - 1)).rev() {
                let nout = sizes[ell + 1];
                dispatch_add_offsets(&ctx, &offs[ell], &offs[ell + 1], nout);
            }
        }
    }

    // Ensure precompute work is finished before timing the fused pass.
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());

    let mut times = Vec::new();
    for _ in 0..10 {
        let gpu_start = Instant::now();
        if l == 0 {
            // Single-kernel pass.
            dispatch_scan(
                &ctx,
                "scan512_write_sums",
                &input_buf,
                &output_buf,
                &dummy_sums_buf,
                n,
            );
        } else {
            // Fused pass: scan input blocks + add precomputed block offsets.
            dispatch_scan(
                &ctx,
                "scan512_write_sums_add_offsets",
                &input_buf,
                &output_buf,
                &offs[0],
                n,
            );
        }
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        times.push(gpu_start.elapsed().as_secs_f64());
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let gpu_median = times[times.len() / 2];
    println!("gpu prefix sum time (seconds): {}", gpu_median);
    println!(
        "gpu effective bandwidth estimate: {} GB/s",
        cpu_memory_gb / gpu_median
    );

    let gpu_out: Vec<u32> = ctx.read_buffer(&output_buf, n);
    for i in 0..n {
        assert_eq!(
            cpu_prefix_sum[i], gpu_out[i],
            "prefix sum mismatch at i={}",
            i
        );
    }
    println!("gpu prefix sum verified OK");
    Ok(())
}
