use std::num::NonZeroU64;
use std::sync::mpsc;

use bytemuck::Pod;
use wgpu::util::DeviceExt;

pub struct ComputeContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl ComputeContext {
    pub async fn new() -> Result<Self, String> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .map_err(|e| e.to_string())?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .map_err(|e| e.to_string())?;
        Ok(Self { device, queue })
    }

    pub fn create_storage_buffer<T: Pod>(&self, data: &[T]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("storage_buffer"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        })
    }

    pub fn create_uniform_buffer<T: Pod>(&self, data: &T) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniform_buffer"),
            contents: bytemuck::bytes_of(data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    #[allow(dead_code)]
    pub fn create_uniform_buffer_bytes(&self, bytes: &[u8]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("uniform_buffer"),
            contents: bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    #[allow(dead_code)]
    pub fn update_buffer<T: Pod>(&self, buffer: &wgpu::Buffer, data: &T) {
        self.queue.write_buffer(buffer, 0, bytemuck::bytes_of(data));
    }

    pub fn read_buffer<T: Pod>(&self, buffer: &wgpu::Buffer, count: usize) -> Vec<T> {
        let size = (std::mem::size_of::<T>() * count) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_buffer"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback_encoder"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let _ = rx.recv().expect("map_async receiver dropped");

        let data = slice.get_mapped_range();
        let out = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        out
    }
}

#[allow(dead_code)]
pub fn assert_uniform_size<T: Pod>(expected_bytes: usize) {
    let actual = std::mem::size_of::<T>();
    assert!(
        actual == expected_bytes,
        "Uniform size mismatch: expected {} bytes, got {} bytes",
        expected_bytes,
        actual
    );
}

pub struct BufferBindingDesc {
    pub binding: u32,
    pub visibility: wgpu::ShaderStages,
    pub ty: wgpu::BufferBindingType,
    pub has_dynamic_offset: bool,
    pub min_binding_size: Option<NonZeroU64>,
}

impl BufferBindingDesc {
    pub fn layout_entry(&self) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding: self.binding,
            visibility: self.visibility,
            ty: wgpu::BindingType::Buffer {
                ty: self.ty,
                has_dynamic_offset: self.has_dynamic_offset,
                min_binding_size: self.min_binding_size,
            },
            count: None,
        }
    }
}

pub struct BufferBinding<'a> {
    pub binding: u32,
    pub buffer: &'a wgpu::Buffer,
    pub offset: u64,
    pub size: Option<NonZeroU64>,
}

impl<'a> BufferBinding<'a> {
    pub fn entry(&self) -> wgpu::BindGroupEntry<'a> {
        wgpu::BindGroupEntry {
            binding: self.binding,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: self.buffer,
                offset: self.offset,
                size: self.size,
            }),
        }
    }
}

pub struct BindingsBuilder<'a> {
    descs: Vec<BufferBindingDesc>,
    resources: Vec<BufferBinding<'a>>,
}

impl<'a> BindingsBuilder<'a> {
    pub fn new() -> Self {
        Self {
            descs: Vec::new(),
            resources: Vec::new(),
        }
    }

    pub fn storage(mut self, binding: u32, buffer: &'a wgpu::Buffer, read_only: bool) -> Self {
        self.descs.push(BufferBindingDesc {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        });
        self.resources.push(BufferBinding {
            binding,
            buffer,
            offset: 0,
            size: None,
        });
        self
    }

    pub fn uniform(mut self, binding: u32, buffer: &'a wgpu::Buffer) -> Self {
        self.descs.push(BufferBindingDesc {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        });
        self.resources.push(BufferBinding {
            binding,
            buffer,
            offset: 0,
            size: None,
        });
        self
    }

    pub fn build(self) -> (Vec<BufferBindingDesc>, Vec<BufferBinding<'a>>) {
        (self.descs, self.resources)
    }
}

pub struct ComputeKernel {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
}

impl ComputeKernel {
    pub fn new(
        ctx: &ComputeContext,
        shader_src: &str,
        entry: &str,
        bindings: &[BufferBindingDesc],
        resources: &[BufferBinding<'_>],
    ) -> Self {
        let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compute_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });
        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("compute_bind_group_layout"),
                entries: &bindings.iter().map(|b| b.layout_entry()).collect::<Vec<_>>(),
            });
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("compute_pipeline_layout"),
                bind_group_layouts: &[&layout],
                immediate_size: 0,
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("compute_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compute_bind_group"),
            layout: &layout,
            entries: &resources.iter().map(|r| r.entry()).collect::<Vec<_>>(),
        });
        Self { pipeline, bind_group }
    }

    pub fn dispatch(&self, ctx: &ComputeContext, x: u32, y: u32, z: u32) {
        let mut encoder =
            ctx.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("compute_encoder"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(x, y, z);
        }
        ctx.queue.submit(Some(encoder.finish()));
    }
}

