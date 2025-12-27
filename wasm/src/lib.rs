pub mod ply_splat_core;

#[cfg(target_arch = "wasm32")]
mod ply_splat_wasm;

pub use ply_splat_core::{
    parse_splat_ply_core, parse_splat_ply_core_with_opts, PlyError, PlyFormat, SplatPlyBuffersCore,
};

#[cfg(target_arch = "wasm32")]
pub use ply_splat_wasm::{parse_splat_ply, parse_splat_ply_with_opts, SplatPlyBuffers};
