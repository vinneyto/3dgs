pub mod ply_splat_core;
pub mod bitops_core;

#[cfg(target_arch = "wasm32")]
mod ply_splat_wasm;

#[cfg(target_arch = "wasm32")]
mod bitops_wasm;

pub use ply_splat_core::{
    parse_splat_ply_core, parse_splat_ply_core_with_opts, PlyError, PlyFormat, SplatPlyBuffersCore,
};

#[cfg(target_arch = "wasm32")]
pub use ply_splat_wasm::{parse_splat_ply, parse_splat_ply_with_opts, SplatPlyBuffers};

pub use bitops_core::shift_right_report_u32 as shift_right_report_u32_core;
pub use bitops_core::is_bit_set_u32 as is_bit_set_u32_core;
pub use bitops_core::set_bit_u32 as set_bit_u32_core;
pub use bitops_core::hamming_distance_u32 as hamming_distance_u32_core;
pub use bitops_core::powers_of_two_u32 as powers_of_two_u32_core;

#[cfg(target_arch = "wasm32")]
pub use bitops_wasm::shift_right_report_u32;

#[cfg(target_arch = "wasm32")]
pub use bitops_wasm::is_bit_set_u32;

#[cfg(target_arch = "wasm32")]
pub use bitops_wasm::set_bit_u32;

#[cfg(target_arch = "wasm32")]
pub use bitops_wasm::hamming_distance_u32;

#[cfg(target_arch = "wasm32")]
pub use bitops_wasm::powers_of_two_u32;
