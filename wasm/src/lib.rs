pub mod splats;
pub mod splats_parser;
pub mod parsers;
pub mod bitops_core;
pub mod octree;

#[cfg(target_arch = "wasm32")]
mod splats_wasm;

#[cfg(target_arch = "wasm32")]
mod bitops_wasm;

pub use splats::Splats;

#[cfg(target_arch = "wasm32")]
pub use splats_wasm::{
    parse_splat_ply, parse_splat_ply_with_opts, parse_splat_sogs_v2, parse_splats_auto,
    SplatsBuffers,
};

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
