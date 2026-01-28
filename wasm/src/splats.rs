#[derive(Debug, Clone)]
pub struct Splats {
    pub count: u32,
    /// Human-readable source format identifier (e.g. "binary_little_endian", "ascii", "sogs_v2").
    pub format: String,
    pub center: Box<[f32]>,        // 3N
    pub covariance: Box<[f32]>,    // 6N
    pub rgba: Box<[u32]>,          // N (packed RGBA8)
    pub sh_coeffs_l1: Box<[f32]>,  // 9 * N (3 coeffs * RGB)
    pub sh_coeffs_l2_packed: Box<[u32]>, // 10 * N (packed i16 vec3 -> 2 u32)
    pub sh_coeffs_l2_scale: f32,
    pub sh_coeffs_l3_packed: Box<[u32]>, // 14 * N (packed i16 vec3 -> 2 u32)
    pub sh_coeffs_l3_scale: f32,
    pub sh_degree: u32,
    pub bbox_min: [f32; 3],
    pub bbox_max: [f32; 3],
}

