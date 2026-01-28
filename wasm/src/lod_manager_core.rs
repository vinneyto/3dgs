use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct LodNode {
    pub id: u32,
    pub min: [f32; 3],
    pub max: [f32; 3],
}

#[derive(Clone, Debug)]
pub enum LodManagerError {
    InvalidMatrix,
    InvalidMeta(String),
}

impl std::fmt::Display for LodManagerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LodManagerError::InvalidMatrix => write!(f, "LOD: expected 4x4 view-projection matrix"),
            LodManagerError::InvalidMeta(msg) => write!(f, "LOD: invalid meta json: {msg}"),
        }
    }
}

impl std::error::Error for LodManagerError {}

#[derive(Clone, Copy, Debug)]
struct Plane {
    nx: f32,
    ny: f32,
    nz: f32,
    d: f32,
}

impl Plane {
    fn normalize(&mut self) {
        let len = (self.nx * self.nx + self.ny * self.ny + self.nz * self.nz).sqrt();
        if len > 0.0 {
            let inv = 1.0 / len;
            self.nx *= inv;
            self.ny *= inv;
            self.nz *= inv;
            self.d *= inv;
        }
    }

    fn test_aabb(&self, min: &[f32; 3], max: &[f32; 3]) -> bool {
        let px = if self.nx >= 0.0 { max[0] } else { min[0] };
        let py = if self.ny >= 0.0 { max[1] } else { min[1] };
        let pz = if self.nz >= 0.0 { max[2] } else { min[2] };
        (self.nx * px + self.ny * py + self.nz * pz + self.d) >= 0.0
    }
}

#[derive(Clone, Copy, Debug)]
struct Frustum {
    planes: [Plane; 6],
}

impl Frustum {
    fn from_view_proj(m: &[f32; 16]) -> Self {
        let mut planes = [
            Plane { nx: m[3] - m[0], ny: m[7] - m[4], nz: m[11] - m[8], d: m[15] - m[12] },
            Plane { nx: m[3] + m[0], ny: m[7] + m[4], nz: m[11] + m[8], d: m[15] + m[12] },
            Plane { nx: m[3] + m[1], ny: m[7] + m[5], nz: m[11] + m[9], d: m[15] + m[13] },
            Plane { nx: m[3] - m[1], ny: m[7] - m[5], nz: m[11] - m[9], d: m[15] - m[13] },
            Plane { nx: m[3] - m[2], ny: m[7] - m[6], nz: m[11] - m[10], d: m[15] - m[14] },
            Plane { nx: m[3] + m[2], ny: m[7] + m[6], nz: m[11] + m[10], d: m[15] + m[14] },
        ];
        for plane in &mut planes {
            plane.normalize();
        }
        Self { planes }
    }

    fn intersects_aabb(&self, min: &[f32; 3], max: &[f32; 3]) -> bool {
        for plane in &self.planes {
            if !plane.test_aabb(min, max) {
                return false;
            }
        }
        true
    }
}

#[derive(Clone, Debug)]
pub struct LodTileQuery {
    nodes: Vec<LodNode>,
}

impl LodTileQuery {
    pub fn new(nodes: Vec<LodNode>) -> Self {
        Self { nodes }
    }

    pub fn query_view_proj(&self, view_proj: &[f32]) -> Result<Vec<u32>, LodManagerError> {
        if view_proj.len() != 16 {
            return Err(LodManagerError::InvalidMatrix);
        }
        let mut m = [0.0f32; 16];
        m.copy_from_slice(view_proj);
        let frustum = Frustum::from_view_proj(&m);
        let mut result = Vec::new();
        for node in &self.nodes {
            if frustum.intersects_aabb(&node.min, &node.max) {
                result.push(node.id);
            }
        }
        Ok(result)
    }
}

#[derive(Clone, Debug)]
pub struct LodMetaNode {
    pub bound: LodBound,
    pub children: Option<Vec<LodMetaNode>>,
    pub lods: Option<HashMap<String, LodMetaLod>>,
}

#[derive(Clone, Debug)]
pub struct LodMetaLod {
    pub file: u32,
    pub offset: u32,
    pub count: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct LodBound {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

pub fn collect_lod_nodes(
    tree: &LodMetaNode,
    lod_index: usize,
    out: &mut Vec<LodNode>,
    next_id: &mut u32,
) {
    if let Some(lods) = &tree.lods {
        if let Some(lod) = lods.get(&lod_index.to_string()) {
            if lod.count > 0 {
                out.push(LodNode {
                    id: *next_id,
                    min: tree.bound.min,
                    max: tree.bound.max,
                });
                *next_id += 1;
            }
        }
        return;
    }

    if let Some(children) = &tree.children {
        for child in children {
            collect_lod_nodes(child, lod_index, out, next_id);
        }
    }
}
