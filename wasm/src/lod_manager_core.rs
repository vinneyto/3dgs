use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct LodNode {
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub file_index: u32,
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
pub struct LodChunkManager {
    nodes: Vec<LodNode>,
    file_states: Vec<u8>,
    pending: Vec<u32>,
    max_requests_per_tick: usize,
}

impl LodChunkManager {
    pub fn new(nodes: Vec<LodNode>, file_count: usize, max_requests_per_tick: usize) -> Self {
        Self {
            nodes,
            file_states: vec![0u8; file_count],
            pending: Vec::new(),
            max_requests_per_tick: max_requests_per_tick.max(1),
        }
    }

    pub fn update_view_proj(&mut self, view_proj: &[f32]) -> Result<(), LodManagerError> {
        if view_proj.len() != 16 {
            return Err(LodManagerError::InvalidMatrix);
        }
        let mut m = [0.0f32; 16];
        m.copy_from_slice(view_proj);
        let frustum = Frustum::from_view_proj(&m);

        let mut emitted = 0usize;
        for node in &self.nodes {
            if emitted >= self.max_requests_per_tick {
                break;
            }
            if !frustum.intersects_aabb(&node.min, &node.max) {
                continue;
            }
            let state = self.file_states[node.file_index as usize];
            if state != 0 {
                continue;
            }
            self.file_states[node.file_index as usize] = 1;
            self.pending.push(node.file_index);
            emitted += 1;
        }
        Ok(())
    }

    pub fn drain_requests(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.pending)
    }

    pub fn mark_loaded(&mut self, file_index: u32) {
        if let Some(state) = self.file_states.get_mut(file_index as usize) {
            *state = 2;
        }
    }

    pub fn mark_unrequested(&mut self, file_index: u32) {
        if let Some(state) = self.file_states.get_mut(file_index as usize) {
            *state = 0;
        }
    }

    pub fn file_state(&self, file_index: u32) -> u8 {
        self.file_states.get(file_index as usize).copied().unwrap_or(0)
    }
}

#[derive(Clone, Debug)]
pub struct LodMeta {
    pub lod_levels: usize,
    pub filenames: Vec<String>,
    pub tree: LodMetaNode,
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

pub fn flatten_lod_nodes(
    tree: &LodMetaNode,
    lod_index: usize,
    out: &mut Vec<LodNode>,
) {
    if let Some(lods) = &tree.lods {
        if let Some(lod) = lods.get(&lod_index.to_string()) {
            if lod.file != u32::MAX && lod.count > 0 {
                out.push(LodNode {
                    min: tree.bound.min,
                    max: tree.bound.max,
                    file_index: lod.file,
                });
            }
        }
        return;
    }

    if let Some(children) = &tree.children {
        for child in children {
            flatten_lod_nodes(child, lod_index, out);
        }
    }
}
