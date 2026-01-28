use std::collections::HashMap;

use js_sys::Uint32Array;
use wasm_bindgen::prelude::*;

use crate::lod_manager_core::{flatten_lod_nodes, LodBound, LodChunkManager, LodMetaLod, LodMetaNode};

#[derive(serde::Deserialize)]
struct LodMetaSerde {
    #[serde(rename = "lodLevels")]
    #[allow(dead_code)]
    lod_levels: usize,
    filenames: Vec<String>,
    tree: LodMetaNodeSerde,
}

#[derive(serde::Deserialize)]
struct LodMetaNodeSerde {
    bound: LodBoundSerde,
    children: Option<Vec<LodMetaNodeSerde>>,
    lods: Option<HashMap<String, LodMetaLodSerde>>,
}

#[derive(serde::Deserialize)]
struct LodMetaLodSerde {
    file: u32,
    offset: u32,
    count: u32,
}

#[derive(serde::Deserialize)]
struct LodBoundSerde {
    min: [f32; 3],
    max: [f32; 3],
}

fn convert_node(node: LodMetaNodeSerde) -> LodMetaNode {
    LodMetaNode {
        bound: LodBound {
            min: node.bound.min,
            max: node.bound.max,
        },
        children: node.children.map(|children| children.into_iter().map(convert_node).collect()),
        lods: node.lods.map(|lods| {
            lods.into_iter()
                .map(|(k, v)| {
                    (
                        k,
                        LodMetaLod {
                            file: v.file,
                            offset: v.offset,
                            count: v.count,
                        },
                    )
                })
                .collect()
        }),
    }
}

#[wasm_bindgen(js_name = LodChunkManager)]
pub struct LodChunkManagerWasm {
    inner: LodChunkManager,
}

#[wasm_bindgen]
impl LodChunkManagerWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(meta_json: &str, lod_index: u32, max_requests_per_tick: u32) -> Result<LodChunkManagerWasm, JsValue> {
        let meta: LodMetaSerde =
            serde_json::from_str(meta_json).map_err(|e| JsValue::from_str(&e.to_string()))?;

        let root = convert_node(meta.tree);
        let mut nodes = Vec::new();
        flatten_lod_nodes(&root, lod_index as usize, &mut nodes);

        let manager = LodChunkManager::new(nodes, meta.filenames.len(), max_requests_per_tick as usize);

        Ok(LodChunkManagerWasm { inner: manager })
    }

    pub fn update_view_proj(&mut self, view_proj: &[f32]) -> Result<(), JsValue> {
        self.inner
            .update_view_proj(view_proj)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    pub fn drain_requests(&mut self) -> Uint32Array {
        let requests = self.inner.drain_requests();
        Uint32Array::from(requests.as_slice())
    }

    pub fn mark_loaded(&mut self, file_index: u32) {
        self.inner.mark_loaded(file_index);
    }

    pub fn mark_unrequested(&mut self, file_index: u32) {
        self.inner.mark_unrequested(file_index);
    }

    pub fn file_state(&self, file_index: u32) -> u8 {
        self.inner.file_state(file_index)
    }
}
