use std::collections::HashMap;

use js_sys::Uint32Array;
use wasm_bindgen::prelude::*;

use crate::lod_manager_core::{collect_lod_nodes, LodBound, LodMetaLod, LodMetaNode, LodTileQuery};

#[derive(serde::Deserialize)]
struct LodMetaSerde {
    #[serde(rename = "lodLevels")]
    #[allow(dead_code)]
    lod_levels: usize,
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

#[wasm_bindgen(js_name = LodTileQuery)]
pub struct LodTileQueryWasm {
    inner: LodTileQuery,
}

#[wasm_bindgen]
impl LodTileQueryWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(meta_json: &str, lod_index: u32) -> Result<LodTileQueryWasm, JsValue> {
        let meta: LodMetaSerde =
            serde_json::from_str(meta_json).map_err(|e| JsValue::from_str(&e.to_string()))?;

        let root = convert_node(meta.tree);
        let mut nodes = Vec::new();
        let mut next_id = 0u32;
        collect_lod_nodes(&root, lod_index as usize, &mut nodes, &mut next_id);

        let query = LodTileQuery::new(nodes);

        Ok(LodTileQueryWasm { inner: query })
    }

    pub fn query_view_proj(&self, view_proj: &[f32]) -> Result<Uint32Array, JsValue> {
        let result = self
            .inner
            .query_view_proj(view_proj)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Uint32Array::from(result.as_slice()))
    }
}
