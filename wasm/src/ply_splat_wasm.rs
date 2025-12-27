use wasm_bindgen::prelude::*;

use crate::ply_splat_core::{parse_splat_ply_core, parse_splat_ply_core_with_opts, SplatPlyBuffersCore};

#[wasm_bindgen]
pub struct SplatPlyBuffers {
    inner: SplatPlyBuffersCore,
}

#[wasm_bindgen]
impl SplatPlyBuffers {
    #[wasm_bindgen(getter)]
    pub fn count(&self) -> u32 {
        self.inner.count
    }

    #[wasm_bindgen(getter)]
    pub fn format(&self) -> String {
        self.inner.format.as_str().to_string()
    }

    #[wasm_bindgen(getter)]
    pub fn center(&self) -> js_sys::Float32Array {
        unsafe { js_sys::Float32Array::view(&self.inner.center) }
    }

    #[wasm_bindgen(getter)]
    pub fn covariance(&self) -> js_sys::Float32Array {
        unsafe { js_sys::Float32Array::view(&self.inner.covariance) }
    }

    #[wasm_bindgen(getter)]
    pub fn rgba(&self) -> js_sys::Uint32Array {
        unsafe { js_sys::Uint32Array::view(&self.inner.rgba) }
    }

    #[wasm_bindgen(getter, js_name = bboxMin)]
    pub fn bbox_min(&self) -> js_sys::Float32Array {
        unsafe { js_sys::Float32Array::view(&self.inner.bbox_min) }
    }

    #[wasm_bindgen(getter, js_name = bboxMax)]
    pub fn bbox_max(&self) -> js_sys::Float32Array {
        unsafe { js_sys::Float32Array::view(&self.inner.bbox_max) }
    }
}

#[wasm_bindgen]
pub fn parse_splat_ply(bytes: &[u8]) -> Result<SplatPlyBuffers, JsValue> {
    let inner = parse_splat_ply_core(bytes).map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(SplatPlyBuffers { inner })
}

#[wasm_bindgen]
pub fn parse_splat_ply_with_opts(
    bytes: &[u8],
    assume_log_scale: bool,
    assume_logit_opacity: bool,
) -> Result<SplatPlyBuffers, JsValue> {
    let inner = parse_splat_ply_core_with_opts(bytes, assume_log_scale, assume_logit_opacity)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(SplatPlyBuffers { inner })
}


