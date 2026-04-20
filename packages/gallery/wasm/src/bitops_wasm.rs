use wasm_bindgen::prelude::*;

use crate::bitops_core;

#[wasm_bindgen]
pub fn shift_right_report_u32(a: u32, shift: u32) -> String {
    bitops_core::shift_right_report_u32(a, shift)
}

#[wasm_bindgen]
pub fn is_bit_set_u32(a: u32, k: u32) -> bool {
    bitops_core::is_bit_set_u32(a, k)
}

#[wasm_bindgen]
pub fn set_bit_u32(a: u32, k: u32) -> u32 {
    bitops_core::set_bit_u32(a, k)
}

#[wasm_bindgen]
pub fn hamming_distance_u32(a: u32, b: u32) -> u32 {
    bitops_core::hamming_distance_u32(a, b)
}

#[wasm_bindgen]
pub fn powers_of_two_u32(a: u32) -> js_sys::Array {
    let arr = js_sys::Array::new();
    for v in bitops_core::powers_of_two_u32(a) {
        arr.push(&JsValue::from(v));
    }
    arr
}


