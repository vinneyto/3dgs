use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[wasm_bindgen]
pub fn greet(name: &str) -> String {
    format!("Привет, {name}!")
}

#[wasm_bindgen]
pub fn fib(n: u32) -> u32 {
    // Итеративно (быстрее и без рекурсии)
    let mut a = 0u32;
    let mut b = 1u32;
    for _ in 0..n {
        let next = a.wrapping_add(b);
        a = b;
        b = next;
    }
    a
}

#[wasm_bindgen]
pub fn dot(a: &[f32], b: &[f32]) -> Result<f32, JsValue> {
    if a.len() != b.len() {
        return Err(JsValue::from_str("dot(): arrays must have the same length"));
    }

    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    Ok(sum)
}
