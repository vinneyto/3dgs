type RustWasmModule = typeof import("../wasm/pkg/rust_wasm");

let cached: Promise<RustWasmModule> | null = null;

export async function getRustWasm(): Promise<RustWasmModule> {
  if (cached) return cached;

  cached = (async () => {
    const mod = await import("../wasm/pkg/rust_wasm");
    // wasm-bindgen init now prefers an object parameter to avoid a deprecation warning.
    await mod.default({
      module_or_path: new URL("../wasm/pkg/rust_wasm_bg.wasm", import.meta.url),
    });
    return mod;
  })();

  return cached;
}
