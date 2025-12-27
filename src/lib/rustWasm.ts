type RustWasmModule = typeof import("../wasm/pkg/rust_wasm");

let cached: Promise<RustWasmModule> | null = null;

export async function getRustWasm(): Promise<RustWasmModule> {
  if (cached) return cached;

  cached = (async () => {
    const mod = await import("../wasm/pkg/rust_wasm");
    await mod.default(new URL("../wasm/pkg/rust_wasm_bg.wasm", import.meta.url));
    return mod;
  })();

  return cached;
}


