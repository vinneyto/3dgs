import { useEffect, useMemo, useState } from "react";

type DemoState =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "ready"; message: string; add: number; fib: number; dot: number }
  | { kind: "error"; error: string };

export function RustWasmDemoPage() {
  const [state, setState] = useState<DemoState>({ kind: "idle" });

  const inputA = useMemo(() => new Float32Array([1, 2, 3, 4]), []);
  const inputB = useMemo(() => new Float32Array([10, 20, 30, 40]), []);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      setState({ kind: "loading" });
      try {
        const mod = await import("../wasm/pkg/rust_wasm");
        await mod.default(
          new URL("../wasm/pkg/rust_wasm_bg.wasm", import.meta.url)
        );

        const message = mod.greet("WASM");
        const add = mod.add(2, 40);
        const fib = mod.fib(10);
        const dot = mod.dot(inputA, inputB);

        if (!cancelled) {
          setState({ kind: "ready", message, add, fib, dot });
        }
      } catch (e) {
        const error = e instanceof Error ? e.message : String(e);
        if (!cancelled) setState({ kind: "error", error });
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [inputA, inputB]);

  return (
    <div style={{ padding: 16 }}>
      <h2>Rust → WASM demo</h2>
      <p>
        Эта страница загружает WASM, собранный из Rust (через{" "}
        <code>wasm-pack</code>), и вызывает экспортированные функции.
      </p>

      {state.kind === "idle" && <div>Idle…</div>}
      {state.kind === "loading" && <div>Loading WASM…</div>}
      {state.kind === "error" && (
        <div style={{ color: "crimson" }}>
          <div>WASM error:</div>
          <pre style={{ whiteSpace: "pre-wrap" }}>{state.error}</pre>
        </div>
      )}
      {state.kind === "ready" && (
        <div style={{ display: "grid", gap: 8, maxWidth: 640 }}>
          <div>
            <b>greet()</b>: {state.message}
          </div>
          <div>
            <b>add(2, 40)</b>: {state.add}
          </div>
          <div>
            <b>fib(10)</b>: {state.fib}
          </div>
          <div>
            <b>dot([1,2,3,4], [10,20,30,40])</b>: {state.dot}
          </div>
        </div>
      )}
    </div>
  );
}
