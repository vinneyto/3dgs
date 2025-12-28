import { useEffect, useMemo, useState } from "react";
import { getRustWasm } from "../lib/rustWasm";

type State =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "ready"; report: string }
  | { kind: "error"; error: string };

function parseU32(text: string): number {
  const t = text.trim().toLowerCase();
  if (!t) return 0;
  const n = t.startsWith("0x") ? Number.parseInt(t.slice(2), 16) : Number.parseInt(t, 10);
  if (!Number.isFinite(n)) return 0;
  // force u32
  return n >>> 0;
}

export function RustBitOpsPage() {
  const [aText, setAText] = useState("0x12345678");
  const [bText, setBText] = useState("0x0F0F0F0F");
  const [shiftText, setShiftText] = useState("5");
  const [kText, setKText] = useState("7");
  const [state, setState] = useState<State>({ kind: "idle" });
  const [bitState, setBitState] = useState<boolean>(false);
  const [setResult, setSetResult] = useState<number>(0);
  const [hamming, setHamming] = useState<number>(0);
  const [powers, setPowers] = useState<number[]>([]);

  const a = useMemo(() => parseU32(aText), [aText]);
  const b = useMemo(() => parseU32(bText), [bText]);
  const shift = useMemo(() => parseU32(shiftText), [shiftText]);
  const k = useMemo(() => parseU32(kText), [kText]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setState({ kind: "loading" });
      try {
        const mod = await getRustWasm();
        const report = mod.shift_right_report_u32(a, shift);
        const bit = mod.is_bit_set_u32(a, k);
        const next = mod.set_bit_u32(a, k);
        const dist = mod.hamming_distance_u32(a, b);
        const pows = Array.from(mod.powers_of_two_u32(a) as unknown as number[]);
        if (!cancelled) setState({ kind: "ready", report });
        if (!cancelled) setBitState(bit);
        if (!cancelled) setSetResult(next);
        if (!cancelled) setHamming(dist);
        if (!cancelled) setPowers(pows);
      } catch (e) {
        const error = e instanceof Error ? e.message : String(e);
        if (!cancelled) setState({ kind: "error", error });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [a, b, shift, k]);

  return (
    <div style={{ padding: 16 }}>
      <h2>Rust shift-right playground</h2>
      <p>
        The report below is generated in Rust and returned to React as a single string.
        Try decimal or hex inputs (e.g. <code>42</code> or <code>0x2A</code>).
      </p>

      <div style={{ display: "grid", gap: 8, maxWidth: 520 }}>
        <label>
          a (u32):{" "}
          <input
            value={aText}
            onChange={(e) => setAText(e.target.value)}
            style={{ width: "100%" }}
          />
        </label>
        <label>
          b (u32):{" "}
          <input
            value={bText}
            onChange={(e) => setBText(e.target.value)}
            style={{ width: "100%" }}
          />
        </label>
        <label>
          shift:{" "}
          <input
            value={shiftText}
            onChange={(e) => setShiftText(e.target.value)}
            style={{ width: "100%" }}
          />
        </label>
        <label>
          k-th bit to check (0..31):{" "}
          <input
            value={kText}
            onChange={(e) => setKText(e.target.value)}
            style={{ width: "100%" }}
          />
        </label>
      </div>

      {state.kind === "loading" && <div style={{ marginTop: 12 }}>Generating report…</div>}
      {state.kind === "error" && (
        <div style={{ marginTop: 12, color: "crimson" }}>
          <div>Error:</div>
          <pre style={{ whiteSpace: "pre-wrap" }}>{state.error}</pre>
        </div>
      )}
      {state.kind === "ready" && (
        <>
          <div style={{ marginTop: 12 }}>
            <b>is_bit_set_u32(a, k)</b>: {bitState ? "true" : "false"}
          </div>
          <div style={{ marginTop: 8 }}>
            <b>set_bit_u32(a, k)</b>: {setResult} (<code>0x{setResult.toString(16).toUpperCase()}</code>)
          </div>
          <div style={{ marginTop: 8 }}>
            <b>Hamming distance</b>: (a ^ b).count_ones() = {hamming}{" "}
            <span style={{ opacity: 0.8 }}>
              (a ^ b = <code>0x{(a ^ b).toString(16).toUpperCase()}</code>)
            </span>
          </div>
          <div style={{ marginTop: 8 }}>
            <b>Powers of two in a</b>:{" "}
            <code>[{powers.join(", ")}]</code>
          </div>
          <pre
            style={{
              marginTop: 12,
              padding: 12,
              border: "1px solid rgba(255,255,255,0.12)",
              borderRadius: 8,
              overflow: "auto",
              maxHeight: "60vh",
            }}
          >
            {state.report}
          </pre>
        </>
      )}
    </div>
  );
}


