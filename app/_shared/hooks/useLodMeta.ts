import { useEffect, useState } from "react";
import type { LodMeta } from "../lod/types";

export type LodMetaState =
  | { status: "idle"; meta: null; error: null }
  | { status: "loading"; meta: null; error: null }
  | { status: "ready"; meta: LodMeta; error: null }
  | { status: "error"; meta: null; error: string };

export function useLodMeta(url: string): LodMetaState {
  const [state, setState] = useState<LodMetaState>({
    status: "idle",
    meta: null,
    error: null,
  });

  useEffect(() => {
    let cancelled = false;
    const ac = new AbortController();

    setState({ status: "loading", meta: null, error: null });

    (async () => {
      try {
        const res = await fetch(url, { signal: ac.signal });
        if (!res.ok) {
          throw new Error(`fetch failed: ${res.status} ${res.statusText}`);
        }
        const meta = (await res.json()) as LodMeta;
        if (!cancelled) {
          setState({ status: "ready", meta, error: null });
        }
      } catch (err) {
        if (cancelled || (err as Error)?.name === "AbortError") {
          return;
        }
        const message = err instanceof Error ? err.message : String(err);
        setState({ status: "error", meta: null, error: message });
      }
    })();

    return () => {
      cancelled = true;
      ac.abort();
    };
  }, [url]);

  return state;
}
