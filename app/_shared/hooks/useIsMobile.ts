"use client";

import { useEffect, useState } from "react";

const DEFAULT_BREAKPOINT_PX = 820;

export function useIsMobile(
  breakpointPx: number = DEFAULT_BREAKPOINT_PX
): boolean {
  const query = `(max-width: ${breakpointPx}px)`;

  const [isMobile, setIsMobile] = useState(() => {
    if (typeof window === "undefined") return false;
    return window.matchMedia(query).matches;
  });

  useEffect(() => {
    const mq = window.matchMedia(query);
    const onChange = () => setIsMobile(mq.matches);

    mq.addEventListener?.("change", onChange);
    return () => mq.removeEventListener?.("change", onChange);
  }, [query]);

  return isMobile;
}
