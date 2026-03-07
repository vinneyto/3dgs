"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import type { ReactNode } from "react";
import { useEffect, useState } from "react";
import { useIsMobile } from "@/app/_shared/hooks/useIsMobile";

const demos = [
  { href: "/covariance", label: "Covariance (debug)" },
  { href: "/splat-quad", label: "Splat quad (debug)" },
  { href: "/compare", label: "Compare (ellipsoid + sprite)" },
  { href: "/instanced", label: "Instanced splats (storage buffer)" },
  { href: "/billboard-circles", label: "Billboard circles (TSL)" },
  {
    href: "/billboard-circles-buffer",
    label: "Billboard circles (storage buffer)",
  },
  { href: "/ply-header", label: "PLY header (cactus)" },
  { href: "/ply-ellipsoids", label: "PLY ellipsoids (instanced)" },
  { href: "/ply-gaussians", label: "PLY gaussians (WIP)" },
  { href: "/ply-gaussians-gpu-culling", label: "PLY gaussians (GPU culling)" },
  { href: "/photo-ply", label: "Photo PLY (WIP)" },
  { href: "/ref-splats", label: "Ref splats (PLY)" },
  { href: "/prefix-sum", label: "Prefix sum (CPU / TS)" },
  { href: "/radix-sort-webgpu", label: "Radix sort (WebGPU setup)" },
  { href: "/radix-sort-three-tsl", label: "Radix sort (three.js + TSL)" },
  { href: "/rust-wasm", label: "PLY parse (Rust WASM)" },
  { href: "/rust-sogs-v2", label: "SOGS v2 parse (Rust WASM)" },
  { href: "/room-sog", label: "Room SOG render (Rust WASM)" },
  { href: "/room-sog-gpu-culling", label: "Room SOG (GPU culling)" },
  { href: "/truck-high-ply", label: "Truck PLY render (Rust WASM)" },
  { href: "/truck-high-ply-gpu-culling", label: "Truck PLY (GPU culling)" },
  { href: "/rust-bitops", label: "Rust bitwise ops" },
  { href: "/sh-sphere", label: "SH sphere (widget)" },
];

function MobileShell({
  children,
  pathname,
}: {
  children: ReactNode;
  pathname: string;
}) {
  const router = useRouter();

  return (
    <div className="appShell appShellMobile">
      <header className="mobileHeader">
        <div className="mobileTitle">3DGS</div>
        <select
          className="mobileSelect"
          value={pathname}
          onChange={(e) => router.push(e.target.value)}
          aria-label="Select demo"
        >
          {demos.map(({ href, label }) => (
            <option key={href} value={href}>
              {label}
            </option>
          ))}
        </select>
      </header>
      <main className="content contentMobile">{children}</main>
    </div>
  );
}

function DesktopShell({
  children,
  pathname,
}: {
  children: ReactNode;
  pathname: string;
}) {
  const [query, setQuery] = useState("");
  const q = query.trim().toLowerCase();
  const filteredDemos = q
    ? demos.filter(({ href, label }) => {
        const hay = `${label} ${href}`.toLowerCase();
        return hay.includes(q);
      })
    : demos;

  return (
    <div className="appShell appShellDesktop">
      <aside className="sidebar">
        <div className="sidebarHeader">3DGS examples</div>
        <div className="sidebarSearch">
          <input
            className="sidebarSearchInput"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search demos…"
            aria-label="Search demos"
          />
        </div>
        <nav className="nav navScroll" aria-label="Demo list">
          {filteredDemos.length ? (
            filteredDemos.map(({ href, label }) => {
              const isActive = pathname === href;
              return (
                <Link
                  key={href}
                  href={href}
                  className={`navLink${isActive ? " active" : ""}`}
                >
                  {label}
                </Link>
              );
            })
          ) : (
            <div className="navEmpty">No demos match “{query.trim()}”.</div>
          )}
        </nav>
      </aside>
      <main className="content">{children}</main>
    </div>
  );
}

export function DemoShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const [hydrated, setHydrated] = useState(false);
  const isMobile = useIsMobile(820);

  // Avoid hydration mismatch: server cannot know viewport width, so defer shell
  // selection to the client after hydration.
  useEffect(() => {
    const raf = window.requestAnimationFrame(() => setHydrated(true));
    return () => window.cancelAnimationFrame(raf);
  }, []);

  if (!hydrated) {
    return <div className="appBoot" />;
  }

  return isMobile ? (
    <MobileShell pathname={pathname}>{children}</MobileShell>
  ) : (
    <DesktopShell key={pathname} pathname={pathname}>
      {children}
    </DesktopShell>
  );
}
