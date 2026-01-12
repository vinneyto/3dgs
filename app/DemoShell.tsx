"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";

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
  { href: "/photo-ply", label: "Photo PLY (WIP)" },
  { href: "/ref-splats", label: "Ref splats (PLY)" },
  { href: "/rust-wasm", label: "PLY parse (Rust WASM)" },
  { href: "/rust-bitops", label: "Rust bitwise ops" },
  { href: "/sh-sphere", label: "SH sphere (widget)" },
];

export function DemoShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();

  return (
    <div className="appShell">
      <aside className="sidebar">
        <div className="sidebarHeader">3DGS examples</div>
        <nav className="nav">
          {demos.map(({ href, label }) => {
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
          })}
        </nav>
      </aside>
      <main className="content">{children}</main>
    </div>
  );
}
