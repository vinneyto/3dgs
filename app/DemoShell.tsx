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
  { href: "/photo-ply", label: "Photo PLY (WIP)" },
  { href: "/ref-splats", label: "Ref splats (PLY)" },
  { href: "/rust-wasm", label: "PLY parse (Rust WASM)" },
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
  return (
    <div className="appShell appShellDesktop">
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
    <DesktopShell pathname={pathname}>{children}</DesktopShell>
  );
}
