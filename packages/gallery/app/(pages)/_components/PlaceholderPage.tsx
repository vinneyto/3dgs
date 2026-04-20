"use client";

import type { ReactNode } from "react";

export function PlaceholderPage({
  title,
  children,
}: {
  title: string;
  children?: ReactNode;
}) {
  return (
    <div className="page">
      <div className="pageHeader">
        <h1>{title}</h1>
        <p className="muted">TODO: перенести эту демку из `tmp/src/pages`.</p>
        {children}
      </div>
    </div>
  );
}
