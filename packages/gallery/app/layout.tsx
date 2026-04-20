import "./globals.css";

import type { Metadata } from "next";
import { DemoShell } from "./DemoShell";
import "katex/dist/katex.min.css";

export const metadata: Metadata = { title: "3DGS examples" };

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <DemoShell>{children}</DemoShell>
      </body>
    </html>
  );
}
