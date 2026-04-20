import type { ReactNode } from "react";
import { InlineMath, BlockMath } from "react-katex";

type Token =
  | { kind: "text"; value: string }
  | { kind: "inline"; value: string }
  | { kind: "block"; value: string };

function tokenizeMath(input: string): Token[] {
  // Supports: \( ... \) (inline), \[ ... \] (block), $$ ... $$ (block)
  const tokens: Token[] = [];
  let i = 0;

  const pushText = (s: string) => {
    if (s) tokens.push({ kind: "text", value: s });
  };

  while (i < input.length) {
    const nextInline = input.indexOf("\\(", i);
    const nextBlock1 = input.indexOf("\\[", i);
    const nextBlock2 = input.indexOf("$$", i);

    let next = -1;
    let kind: "inline" | "block1" | "block2" | null = null;

    const candidates = [
      { pos: nextInline, kind: "inline" as const },
      { pos: nextBlock1, kind: "block1" as const },
      { pos: nextBlock2, kind: "block2" as const },
    ].filter((c) => c.pos !== -1);

    if (candidates.length) {
      candidates.sort((a, b) => a.pos - b.pos);
      next = candidates[0].pos;
      kind = candidates[0].kind;
    }

    if (next === -1 || kind === null) {
      pushText(input.slice(i));
      break;
    }

    // text before delimiter
    pushText(input.slice(i, next));

    if (kind === "inline") {
      const end = input.indexOf("\\)", next + 2);
      if (end === -1) {
        pushText(input.slice(next));
        break;
      }
      const math = input.slice(next + 2, end);
      tokens.push({ kind: "inline", value: math });
      i = end + 2;
      continue;
    }

    if (kind === "block1") {
      const end = input.indexOf("\\]", next + 2);
      if (end === -1) {
        pushText(input.slice(next));
        break;
      }
      const math = input.slice(next + 2, end);
      tokens.push({ kind: "block", value: math });
      i = end + 2;
      continue;
    }

    // kind === "block2"
    {
      const end = input.indexOf("$$", next + 2);
      if (end === -1) {
        pushText(input.slice(next));
        break;
      }
      const math = input.slice(next + 2, end);
      tokens.push({ kind: "block", value: math });
      i = end + 2;
    }
  }

  return tokens;
}

export function MathText({ children }: { children: string }): ReactNode {
  const tokens = tokenizeMath(children);

  return (
    <>
      {tokens.map((t, idx) => {
        if (t.kind === "text") return <span key={idx}>{t.value}</span>;
        if (t.kind === "inline") return <InlineMath key={idx} math={t.value} />;
        return <BlockMath key={idx} math={t.value} />;
      })}
    </>
  );
}
