export type WgslMessage = {
  lineNum: number;
  linePos: number;
  offset: number;
  length: number;
  message: string;
  type: string;
};

export function excerptLines(
  code: string,
  lineNum1Based: number,
  radius = 2,
): string {
  const lines = code.split("\n");
  const line0 = Math.max(1, lineNum1Based | 0);
  const start = Math.max(1, line0 - radius);
  const end = Math.min(lines.length, line0 + radius);
  const width = String(end).length;
  const out: string[] = [];
  for (let i = start; i <= end; i++) {
    const prefix = i === line0 ? ">" : " ";
    out.push(
      `${prefix} ${String(i).padStart(width, " ")}|${lines[i - 1] ?? ""}`,
    );
  }
  return out.join("\n");
}

function extractLineNumFromMessage(msg: string): number | null {
  const m = msg.match(/line\s+(\d+)/i);
  if (!m) return null;
  const n = Number(m[1]);
  return Number.isFinite(n) && n > 0 ? (n | 0) : null;
}

export function wrapWgslSourceError({
  label,
  stage,
  code,
  error,
}: {
  label: string;
  stage: string;
  code: string;
  error: unknown;
}): Error {
  const msg = error instanceof Error ? error.message : String(error);
  const line = extractLineNumFromMessage(msg);
  const ex = line ? `\n${excerptLines(code, line, 4)}` : "";
  return new Error(`[webgpu] WGSL error in "${label}" at ${stage}: ${msg}${ex}`);
}

function formatCompilationInfo(code: string, messages: WgslMessage[]): string {
  const errors = messages.filter((m) => m.type === "error");
  const warnings = messages.filter((m) => m.type === "warning");
  const notes = messages.filter(
    (m) => m.type !== "error" && m.type !== "warning",
  );

  const parts: string[] = [];
  if (errors.length) parts.push(`errors: ${errors.length}`);
  if (warnings.length) parts.push(`warnings: ${warnings.length}`);
  if (notes.length) parts.push(`notes: ${notes.length}`);

  const header = parts.length ? parts.join(", ") : "no messages";

  const msgs = [...errors, ...warnings, ...notes].slice(0, 20);
  const formatted = msgs
    .map((m) => {
      const where =
        m.lineNum && m.linePos ? `L${m.lineNum}:${m.linePos}` : "(no location)";
      const ex = m.lineNum ? `\n${excerptLines(code, m.lineNum)}` : "";
      return `- ${m.type} ${where}: ${m.message}${ex}`;
    })
    .join("\n");

  return `${header}\n${formatted}`.trim();
}

export async function createComputePipelineWithWgslDebug({
  device,
  label,
  code,
  layout,
  entryPoint = "main",
}: {
  device: GPUDevice;
  label: string;
  code: string;
  layout: GPUPipelineLayout;
  entryPoint?: string;
}): Promise<{ shaderModule: GPUShaderModule; pipeline: GPUComputePipeline }> {
  let shaderModule: GPUShaderModule;
  try {
    shaderModule = device.createShaderModule({
      label: `${label}_wgsl`,
      code,
    });
  } catch (e) {
    throw wrapWgslSourceError({
      label,
      stage: "device.createShaderModule",
      code,
      error: e,
    });
  }

  // Some environments may not expose compilation info; best-effort only.
  const getInfo = async (): Promise<WgslMessage[] | null> => {
    const anyModule = shaderModule as unknown as {
      getCompilationInfo?: () => Promise<{ messages: WgslMessage[] }>;
    };
    if (!anyModule.getCompilationInfo) return null;
    const info = await anyModule.getCompilationInfo();
    return info.messages ?? [];
  };

  const messages = await getInfo();
  const hasErrors = !!messages?.some((m) => m.type === "error");
  if (hasErrors) {
    const details = formatCompilationInfo(code, messages ?? []);
    throw new Error(
      `[webgpu] WGSL compilation failed for "${label}".\n${details}`,
    );
  }

  try {
    const anyDevice = device as unknown as {
      createComputePipelineAsync?: (
        desc: GPUComputePipelineDescriptor,
      ) => Promise<GPUComputePipeline>;
    };

    const desc: GPUComputePipelineDescriptor = {
      label,
      layout,
      compute: { module: shaderModule, entryPoint },
    };

    const pipeline = anyDevice.createComputePipelineAsync
      ? await anyDevice.createComputePipelineAsync(desc)
      : device.createComputePipeline(desc);
    return { shaderModule, pipeline };
  } catch (e) {
    const msgs2 = (await getInfo()) ?? messages ?? [];
    const details = msgs2.length ? `\n${formatCompilationInfo(code, msgs2)}` : "";
    const errMsg = e instanceof Error ? e.message : String(e);
    throw new Error(
      `[webgpu] Failed to create compute pipeline for "${label}": ${errMsg}${details}`,
    );
  }
}

