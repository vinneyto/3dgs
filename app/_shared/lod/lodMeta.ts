import type { LodMeta } from "./types";

export function resolveLodFileUrls(meta: LodMeta, metaUrl: string): string[] {
  const baseUrl =
    typeof window !== "undefined"
      ? new URL(metaUrl, window.location.href)
      : new URL(metaUrl, "http://localhost/");
  return meta.filenames.map((filename) => {
    if (filename.startsWith("http://") || filename.startsWith("https://")) {
      return filename;
    }
    if (filename.startsWith("/")) {
      return filename;
    }
    return new URL(filename, baseUrl).toString();
  });
}
