import type { LodChunkNode, LodLeaf, LodMeta, LodMetaLod, LodMetaNode } from "./types";

const DEFAULT_LOD_ENTRY: LodMetaLod = { file: -1, offset: 0, count: 0 };

function normalizeLods(
  lods: Record<string, LodMetaLod> | undefined,
  lodLevels: number,
): LodMetaLod[] {
  const entries: LodMetaLod[] = new Array(lodLevels);
  for (let i = 0; i < lodLevels; i += 1) {
    const key = String(i);
    entries[i] = lods?.[key] ?? DEFAULT_LOD_ENTRY;
  }
  return entries;
}

function flattenTree(
  node: LodMetaNode,
  lodLevels: number,
  out: LodLeaf[],
) {
  if (node.lods) {
    out.push({
      id: out.length,
      min: [node.bound.min[0], node.bound.min[1], node.bound.min[2]],
      max: [node.bound.max[0], node.bound.max[1], node.bound.max[2]],
      lods: normalizeLods(node.lods, lodLevels),
    });
    return;
  }

  if (node.children) {
    for (const child of node.children) {
      flattenTree(child, lodLevels, out);
    }
  }
}

export function flattenLodMeta(meta: LodMeta): LodLeaf[] {
  const leaves: LodLeaf[] = [];
  flattenTree(meta.tree, meta.lodLevels, leaves);
  return leaves;
}

export function selectLodNodes(
  leaves: LodLeaf[],
  lodIndex: number,
): LodChunkNode[] {
  const nodes: LodChunkNode[] = [];
  for (const leaf of leaves) {
    const lod = leaf.lods[lodIndex];
    if (!lod || lod.file < 0 || lod.count <= 0) {
      continue;
    }
    nodes.push({
      id: leaf.id,
      min: leaf.min,
      max: leaf.max,
      fileIndex: lod.file,
      offset: lod.offset,
      count: lod.count,
    });
  }
  return nodes;
}

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
