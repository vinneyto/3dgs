export type LodMetaLod = {
  file: number;
  offset: number;
  count: number;
};

export type LodMetaNode = {
  bound: {
    min: number[];
    max: number[];
  };
  children?: LodMetaNode[];
  lods?: Record<string, LodMetaLod>;
};

export type LodMeta = {
  lodLevels: number;
  environment?: string | null;
  filenames: string[];
  tree: LodMetaNode;
};

export type LodLeaf = {
  id: number;
  min: [number, number, number];
  max: [number, number, number];
  lods: LodMetaLod[];
};

export type LodChunkNode = {
  id: number;
  min: [number, number, number];
  max: [number, number, number];
  fileIndex: number;
  offset: number;
  count: number;
};
