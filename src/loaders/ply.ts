// splat_ply_parser.ts
// Minimal PLY parser for 3D Gaussian Splats:
// - Reads PLY header (ascii/binary_little_endian/binary_big_endian)
// - Extracts: center (x,y,z), covariance (6 via scale+quat), rgba (optional rgb + opacity)
// - Ignores spherical harmonics (f_dc / f_rest) intentionally.
//
// Usage:
//   import { parseSplatPly } from "./splat_ply_parser";
//   const bytes = new Uint8Array(arrayBuffer);
//   const splat = parseSplatPly(bytes);
//   console.log(splat.count, splat.center, splat.covariance, splat.rgba);

export type PlyFormat = "ascii" | "binary_little_endian" | "binary_big_endian";

export type PlyScalarType =
  | "char"
  | "uchar"
  | "short"
  | "ushort"
  | "int"
  | "uint"
  | "float"
  | "double";

export type PlyProperty =
  | { kind: "scalar"; name: string; type: PlyScalarType }
  | {
      kind: "list";
      name: string;
      countType: PlyScalarType;
      itemType: PlyScalarType;
    };

export type PlyElement = {
  name: string;
  count: number;
  properties: PlyProperty[];
};

export type PlyHeader = {
  format: PlyFormat;
  version: string;
  elements: PlyElement[];
  comments: string[];
};

export type SplatPlyBuffers = {
  count: number;
  center: Float32Array; // 3N
  /** 6N floats: [m11,m12,m13,m22,m23,m33] per splat (symmetric 3x3 covariance). */
  covariance: Float32Array;
  rgba: Uint8Array; // 4N  (0..255)
  format: PlyFormat;
};

export type ParseSplatPlyOptions = {
  /** Typical 3DGS PLY stores log-scales; set false if your file stores linear scales. */
  assumeLogScale?: boolean; // default true
  /** Typical 3DGS PLY stores opacity as logit; set false if your file stores linear alpha [0..1]. */
  assumeLogitOpacity?: boolean; // default true
  /** If no rgb in file, use this fallback color (0..255). */
  defaultRGBA?: [number, number, number, number]; // default [255,255,255,255]
  /** If element name isn't "vertex", allow override. */
  vertexElementName?: string; // default "vertex"
};

const TYPE_SIZE: Record<PlyScalarType, number> = {
  char: 1,
  uchar: 1,
  short: 2,
  ushort: 2,
  int: 4,
  uint: 4,
  float: 4,
  double: 8,
};

function isScalarType(t: string): t is PlyScalarType {
  return (
    t === "char" ||
    t === "uchar" ||
    t === "short" ||
    t === "ushort" ||
    t === "int" ||
    t === "uint" ||
    t === "float" ||
    t === "double"
  );
}

function sigmoid(x: number): number {
  // Stable-ish sigmoid for typical logit ranges.
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  } else {
    const z = Math.exp(x);
    return z / (1 + z);
  }
}

function normalizeQuat(
  x: number,
  y: number,
  z: number,
  w: number
): [number, number, number, number] {
  const len = Math.hypot(x, y, z, w) || 1;
  return [x / len, y / len, z / len, w / len];
}

function quatToMat3Cols(
  x: number,
  y: number,
  z: number,
  w: number
): {
  c0: [number, number, number];
  c1: [number, number, number];
  c2: [number, number, number];
} {
  // Rotation matrix from unit quaternion (x,y,z,w). Columns of R.
  const xx = x * x,
    yy = y * y,
    zz = z * z;
  const xy = x * y,
    xz = x * z,
    yz = y * z;
  const wx = w * x,
    wy = w * y,
    wz = w * z;

  const r00 = 1 - 2 * (yy + zz);
  const r01 = 2 * (xy - wz);
  const r02 = 2 * (xz + wy);

  const r10 = 2 * (xy + wz);
  const r11 = 1 - 2 * (xx + zz);
  const r12 = 2 * (yz - wx);

  const r20 = 2 * (xz - wy);
  const r21 = 2 * (yz + wx);
  const r22 = 1 - 2 * (xx + yy);

  return {
    c0: [r00, r10, r20],
    c1: [r01, r11, r21],
    c2: [r02, r12, r22],
  };
}

function covarianceFromQuatScale(
  qx: number,
  qy: number,
  qz: number,
  qw: number,
  sx: number,
  sy: number,
  sz: number
): {
  m11: number;
  m12: number;
  m13: number;
  m22: number;
  m23: number;
  m33: number;
} {
  const [x, y, z, w] = normalizeQuat(qx, qy, qz, qw);
  const { c0, c1, c2 } = quatToMat3Cols(x, y, z, w);

  const sx2 = sx * sx;
  const sy2 = sy * sy;
  const sz2 = sz * sz;

  // Sigma = sx^2*c0*c0^T + sy^2*c1*c1^T + sz^2*c2*c2^T
  const m11 = sx2 * c0[0] * c0[0] + sy2 * c1[0] * c1[0] + sz2 * c2[0] * c2[0];
  const m12 = sx2 * c0[0] * c0[1] + sy2 * c1[0] * c1[1] + sz2 * c2[0] * c2[1];
  const m13 = sx2 * c0[0] * c0[2] + sy2 * c1[0] * c1[2] + sz2 * c2[0] * c2[2];

  const m22 = sx2 * c0[1] * c0[1] + sy2 * c1[1] * c1[1] + sz2 * c2[1] * c2[1];
  const m23 = sx2 * c0[1] * c0[2] + sy2 * c1[1] * c1[2] + sz2 * c2[1] * c2[2];

  const m33 = sx2 * c0[2] * c0[2] + sy2 * c1[2] * c1[2] + sz2 * c2[2] * c2[2];

  return { m11, m12, m13, m22, m23, m33 };
}

function findHeaderEnd(bytes: Uint8Array): {
  headerEnd: number;
  newline: "\n" | "\r\n";
} {
  // search ASCII "end_header\n" or "end_header\r\n"
  const pat = [0x65, 0x6e, 0x64, 0x5f, 0x68, 0x65, 0x61, 0x64, 0x65, 0x72]; // "end_header"
  for (let i = 0; i <= bytes.length - pat.length; i++) {
    let ok = true;
    for (let j = 0; j < pat.length; j++) {
      if (bytes[i + j] !== pat[j]) {
        ok = false;
        break;
      }
    }
    if (!ok) continue;

    const k = i + pat.length;
    if (k < bytes.length && bytes[k] === 0x0a)
      return { headerEnd: k + 1, newline: "\n" };
    if (k + 1 < bytes.length && bytes[k] === 0x0d && bytes[k + 1] === 0x0a)
      return { headerEnd: k + 2, newline: "\r\n" };
  }
  throw new Error("PLY: can't find end_header");
}

export function parseHeader(bytes: Uint8Array): {
  header: PlyHeader;
  dataOffset: number;
  newline: "\n" | "\r\n";
} {
  const { headerEnd, newline } = findHeaderEnd(bytes);
  const headerText = new TextDecoder("utf-8").decode(
    bytes.subarray(0, headerEnd)
  );
  const lines = headerText
    .split(newline)
    .map((l) => l.trim())
    .filter(Boolean);

  if (lines[0] !== "ply")
    throw new Error(`PLY: first line must be "ply", got "${lines[0]}"`);

  let format: PlyFormat | null = null;
  let version = "";
  const comments: string[] = [];
  const elements: PlyElement[] = [];
  let current: PlyElement | null = null;

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i];
    if (line === "end_header") break;
    const parts = line.split(/\s+/);
    const tag = parts[0];

    if (tag === "comment") {
      comments.push(line.slice("comment".length).trim());
      continue;
    }
    if (tag === "format") {
      const fmt = parts[1] as PlyFormat;
      const ver = parts[2];
      if (
        fmt !== "ascii" &&
        fmt !== "binary_little_endian" &&
        fmt !== "binary_big_endian"
      ) {
        throw new Error(`PLY: unsupported format "${parts[1]}"`);
      }
      format = fmt;
      version = ver;
      continue;
    }
    if (tag === "element") {
      const name = parts[1];
      const count = Number(parts[2]);
      if (!Number.isFinite(count) || count < 0)
        throw new Error(`PLY: bad element count "${parts[2]}"`);
      current = { name, count, properties: [] };
      elements.push(current);
      continue;
    }
    if (tag === "property") {
      if (!current) throw new Error("PLY: property before element");
      if (parts[1] === "list") {
        const countType = parts[2];
        const itemType = parts[3];
        const name = parts[4];
        if (!isScalarType(countType) || !isScalarType(itemType))
          throw new Error(`PLY: bad list types ${countType} ${itemType}`);
        current.properties.push({ kind: "list", name, countType, itemType });
      } else {
        const type = parts[1];
        const name = parts[2];
        if (!isScalarType(type))
          throw new Error(`PLY: bad scalar type ${type}`);
        current.properties.push({ kind: "scalar", name, type });
      }
      continue;
    }
    if (tag === "obj_info") continue;

    throw new Error(`PLY: unknown header directive "${line}"`);
  }

  if (!format) throw new Error("PLY: missing format");
  return {
    header: { format, version, elements, comments },
    dataOffset: headerEnd,
    newline,
  };
}

function getBinaryReader(type: PlyScalarType) {
  switch (type) {
    case "char":
      return (dv: DataView, o: number, _: boolean) => dv.getInt8(o);
    case "uchar":
      return (dv: DataView, o: number, _: boolean) => dv.getUint8(o);
    case "short":
      return (dv: DataView, o: number, le: boolean) => dv.getInt16(o, le);
    case "ushort":
      return (dv: DataView, o: number, le: boolean) => dv.getUint16(o, le);
    case "int":
      return (dv: DataView, o: number, le: boolean) => dv.getInt32(o, le);
    case "uint":
      return (dv: DataView, o: number, le: boolean) => dv.getUint32(o, le);
    case "float":
      return (dv: DataView, o: number, le: boolean) => dv.getFloat32(o, le);
    case "double":
      return (dv: DataView, o: number, le: boolean) => dv.getFloat64(o, le);
  }
}

function clamp255(x: number): number {
  return x < 0 ? 0 : x > 255 ? 255 : x | 0;
}

function isProbablyByteColorType(t: PlyScalarType): boolean {
  return t === "uchar" || t === "char";
}

function lowerMapProperties(el: PlyElement) {
  // map lowercased prop name -> { prop, index }
  const m = new Map<string, { prop: PlyProperty; index: number }>();
  el.properties.forEach((p, i) =>
    m.set(p.name.toLowerCase(), { prop: p, index: i })
  );
  return m;
}

function pickName<T>(map: Map<string, T>, names: string[]): T | null {
  for (const n of names) {
    const v = map.get(n);
    if (v) return v;
  }
  return null;
}

export function parseSplatPly(
  bytes: Uint8Array,
  opts: ParseSplatPlyOptions = {}
): SplatPlyBuffers {
  const assumeLogScale = opts.assumeLogScale ?? true;
  const assumeLogitOpacity = opts.assumeLogitOpacity ?? true;
  const defaultRGBA = opts.defaultRGBA ?? [255, 255, 255, 255];
  const vertexName = (opts.vertexElementName ?? "vertex").toLowerCase();

  const { header, dataOffset, newline } = parseHeader(bytes);
  const el = header.elements.find((e) => e.name.toLowerCase() === vertexName);
  if (!el) throw new Error(`PLY: element "${vertexName}" not found`);

  // Minimal restriction: no list props on vertex.
  if (el.properties.some((p) => p.kind === "list")) {
    throw new Error(
      `PLY: vertex has list properties — not supported by this minimal splat parser`
    );
  }

  const pmap = lowerMapProperties(el);

  const px = pickName(pmap, ["x", "pos_x", "position_x"]);
  const py = pickName(pmap, ["y", "pos_y", "position_y"]);
  const pz = pickName(pmap, ["z", "pos_z", "position_z"]);
  if (!px || !py || !pz) throw new Error("PLY: missing x/y/z in vertex");

  const s0 = pickName(pmap, ["scale_0", "sx", "scale_x", "scalex"]);
  const s1 = pickName(pmap, ["scale_1", "sy", "scale_y", "scaley"]);
  const s2 = pickName(pmap, ["scale_2", "sz", "scale_z", "scalez"]);
  if (!s0 || !s1 || !s2)
    throw new Error("PLY: missing scale (scale_0..2 or sx/sy/sz) in vertex");

  const r0 = pickName(pmap, ["rot_0", "qx"]);
  const r1 = pickName(pmap, ["rot_1", "qy"]);
  const r2 = pickName(pmap, ["rot_2", "qz"]);
  const r3 = pickName(pmap, ["rot_3", "qw"]);
  if (!r0 || !r1 || !r2 || !r3)
    throw new Error(
      "PLY: missing rotation (rot_0..3 or qx/qy/qz/qw) in vertex"
    );

  const op = pickName(pmap, ["opacity", "alpha", "opac"]);
  if (!op) throw new Error('PLY: missing "opacity" (or alpha) in vertex');

  // Optional color
  const pr = pickName(pmap, ["red", "r"]);
  const pg = pickName(pmap, ["green", "g"]);
  const pb = pickName(pmap, ["blue", "b"]);

  const count = el.count;

  const center = new Float32Array(count * 3);
  const covariance = new Float32Array(count * 6);
  const rgba = new Uint8Array(count * 4);

  // init default colors
  for (let i = 0; i < count; i++) {
    const o = i * 4;
    rgba[o + 0] = defaultRGBA[0];
    rgba[o + 1] = defaultRGBA[1];
    rgba[o + 2] = defaultRGBA[2];
    rgba[o + 3] = defaultRGBA[3];
  }

  if (
    header.format === "binary_little_endian" ||
    header.format === "binary_big_endian"
  ) {
    const littleEndian = header.format === "binary_little_endian";
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

    // build offsets and stride from property order
    const props = el.properties as Extract<PlyProperty, { kind: "scalar" }>[];
    const offsets: number[] = [];
    let stride = 0;
    for (const p of props) {
      offsets.push(stride);
      stride += TYPE_SIZE[p.type];
    }

    const readScalar = (
      propIndex: number,
      type: PlyScalarType,
      base: number
    ) => {
      const reader = getBinaryReader(type);
      return reader(dv, base + offsets[propIndex], littleEndian);
    };

    const getIdx = (entry: { index: number }) => entry.index;
    const ix = getIdx(px),
      iy = getIdx(py),
      iz = getIdx(pz);
    const is0 = getIdx(s0),
      is1 = getIdx(s1),
      is2 = getIdx(s2);
    const ir0 = getIdx(r0),
      ir1 = getIdx(r1),
      ir2 = getIdx(r2),
      ir3 = getIdx(r3);
    const iop = getIdx(op);

    const tx = (px.prop as any).type as PlyScalarType;
    const ty = (py.prop as any).type as PlyScalarType;
    const tz = (pz.prop as any).type as PlyScalarType;

    const ts0 = (s0.prop as any).type as PlyScalarType;
    const ts1 = (s1.prop as any).type as PlyScalarType;
    const ts2 = (s2.prop as any).type as PlyScalarType;

    const tr0t = (r0.prop as any).type as PlyScalarType;
    const tr1t = (r1.prop as any).type as PlyScalarType;
    const tr2t = (r2.prop as any).type as PlyScalarType;
    const tr3t = (r3.prop as any).type as PlyScalarType;

    const topt = (op.prop as any).type as PlyScalarType;

    let hasColor = false;
    let ir = -1,
      ig = -1,
      ib = -1;
    let tr: PlyScalarType | null = null,
      tg: PlyScalarType | null = null,
      tb: PlyScalarType | null = null;

    if (pr && pg && pb) {
      hasColor = true;
      ir = pr.index;
      ig = pg.index;
      ib = pb.index;
      tr = (pr.prop as any).type as PlyScalarType;
      tg = (pg.prop as any).type as PlyScalarType;
      tb = (pb.prop as any).type as PlyScalarType;
    }

    let base = dataOffset;
    for (let i = 0; i < count; i++, base += stride) {
      const cx = Number(readScalar(ix, tx, base));
      const cy = Number(readScalar(iy, ty, base));
      const cz = Number(readScalar(iz, tz, base));

      let sx = Number(readScalar(is0, ts0, base));
      let sy = Number(readScalar(is1, ts1, base));
      let sz = Number(readScalar(is2, ts2, base));
      if (assumeLogScale) {
        sx = Math.exp(sx);
        sy = Math.exp(sy);
        sz = Math.exp(sz);
      }

      const qx = Number(readScalar(ir0, tr0t, base));
      const qy = Number(readScalar(ir1, tr1t, base));
      const qz = Number(readScalar(ir2, tr2t, base));
      const qw = Number(readScalar(ir3, tr3t, base));

      const opv = Number(readScalar(iop, topt, base));
      const alpha = assumeLogitOpacity ? sigmoid(opv) : opv;

      const { m11, m12, m13, m22, m23, m33 } = covarianceFromQuatScale(
        qx,
        qy,
        qz,
        qw,
        sx,
        sy,
        sz
      );

      const v3 = i * 3;
      center[v3 + 0] = cx;
      center[v3 + 1] = cy;
      center[v3 + 2] = cz;

      const v6 = i * 6;
      covariance[v6 + 0] = m11;
      covariance[v6 + 1] = m12;
      covariance[v6 + 2] = m13;
      covariance[v6 + 3] = m22;
      covariance[v6 + 4] = m23;
      covariance[v6 + 5] = m33;

      const v4 = i * 4;
      rgba[v4 + 3] = clamp255(alpha * 255);

      if (hasColor && tr && tg && tb) {
        const rv = Number(readScalar(ir, tr, base));
        const gv = Number(readScalar(ig, tg, base));
        const bv = Number(readScalar(ib, tb, base));

        if (
          isProbablyByteColorType(tr) &&
          isProbablyByteColorType(tg) &&
          isProbablyByteColorType(tb)
        ) {
          rgba[v4 + 0] = clamp255(rv);
          rgba[v4 + 1] = clamp255(gv);
          rgba[v4 + 2] = clamp255(bv);
        } else {
          // assume 0..1 floats
          rgba[v4 + 0] = clamp255(rv * 255);
          rgba[v4 + 1] = clamp255(gv * 255);
          rgba[v4 + 2] = clamp255(bv * 255);
        }
      }
    }

    return { count, center, covariance, rgba, format: header.format };
  }

  // ASCII fallback (slow; OK for debugging)
  if (header.format === "ascii") {
    const text = new TextDecoder("utf-8").decode(bytes.subarray(dataOffset));
    const lines = text.split(newline).filter(Boolean);

    const scalarProps = el.properties as Extract<
      PlyProperty,
      { kind: "scalar" }
    >[];
    const nameToCol = new Map<string, number>();
    for (let i = 0; i < scalarProps.length; i++)
      nameToCol.set(scalarProps[i].name.toLowerCase(), i);

    const col = (names: string[]) => {
      for (const n of names) {
        const v = nameToCol.get(n);
        if (v != null) return v;
      }
      return -1;
    };

    const cxC = col(["x", "pos_x", "position_x"]);
    const cyC = col(["y", "pos_y", "position_y"]);
    const czC = col(["z", "pos_z", "position_z"]);
    const s0C = col(["scale_0", "sx", "scale_x", "scalex"]);
    const s1C = col(["scale_1", "sy", "scale_y", "scaley"]);
    const s2C = col(["scale_2", "sz", "scale_z", "scalez"]);
    const r0C = col(["rot_0", "qx"]);
    const r1C = col(["rot_1", "qy"]);
    const r2C = col(["rot_2", "qz"]);
    const r3C = col(["rot_3", "qw"]);
    const opC = col(["opacity", "alpha", "opac"]);
    const rC = col(["red", "r"]);
    const gC = col(["green", "g"]);
    const bC = col(["blue", "b"]);

    if (
      [cxC, cyC, czC, s0C, s1C, s2C, r0C, r1C, r2C, r3C, opC].some((v) => v < 0)
    ) {
      throw new Error("PLY ASCII: missing required columns for splats");
    }

    for (let i = 0; i < count; i++) {
      const parts = lines[i].trim().split(/\s+/);

      const cx = Number(parts[cxC]);
      const cy = Number(parts[cyC]);
      const cz = Number(parts[czC]);

      let sx = Number(parts[s0C]);
      let sy = Number(parts[s1C]);
      let sz = Number(parts[s2C]);
      if (assumeLogScale) {
        sx = Math.exp(sx);
        sy = Math.exp(sy);
        sz = Math.exp(sz);
      }

      const qx = Number(parts[r0C]);
      const qy = Number(parts[r1C]);
      const qz = Number(parts[r2C]);
      const qw = Number(parts[r3C]);

      const opv = Number(parts[opC]);
      const alpha = assumeLogitOpacity ? sigmoid(opv) : opv;

      const { m11, m12, m13, m22, m23, m33 } = covarianceFromQuatScale(
        qx,
        qy,
        qz,
        qw,
        sx,
        sy,
        sz
      );

      const v3 = i * 3;
      center[v3 + 0] = cx;
      center[v3 + 1] = cy;
      center[v3 + 2] = cz;

      const v6 = i * 6;
      covariance[v6 + 0] = m11;
      covariance[v6 + 1] = m12;
      covariance[v6 + 2] = m13;
      covariance[v6 + 3] = m22;
      covariance[v6 + 4] = m23;
      covariance[v6 + 5] = m33;

      const v4 = i * 4;
      rgba[v4 + 3] = clamp255(alpha * 255);

      if (rC >= 0 && gC >= 0 && bC >= 0) {
        // ASCII ambiguous: assume 0..1 floats if <=1 else bytes
        const rv = Number(parts[rC]);
        const gv = Number(parts[gC]);
        const bv = Number(parts[bC]);

        const asFloat01 = rv <= 1 && gv <= 1 && bv <= 1;
        rgba[v4 + 0] = clamp255(asFloat01 ? rv * 255 : rv);
        rgba[v4 + 1] = clamp255(asFloat01 ? gv * 255 : gv);
        rgba[v4 + 2] = clamp255(asFloat01 ? bv * 255 : bv);
      }
    }

    return { count, center, covariance, rgba, format: header.format };
  }

  throw new Error(`PLY: unsupported format ${header.format}`);
}
