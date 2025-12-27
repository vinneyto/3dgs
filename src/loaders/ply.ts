// splat_ply_parser.ts
// Minimal PLY parser for 3D Gaussian Splats:
// - Reads PLY header (ascii/binary_little_endian/binary_big_endian)
// - Extracts: center (x,y,z), covariance (6 via scale+quat), rgba (optional rgb + opacity)
// - Uses f_dc_0..2 (DC spherical harmonics) as a fallback for RGB when explicit red/green/blue is absent.
// - Ignores higher-order spherical harmonics (f_rest_*) intentionally.
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

type PlyScalarProperty = Extract<PlyProperty, { kind: "scalar" }>;
type PlyPropertyEntry = { prop: PlyProperty; index: number };
type PlyScalarPropertyEntry = { prop: PlyScalarProperty; index: number };

function assertScalarEntry(
  entry: PlyPropertyEntry,
  nameForError: string
): asserts entry is PlyScalarPropertyEntry {
  if (entry.prop.kind !== "scalar") {
    throw new Error(`PLY: property "${nameForError}" must be scalar`);
  }
}

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
  /** N packed RGBA8 as uint32: r | (g<<8) | (b<<16) | (a<<24) */
  rgba: Uint32Array;
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
      return (dv: DataView, o: number) => dv.getInt8(o);
    case "uchar":
      return (dv: DataView, o: number) => dv.getUint8(o);
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

function rgbaToUint32(r: number, g: number, b: number, a: number): number {
  // Matches GaussianSplats3D packing (Util.rgbaToInteger / rgbaArrayToInteger)
  return (
    ((r & 255) | ((g & 255) << 8) | ((b & 255) << 16) | ((a & 255) << 24)) >>> 0
  );
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
  assertScalarEntry(px, "x");
  assertScalarEntry(py, "y");
  assertScalarEntry(pz, "z");

  const s0 = pickName(pmap, ["scale_0", "sx", "scale_x", "scalex"]);
  const s1 = pickName(pmap, ["scale_1", "sy", "scale_y", "scaley"]);
  const s2 = pickName(pmap, ["scale_2", "sz", "scale_z", "scalez"]);
  if (!s0 || !s1 || !s2)
    throw new Error("PLY: missing scale (scale_0..2 or sx/sy/sz) in vertex");
  assertScalarEntry(s0, "scale_0");
  assertScalarEntry(s1, "scale_1");
  assertScalarEntry(s2, "scale_2");

  const r0 = pickName(pmap, ["rot_0", "qx"]);
  const r1 = pickName(pmap, ["rot_1", "qy"]);
  const r2 = pickName(pmap, ["rot_2", "qz"]);
  const r3 = pickName(pmap, ["rot_3", "qw"]);
  if (!r0 || !r1 || !r2 || !r3)
    throw new Error(
      "PLY: missing rotation (rot_0..3 or qx/qy/qz/qw) in vertex"
    );
  assertScalarEntry(r0, "rot_0");
  assertScalarEntry(r1, "rot_1");
  assertScalarEntry(r2, "rot_2");
  assertScalarEntry(r3, "rot_3");

  const op = pickName(pmap, ["opacity", "alpha", "opac"]);
  if (!op) throw new Error('PLY: missing "opacity" (or alpha) in vertex');
  assertScalarEntry(op, "opacity");

  // Optional color
  const pr = pickName(pmap, ["red", "r"]);
  const pg = pickName(pmap, ["green", "g"]);
  const pb = pickName(pmap, ["blue", "b"]);
  const fdc0 = pickName(pmap, ["f_dc_0"]);
  const fdc1 = pickName(pmap, ["f_dc_1"]);
  const fdc2 = pickName(pmap, ["f_dc_2"]);
  const SH_C0 = 0.28209479177387814; // matches GaussianSplats3D INRIA parsers

  const count = el.count;

  const center = new Float32Array(count * 3);
  const covariance = new Float32Array(count * 6);
  const rgba = new Uint32Array(count);
  const defaultPacked = rgbaToUint32(
    clamp255(defaultRGBA[0]),
    clamp255(defaultRGBA[1]),
    clamp255(defaultRGBA[2]),
    clamp255(defaultRGBA[3])
  );
  rgba.fill(defaultPacked);

  if (
    header.format === "binary_little_endian" ||
    header.format === "binary_big_endian"
  ) {
    const littleEndian = header.format === "binary_little_endian";
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

    // build offsets and stride from property order (safe because we rejected list props above)
    const props = el.properties as PlyScalarProperty[];
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

    const tx = px.prop.type;
    const ty = py.prop.type;
    const tz = pz.prop.type;

    const ts0 = s0.prop.type;
    const ts1 = s1.prop.type;
    const ts2 = s2.prop.type;

    const tr0t = r0.prop.type;
    const tr1t = r1.prop.type;
    const tr2t = r2.prop.type;
    const tr3t = r3.prop.type;

    const topt = op.prop.type;

    let hasColor = false;
    let ir = -1,
      ig = -1,
      ib = -1;
    let tr: PlyScalarType | null = null,
      tg: PlyScalarType | null = null,
      tb: PlyScalarType | null = null;

    if (pr && pg && pb) {
      hasColor = true;
      assertScalarEntry(pr, "red");
      assertScalarEntry(pg, "green");
      assertScalarEntry(pb, "blue");
      ir = pr.index;
      ig = pg.index;
      ib = pb.index;
      tr = pr.prop.type;
      tg = pg.prop.type;
      tb = pb.prop.type;
    }

    let hasFDC = false;
    let if0 = -1,
      if1 = -1,
      if2 = -1;
    let tf0: PlyScalarType | null = null,
      tf1: PlyScalarType | null = null,
      tf2: PlyScalarType | null = null;

    if (!hasColor && fdc0 && fdc1 && fdc2) {
      hasFDC = true;
      assertScalarEntry(fdc0, "f_dc_0");
      assertScalarEntry(fdc1, "f_dc_1");
      assertScalarEntry(fdc2, "f_dc_2");
      if0 = fdc0.index;
      if1 = fdc1.index;
      if2 = fdc2.index;
      tf0 = fdc0.prop.type;
      tf1 = fdc1.prop.type;
      tf2 = fdc2.prop.type;
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

      const a = clamp255(alpha * 255);
      let r = clamp255(defaultRGBA[0]);
      let g = clamp255(defaultRGBA[1]);
      let b = clamp255(defaultRGBA[2]);

      if (hasColor && tr && tg && tb) {
        const rv = Number(readScalar(ir, tr, base));
        const gv = Number(readScalar(ig, tg, base));
        const bv = Number(readScalar(ib, tb, base));

        if (
          isProbablyByteColorType(tr) &&
          isProbablyByteColorType(tg) &&
          isProbablyByteColorType(tb)
        ) {
          r = clamp255(rv);
          g = clamp255(gv);
          b = clamp255(bv);
        } else {
          // assume 0..1 floats
          r = clamp255(rv * 255);
          g = clamp255(gv * 255);
          b = clamp255(bv * 255);
        }
      } else if (hasFDC && tf0 && tf1 && tf2) {
        // GaussianSplats3D-style DC SH -> RGB conversion:
        // rgb = (0.5 + SH_C0 * f_dc) * 255
        const f0 = Number(readScalar(if0, tf0, base));
        const f1 = Number(readScalar(if1, tf1, base));
        const f2 = Number(readScalar(if2, tf2, base));
        r = clamp255((0.5 + SH_C0 * f0) * 255);
        g = clamp255((0.5 + SH_C0 * f1) * 255);
        b = clamp255((0.5 + SH_C0 * f2) * 255);
      }

      rgba[i] = rgbaToUint32(r, g, b, a);
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
    const f0C = col(["f_dc_0"]);
    const f1C = col(["f_dc_1"]);
    const f2C = col(["f_dc_2"]);
    const SH_C0 = 0.28209479177387814; // matches GaussianSplats3D INRIA parsers

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

      const a = clamp255(alpha * 255);
      let r = clamp255(defaultRGBA[0]);
      let g = clamp255(defaultRGBA[1]);
      let b = clamp255(defaultRGBA[2]);

      if (rC >= 0 && gC >= 0 && bC >= 0) {
        // ASCII ambiguous: assume 0..1 floats if <=1 else bytes
        const rv = Number(parts[rC]);
        const gv = Number(parts[gC]);
        const bv = Number(parts[bC]);

        const asFloat01 = rv <= 1 && gv <= 1 && bv <= 1;
        r = clamp255(asFloat01 ? rv * 255 : rv);
        g = clamp255(asFloat01 ? gv * 255 : gv);
        b = clamp255(asFloat01 ? bv * 255 : bv);
      } else if (f0C >= 0 && f1C >= 0 && f2C >= 0) {
        const f0 = Number(parts[f0C]);
        const f1 = Number(parts[f1C]);
        const f2 = Number(parts[f2C]);
        r = clamp255((0.5 + SH_C0 * f0) * 255);
        g = clamp255((0.5 + SH_C0 * f1) * 255);
        b = clamp255((0.5 + SH_C0 * f2) * 255);
      }

      rgba[i] = rgbaToUint32(r, g, b, a);
    }

    return { count, center, covariance, rgba, format: header.format };
  }

  throw new Error(`PLY: unsupported format ${header.format}`);
}
