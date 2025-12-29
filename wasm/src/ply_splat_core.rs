use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct SplatPlyBuffersCore {
    pub count: u32,
    pub format: PlyFormat,
    pub center: Box<[f32]>,     // 3N
    pub covariance: Box<[f32]>, // 6N
    pub rgba: Box<[u32]>,       // N
    pub bbox_min: [f32; 3],
    pub bbox_max: [f32; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlyFormat {
    Ascii,
    BinaryLittleEndian,
    BinaryBigEndian,
}

impl PlyFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            PlyFormat::Ascii => "ascii",
            PlyFormat::BinaryLittleEndian => "binary_little_endian",
            PlyFormat::BinaryBigEndian => "binary_big_endian",
        }
    }
}

#[derive(Debug, Clone)]
pub enum PlyError {
    Msg(&'static str),
    MsgOwned(String),
}

impl PlyError {
    fn msg(s: &'static str) -> Self {
        PlyError::Msg(s)
    }
}

impl std::fmt::Display for PlyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlyError::Msg(s) => write!(f, "{s}"),
            PlyError::MsgOwned(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for PlyError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PlyScalarType {
    Char,
    UChar,
    Short,
    UShort,
    Int,
    UInt,
    Float,
    Double,
}

impl PlyScalarType {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "char" => Some(Self::Char),
            "uchar" => Some(Self::UChar),
            "short" => Some(Self::Short),
            "ushort" => Some(Self::UShort),
            "int" => Some(Self::Int),
            "uint" => Some(Self::UInt),
            "float" => Some(Self::Float),
            "double" => Some(Self::Double),
            _ => None,
        }
    }

    fn size_bytes(&self) -> usize {
        match self {
            PlyScalarType::Char => 1,
            PlyScalarType::UChar => 1,
            PlyScalarType::Short => 2,
            PlyScalarType::UShort => 2,
            PlyScalarType::Int => 4,
            PlyScalarType::UInt => 4,
            PlyScalarType::Float => 4,
            PlyScalarType::Double => 8,
        }
    }

    fn is_probably_byte_color(&self) -> bool {
        matches!(self, PlyScalarType::Char | PlyScalarType::UChar)
    }
}

#[derive(Clone, Debug)]
enum PlyProperty {
    Scalar { name: String, ty: PlyScalarType },
    List {
        #[allow(dead_code)]
        name: String,
        #[allow(dead_code)]
        count_ty: PlyScalarType,
        #[allow(dead_code)]
        item_ty: PlyScalarType,
    },
}

#[derive(Clone, Debug)]
struct PlyElement {
    name: String,
    count: usize,
    properties: Vec<PlyProperty>,
}

#[derive(Clone, Debug)]
struct ParsedHeader {
    format: PlyFormat,
    elements: Vec<PlyElement>,
    data_offset: usize,
    newline: Newline,
}

#[derive(Clone, Copy, Debug)]
enum Newline {
    Lf,
    CrLf,
}

fn find_header_end(bytes: &[u8]) -> Result<(usize, Newline), PlyError> {
    const PAT: &[u8] = b"end_header";
    if bytes.len() < PAT.len() {
        return Err(PlyError::msg("PLY: can't find end_header"));
    }
    for i in 0..=(bytes.len() - PAT.len()) {
        if &bytes[i..i + PAT.len()] != PAT {
            continue;
        }
        let k = i + PAT.len();
        if k < bytes.len() && bytes[k] == b'\n' {
            return Ok((k + 1, Newline::Lf));
        }
        if k + 1 < bytes.len() && bytes[k] == b'\r' && bytes[k + 1] == b'\n' {
            return Ok((k + 2, Newline::CrLf));
        }
    }
    Err(PlyError::msg("PLY: can't find end_header"))
}

fn parse_header(bytes: &[u8]) -> Result<ParsedHeader, PlyError> {
    let (header_end, newline) = find_header_end(bytes)?;
    let header_text = core::str::from_utf8(&bytes[..header_end])
        .map_err(|_| PlyError::msg("PLY: header is not valid utf-8"))?;

    let mut lines: Vec<&str> = match newline {
        Newline::Lf => header_text.split('\n').collect(),
        Newline::CrLf => header_text.split("\r\n").collect(),
    };
    lines = lines
        .into_iter()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();

    if lines.is_empty() || lines[0] != "ply" {
        return Err(PlyError::msg("PLY: first line must be \"ply\""));
    }

    let mut format: Option<PlyFormat> = None;
    let mut elements: Vec<PlyElement> = Vec::new();
    let mut current: Option<PlyElement> = None;

    for line in lines.iter().skip(1) {
        if *line == "end_header" {
            break;
        }
        let mut it = line.split_whitespace();
        let tag = it.next().unwrap_or("");

        match tag {
            "comment" | "obj_info" => {}
            "format" => {
                let fmt = it.next().unwrap_or("");
                let _ver = it.next().unwrap_or("");
                let f = match fmt {
                    "ascii" => PlyFormat::Ascii,
                    "binary_little_endian" => PlyFormat::BinaryLittleEndian,
                    "binary_big_endian" => PlyFormat::BinaryBigEndian,
                    _ => return Err(PlyError::msg("PLY: unsupported format")),
                };
                format = Some(f);
            }
            "element" => {
                if let Some(el) = current.take() {
                    elements.push(el);
                }
                let name = it.next().ok_or_else(|| PlyError::msg("PLY: bad element"))?;
                let count_str = it
                    .next()
                    .ok_or_else(|| PlyError::msg("PLY: bad element count"))?;
                let count: usize = count_str
                    .parse()
                    .map_err(|_| PlyError::msg("PLY: bad element count"))?;
                current = Some(PlyElement {
                    name: name.to_string(),
                    count,
                    properties: Vec::new(),
                });
            }
            "property" => {
                let cur = current
                    .as_mut()
                    .ok_or_else(|| PlyError::msg("PLY: property before element"))?;
                let t1 = it.next().ok_or_else(|| PlyError::msg("PLY: bad property"))?;
                if t1 == "list" {
                    let count_t = it
                        .next()
                        .ok_or_else(|| PlyError::msg("PLY: bad list property"))?;
                    let item_t = it
                        .next()
                        .ok_or_else(|| PlyError::msg("PLY: bad list property"))?;
                    let name = it
                        .next()
                        .ok_or_else(|| PlyError::msg("PLY: bad list property"))?;
                    let count_ty =
                        PlyScalarType::parse(count_t).ok_or_else(|| PlyError::msg("PLY: bad list type"))?;
                    let item_ty =
                        PlyScalarType::parse(item_t).ok_or_else(|| PlyError::msg("PLY: bad list type"))?;
                    cur.properties.push(PlyProperty::List {
                        name: name.to_string(),
                        count_ty,
                        item_ty,
                    });
                } else {
                    let ty = PlyScalarType::parse(t1)
                        .ok_or_else(|| PlyError::msg("PLY: bad scalar type"))?;
                    let name = it
                        .next()
                        .ok_or_else(|| PlyError::msg("PLY: bad scalar property"))?;
                    cur.properties.push(PlyProperty::Scalar {
                        name: name.to_string(),
                        ty,
                    });
                }
            }
            _ => return Err(PlyError::msg("PLY: unknown header directive")),
        }
    }

    if let Some(el) = current.take() {
        elements.push(el);
    }
    let format = format.ok_or_else(|| PlyError::msg("PLY: missing format"))?;

    Ok(ParsedHeader {
        format,
        elements,
        data_offset: header_end,
        newline,
    })
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

fn clamp255(x: f32) -> u32 {
    if x <= 0.0 {
        0
    } else if x >= 255.0 {
        255
    } else {
        x.floor() as u32
    }
}

fn rgba_to_u32(r: u32, g: u32, b: u32, a: u32) -> u32 {
    ((r & 255) | ((g & 255) << 8) | ((b & 255) << 16) | ((a & 255) << 24)) as u32
}

fn normalize_quat(x: f32, y: f32, z: f32, w: f32) -> (f32, f32, f32, f32) {
    let len = (x * x + y * y + z * z + w * w).sqrt();
    let inv = if len > 0.0 { 1.0 / len } else { 1.0 };
    (x * inv, y * inv, z * inv, w * inv)
}

fn quat_to_mat3_cols(x: f32, y: f32, z: f32, w: f32) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;
    let wx = w * x;
    let wy = w * y;
    let wz = w * z;

    let r00 = 1.0 - 2.0 * (yy + zz);
    let r01 = 2.0 * (xy - wz);
    let r02 = 2.0 * (xz + wy);

    let r10 = 2.0 * (xy + wz);
    let r11 = 1.0 - 2.0 * (xx + zz);
    let r12 = 2.0 * (yz - wx);

    let r20 = 2.0 * (xz - wy);
    let r21 = 2.0 * (yz + wx);
    let r22 = 1.0 - 2.0 * (xx + yy);

    ([r00, r10, r20], [r01, r11, r21], [r02, r12, r22])
}

fn covariance_from_quat_scale(qx: f32, qy: f32, qz: f32, qw: f32, sx: f32, sy: f32, sz: f32) -> [f32; 6] {
    let (x, y, z, w) = normalize_quat(qx, qy, qz, qw);
    let (c0, c1, c2) = quat_to_mat3_cols(x, y, z, w);

    let sx2 = sx * sx;
    let sy2 = sy * sy;
    let sz2 = sz * sz;

    let m11 = sx2 * c0[0] * c0[0] + sy2 * c1[0] * c1[0] + sz2 * c2[0] * c2[0];
    let m12 = sx2 * c0[0] * c0[1] + sy2 * c1[0] * c1[1] + sz2 * c2[0] * c2[1];
    let m13 = sx2 * c0[0] * c0[2] + sy2 * c1[0] * c1[2] + sz2 * c2[0] * c2[2];

    let m22 = sx2 * c0[1] * c0[1] + sy2 * c1[1] * c1[1] + sz2 * c2[1] * c2[1];
    let m23 = sx2 * c0[1] * c0[2] + sy2 * c1[1] * c1[2] + sz2 * c2[1] * c2[2];

    let m33 = sx2 * c0[2] * c0[2] + sy2 * c1[2] * c1[2] + sz2 * c2[2] * c2[2];

    [m11, m12, m13, m22, m23, m33]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QuatLayout {
    /// Quaternion stored as (w, x, y, z). Common for PLY fields `rot_0..rot_3`.
    Wxyz,
    /// Quaternion stored as (x, y, z, w). Some PLY variants use `qx,qy,qz,qw`.
    Xyzw,
}

fn read_scalar(bytes: &[u8], offset: usize, ty: PlyScalarType, little: bool) -> Result<f64, PlyError> {
    let need = ty.size_bytes();
    if offset + need > bytes.len() {
        return Err(PlyError::msg("PLY: out of bounds while reading binary data"));
    }

    let b = &bytes[offset..offset + need];
    let v = match ty {
        PlyScalarType::Char => i8::from_ne_bytes([b[0]]) as f64,
        PlyScalarType::UChar => u8::from_ne_bytes([b[0]]) as f64,
        PlyScalarType::Short => {
            let arr = [b[0], b[1]];
            let n = if little { i16::from_le_bytes(arr) } else { i16::from_be_bytes(arr) };
            n as f64
        }
        PlyScalarType::UShort => {
            let arr = [b[0], b[1]];
            let n = if little { u16::from_le_bytes(arr) } else { u16::from_be_bytes(arr) };
            n as f64
        }
        PlyScalarType::Int => {
            let arr = [b[0], b[1], b[2], b[3]];
            let n = if little { i32::from_le_bytes(arr) } else { i32::from_be_bytes(arr) };
            n as f64
        }
        PlyScalarType::UInt => {
            let arr = [b[0], b[1], b[2], b[3]];
            let n = if little { u32::from_le_bytes(arr) } else { u32::from_be_bytes(arr) };
            n as f64
        }
        PlyScalarType::Float => {
            let arr = [b[0], b[1], b[2], b[3]];
            let n = if little { f32::from_le_bytes(arr) } else { f32::from_be_bytes(arr) };
            n as f64
        }
        PlyScalarType::Double => {
            let arr = [b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]];
            let n = if little { f64::from_le_bytes(arr) } else { f64::from_be_bytes(arr) };
            n
        }
    };
    Ok(v)
}

fn pick_name(map: &HashMap<String, (usize, PlyScalarType)>, names: &[&str]) -> Option<(usize, PlyScalarType)> {
    for n in names {
        if let Some(v) = map.get(&n.to_lowercase()) {
            return Some(*v);
        }
    }
    None
}

pub fn parse_splat_ply_core(bytes: &[u8]) -> Result<SplatPlyBuffersCore, PlyError> {
    parse_splat_ply_core_with_opts(bytes, true, true)
}

pub fn parse_splat_ply_core_with_opts(
    bytes: &[u8],
    assume_log_scale: bool,
    assume_logit_opacity: bool,
) -> Result<SplatPlyBuffersCore, PlyError> {
    let header = parse_header(bytes)?;
    let vertex_name = "vertex";
    let el = header
        .elements
        .iter()
        .find(|e| e.name.to_lowercase() == vertex_name)
        .ok_or_else(|| PlyError::msg("PLY: element \"vertex\" not found"))?;

    if el.properties.iter().any(|p| matches!(p, PlyProperty::List { .. })) {
        return Err(PlyError::msg(
            "PLY: vertex has list properties — not supported by this minimal splat parser",
        ));
    }

    let mut pmap: HashMap<String, (usize, PlyScalarType)> = HashMap::new();
    for (i, p) in el.properties.iter().enumerate() {
        if let PlyProperty::Scalar { name, ty } = p {
            pmap.insert(name.to_lowercase(), (i, *ty));
        }
    }

    let (ix, tx) = pick_name(&pmap, &["x", "pos_x", "position_x"])
        .ok_or_else(|| PlyError::msg("PLY: missing x in vertex"))?;
    let (iy, ty_) = pick_name(&pmap, &["y", "pos_y", "position_y"])
        .ok_or_else(|| PlyError::msg("PLY: missing y in vertex"))?;
    let (iz, tz) = pick_name(&pmap, &["z", "pos_z", "position_z"])
        .ok_or_else(|| PlyError::msg("PLY: missing z in vertex"))?;

    let (is0, ts0) = pick_name(&pmap, &["scale_0", "sx", "scale_x", "scalex"])
        .ok_or_else(|| PlyError::msg("PLY: missing scale_0 in vertex"))?;
    let (is1, ts1) = pick_name(&pmap, &["scale_1", "sy", "scale_y", "scaley"])
        .ok_or_else(|| PlyError::msg("PLY: missing scale_1 in vertex"))?;
    let (is2, ts2) = pick_name(&pmap, &["scale_2", "sz", "scale_z", "scalez"])
        .ok_or_else(|| PlyError::msg("PLY: missing scale_2 in vertex"))?;

    // Quaternion layout:
    // - If PLY contains rot_0..rot_3, interpret as (w, x, y, z).
    // - Otherwise, if it contains qx,qy,qz,qw, interpret as (x, y, z, w).
    let rot0 = pick_name(&pmap, &["rot_0"]);
    let rot1 = pick_name(&pmap, &["rot_1"]);
    let rot2 = pick_name(&pmap, &["rot_2"]);
    let rot3 = pick_name(&pmap, &["rot_3"]);
    let qx_f = pick_name(&pmap, &["qx"]);
    let qy_f = pick_name(&pmap, &["qy"]);
    let qz_f = pick_name(&pmap, &["qz"]);
    let qw_f = pick_name(&pmap, &["qw"]);

    let (quat_layout, (ir0, tr0), (ir1, tr1), (ir2, tr2), (ir3, tr3)) = if let (Some(a), Some(b), Some(c), Some(d)) =
        (rot0, rot1, rot2, rot3)
    {
        (QuatLayout::Wxyz, a, b, c, d)
    } else if let (Some(a), Some(b), Some(c), Some(d)) = (qx_f, qy_f, qz_f, qw_f) {
        (QuatLayout::Xyzw, a, b, c, d)
    } else {
        return Err(PlyError::msg(
            "PLY: missing quaternion fields. Expected either rot_0..rot_3 (wxyz) or qx,qy,qz,qw (xyzw)",
        ));
    };

    let (iop, top) = pick_name(&pmap, &["opacity", "alpha", "opac"])
        .ok_or_else(|| PlyError::msg("PLY: missing opacity in vertex"))?;

    let color_r = pick_name(&pmap, &["red", "r"]);
    let color_g = pick_name(&pmap, &["green", "g"]);
    let color_b = pick_name(&pmap, &["blue", "b"]);

    let fdc0 = pick_name(&pmap, &["f_dc_0"]);
    let fdc1 = pick_name(&pmap, &["f_dc_1"]);
    let fdc2 = pick_name(&pmap, &["f_dc_2"]);
    const SH_C0: f32 = 0.28209479177387814;

    let count = el.count;
    let mut center: Vec<f32> = vec![0.0; count * 3];
    let mut covariance: Vec<f32> = vec![0.0; count * 6];
    let mut rgba: Vec<u32> = vec![rgba_to_u32(255, 255, 255, 255); count];

    let mut bbox_min = [f32::INFINITY, f32::INFINITY, f32::INFINITY];
    let mut bbox_max = [f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];

    match header.format {
        PlyFormat::BinaryLittleEndian | PlyFormat::BinaryBigEndian => {
            let little = header.format == PlyFormat::BinaryLittleEndian;

            let mut offsets: Vec<usize> = Vec::with_capacity(el.properties.len());
            let mut stride: usize = 0;
            for p in el.properties.iter() {
                offsets.push(stride);
                let ty = match p {
                    PlyProperty::Scalar { ty, .. } => *ty,
                    PlyProperty::List { .. } => unreachable!(),
                };
                stride += ty.size_bytes();
            }

            let mut base = header.data_offset;
            for i in 0..count {
                let read = |prop_index: usize, t: PlyScalarType| -> Result<f64, PlyError> {
                    read_scalar(bytes, base + offsets[prop_index], t, little)
                };

                let cx = read(ix, tx)? as f32;
                let cy = read(iy, ty_)? as f32;
                let cz = read(iz, tz)? as f32;

                bbox_min[0] = bbox_min[0].min(cx);
                bbox_min[1] = bbox_min[1].min(cy);
                bbox_min[2] = bbox_min[2].min(cz);
                bbox_max[0] = bbox_max[0].max(cx);
                bbox_max[1] = bbox_max[1].max(cy);
                bbox_max[2] = bbox_max[2].max(cz);

                let mut sx = read(is0, ts0)? as f32;
                let mut sy = read(is1, ts1)? as f32;
                let mut sz = read(is2, ts2)? as f32;
                if assume_log_scale {
                    sx = sx.exp();
                    sy = sy.exp();
                    sz = sz.exp();
                }

                let a0 = read(ir0, tr0)? as f32;
                let a1 = read(ir1, tr1)? as f32;
                let a2 = read(ir2, tr2)? as f32;
                let a3 = read(ir3, tr3)? as f32;
                let (qx, qy, qz, qw) = match quat_layout {
                    QuatLayout::Wxyz => (a1, a2, a3, a0),
                    QuatLayout::Xyzw => (a0, a1, a2, a3),
                };

                let opv = read(iop, top)? as f32;
                let alpha = if assume_logit_opacity { sigmoid(opv) } else { opv };

                let cov = covariance_from_quat_scale(qx, qy, qz, qw, sx, sy, sz);

                let v3 = i * 3;
                center[v3] = cx;
                center[v3 + 1] = cy;
                center[v3 + 2] = cz;

                let v6 = i * 6;
                covariance[v6..v6 + 6].copy_from_slice(&cov);

                let a = clamp255(alpha * 255.0);
                let mut r = 255u32;
                let mut g = 255u32;
                let mut b = 255u32;

                if let (Some((ir, tr)), Some((ig, tg)), Some((ib, tb))) =
                    (color_r, color_g, color_b)
                {
                    let rv = read(ir, tr)? as f32;
                    let gv = read(ig, tg)? as f32;
                    let bv = read(ib, tb)? as f32;

                    if tr.is_probably_byte_color()
                        && tg.is_probably_byte_color()
                        && tb.is_probably_byte_color()
                    {
                        r = clamp255(rv);
                        g = clamp255(gv);
                        b = clamp255(bv);
                    } else {
                        r = clamp255(rv * 255.0);
                        g = clamp255(gv * 255.0);
                        b = clamp255(bv * 255.0);
                    }
                } else if let (Some((if0, tf0)), Some((if1, tf1)), Some((if2, tf2))) =
                    (fdc0, fdc1, fdc2)
                {
                    let f0 = read(if0, tf0)? as f32;
                    let f1 = read(if1, tf1)? as f32;
                    let f2 = read(if2, tf2)? as f32;
                    r = clamp255((0.5 + SH_C0 * f0) * 255.0);
                    g = clamp255((0.5 + SH_C0 * f1) * 255.0);
                    b = clamp255((0.5 + SH_C0 * f2) * 255.0);
                }

                rgba[i] = rgba_to_u32(r, g, b, a);
                base += stride;
            }
        }
        PlyFormat::Ascii => {
            let data = &bytes[header.data_offset..];
            let text = core::str::from_utf8(data)
                .map_err(|_| PlyError::msg("PLY ASCII: data is not valid utf-8"))?;
            let lines: Vec<&str> = match header.newline {
                Newline::Lf => text
                    .split('\n')
                    .filter(|l| !l.trim().is_empty())
                    .collect(),
                Newline::CrLf => text
                    .split("\r\n")
                    .filter(|l| !l.trim().is_empty())
                    .collect(),
            };
            if lines.len() < count {
                return Err(PlyError::msg("PLY ASCII: not enough vertex lines"));
            }

            let mut name_to_col: HashMap<String, usize> = HashMap::new();
            let mut scalar_i = 0usize;
            for p in el.properties.iter() {
                if let PlyProperty::Scalar { name, .. } = p {
                    name_to_col.insert(name.to_lowercase(), scalar_i);
                    scalar_i += 1;
                }
            }

            let col = |names: &[&str]| -> Option<usize> {
                for n in names {
                    if let Some(v) = name_to_col.get(&n.to_lowercase()) {
                        return Some(*v);
                    }
                }
                None
            };

            let cx_c = col(&["x", "pos_x", "position_x"]).ok_or_else(|| PlyError::msg("PLY ASCII: missing x"))?;
            let cy_c = col(&["y", "pos_y", "position_y"]).ok_or_else(|| PlyError::msg("PLY ASCII: missing y"))?;
            let cz_c = col(&["z", "pos_z", "position_z"]).ok_or_else(|| PlyError::msg("PLY ASCII: missing z"))?;
            let s0_c = col(&["scale_0", "sx", "scale_x", "scalex"]).ok_or_else(|| PlyError::msg("PLY ASCII: missing scale_0"))?;
            let s1_c = col(&["scale_1", "sy", "scale_y", "scaley"]).ok_or_else(|| PlyError::msg("PLY ASCII: missing scale_1"))?;
            let s2_c = col(&["scale_2", "sz", "scale_z", "scalez"]).ok_or_else(|| PlyError::msg("PLY ASCII: missing scale_2"))?;
            let r0 = col(&["rot_0"]);
            let r1 = col(&["rot_1"]);
            let r2 = col(&["rot_2"]);
            let r3 = col(&["rot_3"]);
            let qx = col(&["qx"]);
            let qy = col(&["qy"]);
            let qz = col(&["qz"]);
            let qw = col(&["qw"]);

            let (quat_layout, r0_c, r1_c, r2_c, r3_c) = if let (Some(a), Some(b), Some(c), Some(d)) =
                (r0, r1, r2, r3)
            {
                (QuatLayout::Wxyz, a, b, c, d)
            } else if let (Some(a), Some(b), Some(c), Some(d)) = (qx, qy, qz, qw) {
                (QuatLayout::Xyzw, a, b, c, d)
            } else {
                return Err(PlyError::msg(
                    "PLY ASCII: missing quaternion fields. Expected either rot_0..rot_3 (wxyz) or qx,qy,qz,qw (xyzw)",
                ));
            };
            let op_c = col(&["opacity", "alpha", "opac"]).ok_or_else(|| PlyError::msg("PLY ASCII: missing opacity"))?;

            let r_c = col(&["red", "r"]);
            let g_c = col(&["green", "g"]);
            let b_c = col(&["blue", "b"]);
            let f0_c = col(&["f_dc_0"]);
            let f1_c = col(&["f_dc_1"]);
            let f2_c = col(&["f_dc_2"]);

            for i in 0..count {
                let parts: Vec<&str> = lines[i].split_whitespace().collect();
                let parse = |idx: usize| -> Result<f32, PlyError> {
                    parts
                        .get(idx)
                        .ok_or_else(|| PlyError::msg("PLY ASCII: missing column"))?
                        .parse::<f32>()
                        .map_err(|_| PlyError::msg("PLY ASCII: failed to parse number"))
                };

                let cx = parse(cx_c)?;
                let cy = parse(cy_c)?;
                let cz = parse(cz_c)?;

                bbox_min[0] = bbox_min[0].min(cx);
                bbox_min[1] = bbox_min[1].min(cy);
                bbox_min[2] = bbox_min[2].min(cz);
                bbox_max[0] = bbox_max[0].max(cx);
                bbox_max[1] = bbox_max[1].max(cy);
                bbox_max[2] = bbox_max[2].max(cz);

                let mut sx = parse(s0_c)?;
                let mut sy = parse(s1_c)?;
                let mut sz = parse(s2_c)?;
                if assume_log_scale {
                    sx = sx.exp();
                    sy = sy.exp();
                    sz = sz.exp();
                }

                let a0 = parse(r0_c)?;
                let a1 = parse(r1_c)?;
                let a2 = parse(r2_c)?;
                let a3 = parse(r3_c)?;
                let (qx, qy, qz, qw) = match quat_layout {
                    QuatLayout::Wxyz => (a1, a2, a3, a0),
                    QuatLayout::Xyzw => (a0, a1, a2, a3),
                };

                let opv = parse(op_c)?;
                let alpha = if assume_logit_opacity { sigmoid(opv) } else { opv };

                let cov = covariance_from_quat_scale(qx, qy, qz, qw, sx, sy, sz);

                let v3 = i * 3;
                center[v3] = cx;
                center[v3 + 1] = cy;
                center[v3 + 2] = cz;

                let v6 = i * 6;
                covariance[v6..v6 + 6].copy_from_slice(&cov);

                let a = clamp255(alpha * 255.0);
                let mut r = 255u32;
                let mut g = 255u32;
                let mut b = 255u32;

                if let (Some(rc), Some(gc), Some(bc)) = (r_c, g_c, b_c) {
                    let rv = parse(rc)?;
                    let gv = parse(gc)?;
                    let bv = parse(bc)?;
                    let as_float01 = rv <= 1.0 && gv <= 1.0 && bv <= 1.0;
                    r = clamp255(if as_float01 { rv * 255.0 } else { rv });
                    g = clamp255(if as_float01 { gv * 255.0 } else { gv });
                    b = clamp255(if as_float01 { bv * 255.0 } else { bv });
                } else if let (Some(f0c), Some(f1c), Some(f2c)) = (f0_c, f1_c, f2_c) {
                    let f0 = parse(f0c)?;
                    let f1 = parse(f1c)?;
                    let f2 = parse(f2c)?;
                    r = clamp255((0.5 + SH_C0 * f0) * 255.0);
                    g = clamp255((0.5 + SH_C0 * f1) * 255.0);
                    b = clamp255((0.5 + SH_C0 * f2) * 255.0);
                }

                rgba[i] = rgba_to_u32(r, g, b, a);
            }
        }
    }

    Ok(SplatPlyBuffersCore {
        count: count as u32,
        format: header.format,
        center: center.into_boxed_slice(),
        covariance: covariance.into_boxed_slice(),
        rgba: rgba.into_boxed_slice(),
        bbox_min,
        bbox_max,
    })
}


