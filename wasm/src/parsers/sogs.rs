use crate::splats::Splats;
use crate::splats_parser::{ParseError, SplatsParser};

use image::GenericImageView;
use serde::Deserialize;
use std::io::{Cursor, Read};
use zip::ZipArchive;

#[derive(Debug, Deserialize)]
struct PcSogsV2Json {
    version: u32,
    count: u32,
    #[allow(dead_code)]
    antialias: Option<bool>,
    means: MeansV2,
    scales: CodebookFilesV2,
    quats: FilesV2,
    sh0: CodebookFilesV2,
    shN: Option<ShNV2>,
}

#[derive(Debug, Deserialize)]
struct MeansV2 {
    mins: Vec<f32>,
    maxs: Vec<f32>,
    files: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct FilesV2 {
    files: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct CodebookFilesV2 {
    codebook: Vec<f32>,
    files: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ShNV2 {
    #[allow(dead_code)]
    count: u32,
    bands: u32,
    codebook: Vec<f32>,
    files: Vec<String>,
}

pub struct SogsV2Parser;

impl SplatsParser for SogsV2Parser {
    fn parse(bytes: &[u8]) -> Result<Splats, ParseError> {
        parse_sogs_v2_zip(bytes)
    }
}

fn parse_sogs_v2_zip(bytes: &[u8]) -> Result<Splats, ParseError> {
    // Expect a PKZip container (".sog") containing meta.json + referenced images.
    if bytes.len() < 4 || &bytes[0..4] != b"PK\x03\x04" {
        return Err(ParseError::msg("SOGS v2: expected ZIP (PK) container"));
    }

    let mut zip = ZipArchive::new(Cursor::new(bytes)).map_err(|e| {
        ParseError::msg(format!("SOGS v2: failed to read zip archive: {e}"))
    })?;

    let meta_index = find_meta_json_index(&mut zip)?;
    let meta_name = zip
        .by_index(meta_index)
        .map_err(|e| ParseError::msg(format!("SOGS v2: zip read meta.json: {e}")))?
        .name()
        .to_string();
    let prefix = path_prefix(&meta_name);

    let meta_bytes = read_zip_file_by_index(&mut zip, meta_index)?;
    let meta_text = std::str::from_utf8(&meta_bytes)
        .map_err(|_| ParseError::msg("SOGS v2: meta.json is not valid utf-8"))?;

    let json: PcSogsV2Json = serde_json::from_str(meta_text)
        .map_err(|e| ParseError::msg(format!("SOGS v2: invalid meta.json: {e}")))?;
    if json.version != 2 {
        return Err(ParseError::msg(format!(
            "SOGS v2: unsupported version {}",
            json.version
        )));
    }

    let count = json.count as usize;
    if count == 0 {
        return Ok(empty_splats("sogs_v2"));
    }

    // Load required images from zip.
    let means0 = read_zip_file_by_name_with_prefix(&mut zip, &prefix, &json.means.files, 0)?;
    let means1 = read_zip_file_by_name_with_prefix(&mut zip, &prefix, &json.means.files, 1)?;
    let scales0 = read_zip_file_by_name_with_prefix(&mut zip, &prefix, &json.scales.files, 0)?;
    let quats0 = read_zip_file_by_name_with_prefix(&mut zip, &prefix, &json.quats.files, 0)?;
    let sh00 = read_zip_file_by_name_with_prefix(&mut zip, &prefix, &json.sh0.files, 0)?;

    let means0 = decode_rgba(&means0)?;
    let means1 = decode_rgba(&means1)?;
    let scales0 = decode_rgba(&scales0)?;
    let quats0 = decode_rgba(&quats0)?;
    let sh00 = decode_rgba(&sh00)?;

    let mins = &json.means.mins;
    let maxs = &json.means.maxs;
    if mins.len() < 3 || maxs.len() < 3 {
        return Err(ParseError::msg("SOGS v2: means.mins/maxs must have 3 values"));
    }

    if means0.rgba.len() < count * 4 || means1.rgba.len() < count * 4 {
        return Err(ParseError::msg(
            "SOGS v2: means images are smaller than count",
        ));
    }
    if scales0.rgba.len() < count * 4 {
        return Err(ParseError::msg(
            "SOGS v2: scales image is smaller than count",
        ));
    }
    if quats0.rgba.len() < count * 4 {
        return Err(ParseError::msg(
            "SOGS v2: quats image is smaller than count",
        ));
    }
    if sh00.rgba.len() < count * 4 {
        return Err(ParseError::msg("SOGS v2: sh0 image is smaller than count"));
    }

    // Decode centers.
    let mut center = vec![0.0f32; count * 3];
    let mut bbox_min = [f32::INFINITY, f32::INFINITY, f32::INFINITY];
    let mut bbox_max = [f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];

    for i in 0..count {
        let i4 = i * 4;
        let fx =
            (means0.rgba[i4] as u32 + ((means1.rgba[i4] as u32) << 8)) as f32 / 65535.0;
        let fy = (means0.rgba[i4 + 1] as u32 + ((means1.rgba[i4 + 1] as u32) << 8)) as f32
            / 65535.0;
        let fz = (means0.rgba[i4 + 2] as u32 + ((means1.rgba[i4 + 2] as u32) << 8)) as f32
            / 65535.0;

        let mut x = mins[0] + (maxs[0] - mins[0]) * fx;
        let mut y = mins[1] + (maxs[1] - mins[1]) * fy;
        let mut z = mins[2] + (maxs[2] - mins[2]) * fz;

        x = x.signum() * (x.abs().exp() - 1.0);
        y = y.signum() * (y.abs().exp() - 1.0);
        z = z.signum() * (z.abs().exp() - 1.0);

        let i3 = i * 3;
        center[i3] = x;
        center[i3 + 1] = y;
        center[i3 + 2] = z;

        bbox_min[0] = bbox_min[0].min(x);
        bbox_min[1] = bbox_min[1].min(y);
        bbox_min[2] = bbox_min[2].min(z);
        bbox_max[0] = bbox_max[0].max(x);
        bbox_max[1] = bbox_max[1].max(y);
        bbox_max[2] = bbox_max[2].max(z);
    }

    // Decode scales via codebook (shared for x/y/z) and exponentiate.
    if json.scales.codebook.len() < 256 {
        return Err(ParseError::msg("SOGS v2: scales.codebook must have 256 values"));
    }
    let mut scale_lookup = [0.0f32; 256];
    for i in 0..256 {
        scale_lookup[i] = json.scales.codebook[i].exp();
    }
    let mut scales = vec![0.0f32; count * 3];
    for i in 0..count {
        let i4 = i * 4;
        let sx = scale_lookup[scales0.rgba[i4] as usize];
        let sy = scale_lookup[scales0.rgba[i4 + 1] as usize];
        let sz = scale_lookup[scales0.rgba[i4 + 2] as usize];
        let i3 = i * 3;
        scales[i3] = sx;
        scales[i3 + 1] = sy;
        scales[i3 + 2] = sz;
    }

    // Decode quaternions (packed 3 values + order in alpha).
    let sqrt2 = 2.0f32.sqrt();
    let mut quat_lookup = [0.0f32; 256];
    for i in 0..256 {
        quat_lookup[i] = (i as f32 / 255.0 - 0.5) * sqrt2;
    }
    let mut quats = vec![0.0f32; count * 4];
    for i in 0..count {
        let i4 = i * 4;
        let r0 = quat_lookup[quats0.rgba[i4] as usize];
        let r1 = quat_lookup[quats0.rgba[i4 + 1] as usize];
        let r2 = quat_lookup[quats0.rgba[i4 + 2] as usize];
        let rr = (1.0 - r0 * r0 - r1 * r1 - r2 * r2).max(0.0).sqrt();
        let r_order = quats0.rgba[i4 + 3] as i32 - 252;
        if !(0..=3).contains(&r_order) {
            return Err(ParseError::msg("SOGS v2: invalid quaternion order"));
        }
        let qx = if r_order == 0 {
            r0
        } else if r_order == 1 {
            rr
        } else {
            r1
        };
        let qy = if r_order <= 1 {
            r1
        } else if r_order == 2 {
            rr
        } else {
            r2
        };
        let qz = if r_order <= 2 { r2 } else { rr };
        let qw = if r_order == 0 { rr } else { r0 };
        quats[i4] = qx;
        quats[i4 + 1] = qy;
        quats[i4 + 2] = qz;
        quats[i4 + 3] = qw;
    }

    // Decode SH0 -> rgba (rgb via SH_C0*codebook + 0.5, alpha = idx/255).
    if json.sh0.codebook.len() < 256 {
        return Err(ParseError::msg("SOGS v2: sh0.codebook must have 256 values"));
    }
    const SH_C0: f32 = 0.28209479177387814;
    let mut sh0_rgb_lookup = [0.0f32; 256];
    for i in 0..256 {
        sh0_rgb_lookup[i] = SH_C0 * json.sh0.codebook[i] + 0.5;
    }
    let mut rgba = vec![0u32; count];
    for i in 0..count {
        let i4 = i * 4;
        let r = clamp255(sh0_rgb_lookup[sh00.rgba[i4] as usize] * 255.0);
        let g = clamp255(sh0_rgb_lookup[sh00.rgba[i4 + 1] as usize] * 255.0);
        let b = clamp255(sh0_rgb_lookup[sh00.rgba[i4 + 2] as usize] * 255.0);
        let a = clamp255((sh00.rgba[i4 + 3] as f32 / 255.0) * 255.0);
        rgba[i] = rgba_to_u32(r, g, b, a);
    }

    // Compute covariance from quats + scales.
    let mut covariance = vec![0.0f32; count * 6];
    for i in 0..count {
        let i4 = i * 4;
        let i3 = i * 3;
        let cov = covariance_from_quat_scale(
            quats[i4],
            quats[i4 + 1],
            quats[i4 + 2],
            quats[i4 + 3],
            scales[i3],
            scales[i3 + 1],
            scales[i3 + 2],
        );
        let i6 = i * 6;
        covariance[i6..i6 + 6].copy_from_slice(&cov);
    }

    // Decode SHN (optional).
    let mut sh_degree = 0u32;
    let mut sh_coeffs_l1: Vec<f32> = Vec::new();
    let mut sh_coeffs_l2: Vec<f32> = Vec::new();
    let mut sh_coeffs_l3: Vec<f32> = Vec::new();
    if let Some(shn) = &json.shN {
        let use_sh3 = shn.bands >= 3;
        let use_sh2 = shn.bands >= 2;
        let use_sh1 = shn.bands >= 1;
        sh_degree = if use_sh3 { 3 } else if use_sh2 { 2 } else if use_sh1 { 1 } else { 0 };

        if use_sh1 {
            sh_coeffs_l1 = Vec::with_capacity(count * 9);
        }
        if use_sh2 {
            sh_coeffs_l2 = Vec::with_capacity(count * 15);
        }
        if use_sh3 {
            sh_coeffs_l3 = Vec::with_capacity(count * 21);
        }

        if shn.codebook.len() < 256 {
            return Err(ParseError::msg("SOGS v2: shN.codebook must have 256 values"));
        }
        let mut sh_lookup = [0.0f32; 256];
        for i in 0..256 {
            sh_lookup[i] = shn.codebook[i];
        }

        if shn.files.len() < 2 {
            return Err(ParseError::msg("SOGS v2: shN.files must have 2 entries"));
        }
        let centroids_bytes = read_zip_file_by_name(&mut zip, &(prefix.clone() + &shn.files[0]))
            .or_else(|_| read_zip_file_by_name(&mut zip, &shn.files[0]))?;
        let labels_bytes = read_zip_file_by_name(&mut zip, &(prefix.clone() + &shn.files[1]))
            .or_else(|_| read_zip_file_by_name(&mut zip, &shn.files[1]))?;
        let centroids = decode_rgba(&centroids_bytes)?;
        let labels = decode_rgba(&labels_bytes)?;
        if labels.rgba.len() < count * 4 {
            return Err(ParseError::msg("SOGS v2: shN labels image is smaller than count"));
        }

        let width = centroids.width as usize;
        let pixel_count = (centroids.rgba.len() / 4) as usize;
        for i in 0..count {
            let i4 = i * 4;
            let label = labels.rgba[i4] as u32 + ((labels.rgba[i4 + 1] as u32) << 8);
            let col = (label & 63) as usize * 15;
            let row = (label >> 6) as usize;
            let offset = row * width + col;
            let needed = offset + 15; // we may read up to offset+14
            if needed > pixel_count {
                return Err(ParseError::msg("SOGS v2: shN label points outside centroids image"));
            }

            if use_sh1 {
                for k in 0..3usize {
                    let base = (offset + k) * 4;
                    sh_coeffs_l1.push(sh_lookup[centroids.rgba[base + 0] as usize]);
                    sh_coeffs_l1.push(sh_lookup[centroids.rgba[base + 1] as usize]);
                    sh_coeffs_l1.push(sh_lookup[centroids.rgba[base + 2] as usize]);
                }
            }
            if use_sh2 {
                for k in 0..5usize {
                    let base = (offset + 3 + k) * 4;
                    sh_coeffs_l2.push(sh_lookup[centroids.rgba[base + 0] as usize]);
                    sh_coeffs_l2.push(sh_lookup[centroids.rgba[base + 1] as usize]);
                    sh_coeffs_l2.push(sh_lookup[centroids.rgba[base + 2] as usize]);
                }
            }
            if use_sh3 {
                for k in 0..7usize {
                    let base = (offset + 8 + k) * 4;
                    sh_coeffs_l3.push(sh_lookup[centroids.rgba[base + 0] as usize]);
                    sh_coeffs_l3.push(sh_lookup[centroids.rgba[base + 1] as usize]);
                    sh_coeffs_l3.push(sh_lookup[centroids.rgba[base + 2] as usize]);
                }
            }
        }
    }

    let (sh_coeffs_l2_packed, sh_coeffs_l2_scale) = pack_sh_coeffs_i16(&sh_coeffs_l2);
    let (sh_coeffs_l3_packed, sh_coeffs_l3_scale) = pack_sh_coeffs_i16(&sh_coeffs_l3);

    Ok(Splats {
        count: json.count,
        format: "sogs_v2".to_string(),
        center: center.into_boxed_slice(),
        covariance: covariance.into_boxed_slice(),
        rgba: rgba.into_boxed_slice(),
        sh_coeffs_l1: sh_coeffs_l1.into_boxed_slice(),
        sh_coeffs_l2_packed: sh_coeffs_l2_packed.into_boxed_slice(),
        sh_coeffs_l2_scale,
        sh_coeffs_l3_packed: sh_coeffs_l3_packed.into_boxed_slice(),
        sh_coeffs_l3_scale,
        sh_degree,
        bbox_min,
        bbox_max,
    })
}

fn empty_splats(format: &str) -> Splats {
    Splats {
        count: 0,
        format: format.to_string(),
        center: Box::new([]),
        covariance: Box::new([]),
        rgba: Box::new([]),
        sh_coeffs_l1: Box::new([]),
        sh_coeffs_l2_packed: Box::new([]),
        sh_coeffs_l2_scale: 1.0,
        sh_coeffs_l3_packed: Box::new([]),
        sh_coeffs_l3_scale: 1.0,
        sh_degree: 0,
        bbox_min: [0.0; 3],
        bbox_max: [0.0; 3],
    }
}

fn find_meta_json_index(zip: &mut ZipArchive<Cursor<&[u8]>>) -> Result<usize, ParseError> {
    for i in 0..zip.len() {
        let name = zip
            .by_index(i)
            .map_err(|e| ParseError::msg(format!("SOGS v2: zip iterate: {e}")))?
            .name()
            .to_string();
        let base = name
            .rsplit(['/', '\\'])
            .next()
            .unwrap_or(&name)
            .to_string();
        if base == "meta.json" {
            return Ok(i);
        }
    }
    Err(ParseError::msg("SOGS v2: meta.json not found in zip"))
}

fn path_prefix(path: &str) -> String {
    let last_slash = path.rfind('/').unwrap_or(0);
    let last_backslash = path.rfind('\\').unwrap_or(0);
    let idx = last_slash.max(last_backslash);
    if idx == 0 && !(path.starts_with('/') || path.starts_with('\\')) {
        // No prefix
        return String::new();
    }
    path[..=idx].to_string()
}

fn read_zip_file_by_name_with_prefix(
    zip: &mut ZipArchive<Cursor<&[u8]>>,
    prefix: &str,
    files: &[String],
    idx: usize,
) -> Result<Vec<u8>, ParseError> {
    let name = files
        .get(idx)
        .ok_or_else(|| ParseError::msg("SOGS v2: missing referenced file"))?;
    // Try with meta.json prefix first, then raw.
    read_zip_file_by_name(zip, &(prefix.to_string() + name)).or_else(|_| read_zip_file_by_name(zip, name))
}

fn read_zip_file_by_name(
    zip: &mut ZipArchive<Cursor<&[u8]>>,
    name: &str,
) -> Result<Vec<u8>, ParseError> {
    let mut f = zip
        .by_name(name)
        .map_err(|_| ParseError::msg(format!("SOGS v2: missing file in zip: {name}")))?;
    let mut out = Vec::with_capacity(f.size() as usize);
    f.read_to_end(&mut out)
        .map_err(|e| ParseError::msg(format!("SOGS v2: failed reading {name}: {e}")))?;
    Ok(out)
}

fn read_zip_file_by_index(
    zip: &mut ZipArchive<Cursor<&[u8]>>,
    index: usize,
) -> Result<Vec<u8>, ParseError> {
    let mut f = zip
        .by_index(index)
        .map_err(|e| ParseError::msg(format!("SOGS v2: zip read by index: {e}")))?;
    let mut out = Vec::with_capacity(f.size() as usize);
    f.read_to_end(&mut out)
        .map_err(|e| ParseError::msg(format!("SOGS v2: zip read failed: {e}")))?;
    Ok(out)
}

struct DecodedRgba {
    rgba: Vec<u8>,
    width: u32,
    height: u32,
}

fn decode_rgba(bytes: &[u8]) -> Result<DecodedRgba, ParseError> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| ParseError::msg(format!("SOGS v2: image decode failed: {e}")))?;
    let (w, h) = img.dimensions();
    let rgba = img.into_rgba8().into_raw();
    Ok(DecodedRgba {
        rgba,
        width: w,
        height: h,
    })
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
    (r & 255) | ((g & 255) << 8) | ((b & 255) << 16) | ((a & 255) << 24)
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

fn covariance_from_quat_scale(
    qx: f32,
    qy: f32,
    qz: f32,
    qw: f32,
    sx: f32,
    sy: f32,
    sz: f32,
) -> [f32; 6] {
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

fn pack_sh_coeffs_i16(sh: &[f32]) -> (Vec<u32>, f32) {
    if sh.is_empty() {
        return (Vec::new(), 1.0);
    }
    let mut max_abs = 0.0f32;
    for v in sh.iter() {
        max_abs = max_abs.max(v.abs());
    }
    let scale = if max_abs > 0.0 { max_abs / 32767.0 } else { 1.0 };
    let mut out: Vec<u32> = Vec::with_capacity((sh.len() / 3) * 2);
    let mut i = 0usize;
    while i + 2 < sh.len() {
        let q = |v: f32| -> i16 {
            let scaled = (v / scale).round().clamp(-32767.0, 32767.0);
            scaled as i16
        };
        let r = q(sh[i]) as u16;
        let g = q(sh[i + 1]) as u16;
        let b = q(sh[i + 2]) as u16;
        let u0 = (r as u32) | ((g as u32) << 16);
        let u1 = b as u32;
        out.push(u0);
        out.push(u1);
        i += 3;
    }
    (out, scale)
}

