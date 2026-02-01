use rust_wasm::parsers::ply::parse_splat_ply_core_with_opts;
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy)]
struct Params {
    max_depth: u32,
    max_leaf_splats: usize,
    assume_log_scale: bool,
    assume_logit_opacity: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct OutputMeta {
    input_path: String,
    splat_count: u32,
    bbox_min: [f32; 3],
    bbox_max: [f32; 3],
    max_depth: u32,
    max_leaf_splats: usize,
    morton_bits_per_axis: u32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct OutputNode {
    id: u32,
    depth: u32,
    bbox_min: [f32; 3],
    bbox_max: [f32; 3],
    offset: u32,
    count: u32,
    children: [Option<u32>; 8],
    is_leaf: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct OutputStats {
    node_count: u32,
    leaf_count: u32,
    max_depth_reached: u32,
    min_leaf_splats: u32,
    max_leaf_splats: u32,
    avg_leaf_splats: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct OutputRoot {
    meta: OutputMeta,
    stats: OutputStats,
    nodes: Vec<OutputNode>,
}

#[derive(Debug, Clone, Copy)]
struct Range3 {
    x0: u32,
    x1: u32,
    y0: u32,
    y1: u32,
    z0: u32,
    z1: u32,
}

fn print_usage_and_exit() -> ! {
    eprintln!(
        "Usage:\n  cargo run --manifest-path wasm/Cargo.toml --bin splat_transform_lod -- <input.ply> <output.json> [--max-depth N] [--max-leaf-splats N] [--no-assume-log-scale] [--no-assume-logit-opacity]\n"
    );
    std::process::exit(2);
}

fn parse_args() -> (PathBuf, PathBuf, Params) {
    let mut it = std::env::args().skip(1);
    let input = it.next().unwrap_or_default();
    let output = it.next().unwrap_or_default();
    if input.is_empty() || output.is_empty() {
        print_usage_and_exit();
    }

    let mut params = Params {
        max_depth: 10,
        max_leaf_splats: 2048,
        assume_log_scale: true,
        assume_logit_opacity: true,
    };

    while let Some(a) = it.next() {
        match a.as_str() {
            "--max-depth" => {
                let v = it.next().unwrap_or_default();
                params.max_depth = v.parse::<u32>().unwrap_or_else(|_| {
                    eprintln!("Bad --max-depth value: {v}");
                    std::process::exit(2);
                });
            }
            "--max-leaf-splats" => {
                let v = it.next().unwrap_or_default();
                params.max_leaf_splats = v.parse::<usize>().unwrap_or_else(|_| {
                    eprintln!("Bad --max-leaf-splats value: {v}");
                    std::process::exit(2);
                });
            }
            "--no-assume-log-scale" => params.assume_log_scale = false,
            "--no-assume-logit-opacity" => params.assume_logit_opacity = false,
            _ => {
                eprintln!("Unknown arg: {a}");
                print_usage_and_exit();
            }
        }
    }

    if params.max_depth == 0 || params.max_depth > 21 {
        eprintln!("--max-depth must be in [1, 21] (fits in u64 Morton).");
        std::process::exit(2);
    }
    if params.max_leaf_splats == 0 {
        eprintln!("--max-leaf-splats must be > 0.");
        std::process::exit(2);
    }

    (PathBuf::from(input), PathBuf::from(output), params)
}

#[inline]
fn clamp_u32(v: i64, lo: i64, hi: i64) -> u32 {
    if v < lo {
        lo as u32
    } else if v > hi {
        hi as u32
    } else {
        v as u32
    }
}

fn quantize_to_grid(p: f32, min: f32, max: f32, grid: u32) -> u32 {
    if !(max > min) {
        return 0;
    }
    let t = ((p - min) / (max - min)).clamp(0.0, 0.99999994);
    let v = (t * (grid as f32)) as i64;
    clamp_u32(v, 0, (grid as i64) - 1)
}

fn morton3_u64(x: u32, y: u32, z: u32, bits: u32) -> u64 {
    // Pack triplets from MSB to LSB so prefixes correspond to octree nodes.
    let mut code: u64 = 0;
    for b in (0..bits).rev() {
        let xb = ((x >> b) & 1) as u64;
        let yb = ((y >> b) & 1) as u64;
        let zb = ((z >> b) & 1) as u64;
        code = (code << 3) | (xb << 2) | (yb << 1) | zb;
    }
    code
}

fn bbox_from_range(range: Range3, bits: u32, bbox_min: [f32; 3], bbox_max: [f32; 3]) -> ([f32; 3], [f32; 3]) {
    let grid = 1u32 << bits;
    let dx = (bbox_max[0] - bbox_min[0]) / (grid as f32);
    let dy = (bbox_max[1] - bbox_min[1]) / (grid as f32);
    let dz = (bbox_max[2] - bbox_min[2]) / (grid as f32);

    let min = [
        bbox_min[0] + (range.x0 as f32) * dx,
        bbox_min[1] + (range.y0 as f32) * dy,
        bbox_min[2] + (range.z0 as f32) * dz,
    ];
    let max = [
        bbox_min[0] + (range.x1 as f32) * dx,
        bbox_min[1] + (range.y1 as f32) * dy,
        bbox_min[2] + (range.z1 as f32) * dz,
    ];
    (min, max)
}

fn child_range(parent: Range3, child: u32) -> Range3 {
    let xm = (parent.x0 + parent.x1) / 2;
    let ym = (parent.y0 + parent.y1) / 2;
    let zm = (parent.z0 + parent.z1) / 2;
    let x_hi = ((child >> 2) & 1) != 0;
    let y_hi = ((child >> 1) & 1) != 0;
    let z_hi = (child & 1) != 0;
    Range3 {
        x0: if x_hi { xm } else { parent.x0 },
        x1: if x_hi { parent.x1 } else { xm },
        y0: if y_hi { ym } else { parent.y0 },
        y1: if y_hi { parent.y1 } else { ym },
        z0: if z_hi { zm } else { parent.z0 },
        z1: if z_hi { parent.z1 } else { zm },
    }
}

fn build_tree(
    codes: &[u64],
    params: Params,
    bbox_min: [f32; 3],
    bbox_max: [f32; 3],
) -> (Vec<OutputNode>, OutputStats) {
    let bits = params.max_depth;
    let grid = 1u32 << bits;
    let root_range = Range3 {
        x0: 0,
        x1: grid,
        y0: 0,
        y1: grid,
        z0: 0,
        z1: grid,
    };

    let mut nodes: Vec<OutputNode> = Vec::new();

    fn rec(
        nodes: &mut Vec<OutputNode>,
        codes: &[u64],
        params: Params,
        bbox_min: [f32; 3],
        bbox_max: [f32; 3],
        bits: u32,
        depth: u32,
        start: usize,
        end: usize,
        range: Range3,
    ) -> u32 {
        let id = nodes.len() as u32;
        let count = (end - start) as u32;
        let (bmin, bmax) = bbox_from_range(range, bits, bbox_min, bbox_max);

        let out = OutputNode {
            id,
            depth,
            bbox_min: bmin,
            bbox_max: bmax,
            offset: start as u32,
            count,
            children: [None, None, None, None, None, None, None, None],
            is_leaf: true,
        };

        nodes.push(out);

        let should_split = (end - start) > params.max_leaf_splats && depth < bits;
        if !should_split {
            return id;
        }

        let shift = 3 * (bits - 1 - depth);
        let mut i = start;
        while i < end {
            let c = ((codes[i] >> shift) & 7) as u32;
            let mut j = i + 1;
            while j < end && (((codes[j] >> shift) & 7) as u32) == c {
                j += 1;
            }

            let child_id = rec(
                nodes,
                codes,
                params,
                bbox_min,
                bbox_max,
                bits,
                depth + 1,
                i,
                j,
                child_range(range, c),
            );
            nodes[id as usize].children[c as usize] = Some(child_id);
            nodes[id as usize].is_leaf = false;

            i = j;
        }

        id
    }

    rec(
        &mut nodes,
        codes,
        params,
        bbox_min,
        bbox_max,
        bits,
        0,
        0,
        codes.len(),
        root_range,
    );

    // Stats
    let mut leaf_count = 0u32;
    let node_count = nodes.len() as u32;
    let mut max_depth_reached = 0u32;
    let mut min_leaf = u32::MAX;
    let mut max_leaf = 0u32;
    let mut sum_leaf: u64 = 0;

    for n in nodes.iter() {
        max_depth_reached = max_depth_reached.max(n.depth);
        if n.is_leaf {
            leaf_count += 1;
            min_leaf = min_leaf.min(n.count);
            max_leaf = max_leaf.max(n.count);
            sum_leaf += n.count as u64;
        }
    }

    if leaf_count == 0 {
        min_leaf = 0;
    }

    let avg_leaf = if leaf_count > 0 {
        (sum_leaf as f64) / (leaf_count as f64)
    } else {
        0.0
    };

    let stats = OutputStats {
        node_count,
        leaf_count,
        max_depth_reached,
        min_leaf_splats: min_leaf,
        max_leaf_splats: max_leaf,
        avg_leaf_splats: avg_leaf,
    };

    (nodes, stats)
}

fn ensure_parent_dir(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn main() {
    let (input_path, output_path, params) = parse_args();

    let bytes = fs::read(&input_path).unwrap_or_else(|e| {
        eprintln!("Failed to read {}: {e}", input_path.display());
        std::process::exit(1);
    });

    let splats = parse_splat_ply_core_with_opts(&bytes, params.assume_log_scale, params.assume_logit_opacity)
        .unwrap_or_else(|e| {
            eprintln!("Failed to parse PLY: {e}");
            std::process::exit(1);
        });

    let bits = params.max_depth;
    let grid = 1u32 << bits;

    // Build Morton codes in max-depth grid.
    let n = splats.count as usize;
    let mut codes_with_idx: Vec<(u64, u32)> = Vec::with_capacity(n);

    for i in 0..n {
        let cx = splats.center[i * 3];
        let cy = splats.center[i * 3 + 1];
        let cz = splats.center[i * 3 + 2];

        let qx = quantize_to_grid(cx, splats.bbox_min[0], splats.bbox_max[0], grid);
        let qy = quantize_to_grid(cy, splats.bbox_min[1], splats.bbox_max[1], grid);
        let qz = quantize_to_grid(cz, splats.bbox_min[2], splats.bbox_max[2], grid);

        let code = morton3_u64(qx, qy, qz, bits);
        codes_with_idx.push((code, i as u32));
    }

    codes_with_idx.sort_by_key(|(c, _)| *c);
    let codes: Vec<u64> = codes_with_idx.iter().map(|(c, _)| *c).collect();

    let (nodes, stats) = build_tree(&codes, params, splats.bbox_min, splats.bbox_max);

    let out = OutputRoot {
        meta: OutputMeta {
            input_path: input_path.display().to_string(),
            splat_count: splats.count,
            bbox_min: splats.bbox_min,
            bbox_max: splats.bbox_max,
            max_depth: params.max_depth,
            max_leaf_splats: params.max_leaf_splats,
            morton_bits_per_axis: params.max_depth,
        },
        stats,
        nodes,
    };

    ensure_parent_dir(&output_path).unwrap_or_else(|e| {
        eprintln!("Failed to create output directory: {e}");
        std::process::exit(1);
    });

    let json = serde_json::to_string_pretty(&out).unwrap_or_else(|e| {
        eprintln!("Failed to serialize JSON: {e}");
        std::process::exit(1);
    });
    fs::write(&output_path, json).unwrap_or_else(|e| {
        eprintln!("Failed to write {}: {e}", output_path.display());
        std::process::exit(1);
    });

    eprintln!(
        "Wrote LOD tree: {} (nodes={}, leaves={}, maxDepthReached={})",
        output_path.display(),
        out.stats.node_count,
        out.stats.leaf_count,
        out.stats.max_depth_reached
    );
}
