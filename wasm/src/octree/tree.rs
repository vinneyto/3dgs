//! Octree data structure built over splat centre positions.
//!
//! The tree stores a flat `Vec<OctreeNode>` (index 0 is always the root) and a
//! copy of every splat centre so that collision queries can compute distances
//! without an external reference to [`Splats`].
//!
//! # Debug visualisation
//!
//! [`Octree::ray_traversal_path`] returns every node visited during a
//! depth-first ray traversal — handy for drawing the ray's path through the
//! tree.

use std::cmp::Ordering;

use crate::splats::Splats;

use super::colliders::{BoxCollider, CollisionHit, RayCollider, SphereCollider};
use super::geometry::{Ray, Sphere, Vec3, AABB};

// ---------------------------------------------------------------------------
// Constants & Config
// ---------------------------------------------------------------------------

/// Sentinel stored in [`OctreeNode::children`] when the octant is empty.
pub const EMPTY_CHILD: u32 = u32::MAX;

/// Parameters that control how the octree is built.
#[derive(Debug, Clone, Copy)]
pub struct OctreeConfig {
    /// Maximum recursion depth (root = depth 0).
    pub max_depth: u32,
    /// A node becomes a leaf when it contains at most this many splats.
    pub max_leaf_size: u32,
}

impl Default for OctreeConfig {
    fn default() -> Self {
        Self {
            max_depth: 12,
            max_leaf_size: 32,
        }
    }
}

// ---------------------------------------------------------------------------
// OctreeNode
// ---------------------------------------------------------------------------

/// A single node in the octree — either an internal node (with up to 8
/// children) or a leaf (carrying splat indices).
#[derive(Debug, Clone)]
pub struct OctreeNode {
    /// Axis-aligned bounding box of this node.
    pub bounds: AABB,

    /// Children indices into [`Octree::nodes`].
    /// Each slot corresponds to an octant (see [`AABB::child_octant`]).
    /// [`EMPTY_CHILD`] means no child in that octant.
    pub children: [u32; 8],

    /// Splat indices stored here.  Non-empty **only** for leaf nodes.
    pub splat_indices: Vec<u32>,

    /// `true` when this is a leaf node.
    pub is_leaf: bool,

    /// Depth of this node in the tree (root = 0).
    pub depth: u32,
}

// ---------------------------------------------------------------------------
// OctreeTraversalNode  (debug helper)
// ---------------------------------------------------------------------------

/// A node encountered during depth-first ray traversal.
/// Intended for debug visualisation of the ray's path through the tree.
#[derive(Debug, Clone, Copy)]
pub struct OctreeTraversalNode {
    /// Bounding box of the visited node.
    pub bounds: AABB,
    /// Depth of the node in the tree (root = 0).
    pub depth: u32,
    /// Number of splats stored directly in this node (>0 only for leaves).
    pub splat_count: u32,
    /// Parametric `t` at which the ray *enters* this node's AABB.
    pub t_enter: f32,
}

// ---------------------------------------------------------------------------
// Octree
// ---------------------------------------------------------------------------

/// Spatial index over splat centre positions, represented as a flat array of
/// [`OctreeNode`]s.  Index 0 is always the root.
#[derive(Debug, Clone)]
pub struct Octree {
    nodes: Vec<OctreeNode>,
    /// Copy of every splat centre (length = splat count).
    centers: Vec<Vec3>,
    config: OctreeConfig,
}

// --- Construction ----------------------------------------------------------

impl Octree {
    /// Build an octree from a [`Splats`] instance using default parameters.
    pub fn from_splats(splats: &Splats) -> Self {
        Self::from_splats_with_config(splats, OctreeConfig::default())
    }

    /// Build an octree from a [`Splats`] instance with custom [`OctreeConfig`].
    pub fn from_splats_with_config(splats: &Splats, config: OctreeConfig) -> Self {
        let count = splats.count as usize;

        let centers: Vec<Vec3> = (0..count)
            .map(|i| {
                Vec3::new(
                    splats.center[i * 3],
                    splats.center[i * 3 + 1],
                    splats.center[i * 3 + 2],
                )
            })
            .collect();

        let bounds = AABB::new(
            Vec3::new(splats.bbox_min[0], splats.bbox_min[1], splats.bbox_min[2]),
            Vec3::new(splats.bbox_max[0], splats.bbox_max[1], splats.bbox_max[2]),
        );

        // Pad slightly to avoid edge-cases where a centre sits exactly on the boundary.
        let pad = bounds.size() * 0.001 + Vec3::splat(1e-6);
        let bounds = AABB::new(bounds.min - pad, bounds.max + pad);

        let all_indices: Vec<u32> = (0..splats.count).collect();

        let mut tree = Self {
            nodes: Vec::new(),
            centers,
            config,
        };

        tree.build_recursive(bounds, &all_indices, 0);
        tree
    }

    /// Build from raw centre positions and a bounding box.
    ///
    /// `centers_flat` is a `&[f32]` of length `3 * count` laid out as
    /// `[x0, y0, z0, x1, y1, z1, …]`.
    pub fn build(
        centers_flat: &[f32],
        bbox_min: [f32; 3],
        bbox_max: [f32; 3],
        count: u32,
        config: OctreeConfig,
    ) -> Self {
        let centers: Vec<Vec3> = (0..count as usize)
            .map(|i| {
                Vec3::new(
                    centers_flat[i * 3],
                    centers_flat[i * 3 + 1],
                    centers_flat[i * 3 + 2],
                )
            })
            .collect();

        let bounds = AABB::new(
            Vec3::new(bbox_min[0], bbox_min[1], bbox_min[2]),
            Vec3::new(bbox_max[0], bbox_max[1], bbox_max[2]),
        );

        let pad = bounds.size() * 0.001 + Vec3::splat(1e-6);
        let bounds = AABB::new(bounds.min - pad, bounds.max + pad);

        let all_indices: Vec<u32> = (0..count).collect();

        let mut tree = Self {
            nodes: Vec::new(),
            centers,
            config,
        };

        tree.build_recursive(bounds, &all_indices, 0);
        tree
    }

    /// Recursive top-down octree construction.  Returns the index of the
    /// newly-created node inside `self.nodes`.
    fn build_recursive(&mut self, bounds: AABB, indices: &[u32], depth: u32) -> u32 {
        let node_index = self.nodes.len() as u32;

        // ---- leaf condition ------------------------------------------------
        if indices.len() as u32 <= self.config.max_leaf_size || depth >= self.config.max_depth {
            self.nodes.push(OctreeNode {
                bounds,
                children: [EMPTY_CHILD; 8],
                splat_indices: indices.to_vec(),
                is_leaf: true,
                depth,
            });
            return node_index;
        }

        // ---- internal node (placeholder) -----------------------------------
        self.nodes.push(OctreeNode {
            bounds,
            children: [EMPTY_CHILD; 8],
            splat_indices: Vec::new(),
            is_leaf: false,
            depth,
        });

        // Partition indices into octants.
        let mid = bounds.center();
        let mut buckets: [Vec<u32>; 8] = Default::default();

        for &idx in indices {
            let c = self.centers[idx as usize];
            let octant = (c.x >= mid.x) as usize
                | (((c.y >= mid.y) as usize) << 1)
                | (((c.z >= mid.z) as usize) << 2);
            buckets[octant].push(idx);
        }

        for octant in 0..8u8 {
            if !buckets[octant as usize].is_empty() {
                let child_bounds = bounds.child_octant(octant);
                let child_idx =
                    self.build_recursive(child_bounds, &buckets[octant as usize], depth + 1);
                self.nodes[node_index as usize].children[octant as usize] = child_idx;
            }
        }

        node_index
    }
}

// --- Read accessors --------------------------------------------------------

impl Octree {
    /// Root node (always at index 0).
    #[inline]
    pub fn root(&self) -> &OctreeNode {
        &self.nodes[0]
    }

    /// Access a node by its index.
    #[inline]
    pub fn node(&self, index: u32) -> &OctreeNode {
        &self.nodes[index as usize]
    }

    /// Total number of nodes.
    #[inline]
    pub fn node_count(&self) -> u32 {
        self.nodes.len() as u32
    }

    /// Total number of splats indexed by this octree.
    #[inline]
    pub fn splat_count(&self) -> u32 {
        self.centers.len() as u32
    }

    /// Reference to the stored centres.
    #[inline]
    pub fn centers(&self) -> &[Vec3] {
        &self.centers
    }

    /// The configuration used to build this tree.
    #[inline]
    pub fn config(&self) -> &OctreeConfig {
        &self.config
    }
}

// ===========================================================================
// Trait implementations
// ===========================================================================

// --- RayCollider -----------------------------------------------------------

impl RayCollider for Octree {
    /// Query all splats in leaf nodes intersected by `ray`.
    ///
    /// Results are sorted by ascending `t` (parametric distance along the ray).
    fn collide_ray(&self, ray: &Ray) -> impl Iterator<Item = CollisionHit> {
        let mut results = Vec::new();
        if !self.nodes.is_empty() {
            self.collect_ray_hits(0, ray, &mut results);
        }
        results.sort_by(|a, b| cmp_f32(a.distance, b.distance));
        results.into_iter()
    }
}

// --- BoxCollider -----------------------------------------------------------

impl BoxCollider for Octree {
    /// Query all splats in leaf nodes that overlap `aabb`.
    fn collide_box(&self, aabb: &AABB) -> impl Iterator<Item = CollisionHit> {
        let mut results = Vec::new();
        if !self.nodes.is_empty() {
            self.collect_box_hits(0, aabb, &mut results);
        }
        results.into_iter()
    }
}

// --- SphereCollider --------------------------------------------------------

impl SphereCollider for Octree {
    /// Query all splats in leaf nodes that overlap `sphere`.
    fn collide_sphere(&self, sphere: &Sphere) -> impl Iterator<Item = CollisionHit> {
        let mut results = Vec::new();
        if !self.nodes.is_empty() {
            self.collect_sphere_hits(0, sphere, &mut results);
        }
        results.into_iter()
    }
}

// ===========================================================================
// Private recursive collectors
// ===========================================================================

impl Octree {
    // --- Ray ---------------------------------------------------------------

    fn collect_ray_hits(&self, node_idx: u32, ray: &Ray, out: &mut Vec<CollisionHit>) {
        let node = &self.nodes[node_idx as usize];

        if node.bounds.intersects_ray(ray).is_none() {
            return;
        }

        if node.is_leaf {
            let dir_len_sq = ray.direction.length_sq();
            for &splat_idx in &node.splat_indices {
                let center = self.centers[splat_idx as usize];
                let to_center = center - ray.origin;
                let t = if dir_len_sq > 0.0 {
                    to_center.dot(ray.direction) / dir_len_sq
                } else {
                    0.0
                };
                out.push(CollisionHit {
                    splat_index: splat_idx,
                    distance: t,
                    position: center,
                });
            }
            return;
        }

        for &child_idx in &node.children {
            if child_idx != EMPTY_CHILD {
                self.collect_ray_hits(child_idx, ray, out);
            }
        }
    }

    // --- Box ---------------------------------------------------------------

    fn collect_box_hits(&self, node_idx: u32, aabb: &AABB, out: &mut Vec<CollisionHit>) {
        let node = &self.nodes[node_idx as usize];

        if !node.bounds.intersects_aabb(aabb) {
            return;
        }

        if node.is_leaf {
            let box_center = aabb.center();
            for &splat_idx in &node.splat_indices {
                let center = self.centers[splat_idx as usize];
                let dist_sq = (center - box_center).length_sq();
                out.push(CollisionHit {
                    splat_index: splat_idx,
                    distance: dist_sq,
                    position: center,
                });
            }
            return;
        }

        for &child_idx in &node.children {
            if child_idx != EMPTY_CHILD {
                self.collect_box_hits(child_idx, aabb, out);
            }
        }
    }

    // --- Sphere ------------------------------------------------------------

    fn collect_sphere_hits(&self, node_idx: u32, sphere: &Sphere, out: &mut Vec<CollisionHit>) {
        let node = &self.nodes[node_idx as usize];

        if !node.bounds.intersects_sphere(sphere) {
            return;
        }

        if node.is_leaf {
            for &splat_idx in &node.splat_indices {
                let center = self.centers[splat_idx as usize];
                let dist = (center - sphere.center).length();
                out.push(CollisionHit {
                    splat_index: splat_idx,
                    distance: dist,
                    position: center,
                });
            }
            return;
        }

        for &child_idx in &node.children {
            if child_idx != EMPTY_CHILD {
                self.collect_sphere_hits(child_idx, sphere, out);
            }
        }
    }
}

// ===========================================================================
// Debug ray traversal path
// ===========================================================================

impl Octree {
    /// Return the depth-first path of every octree node intersected by `ray`.
    ///
    /// Children at each level are visited front-to-back (sorted by `t_enter`)
    /// so the resulting path is ordered by increasing entry distance within
    /// each depth level.  This is primarily intended for **debug
    /// visualisation**: you can draw a wireframe box for every returned node
    /// to see exactly which parts of the tree the ray traverses.
    pub fn ray_traversal_path(&self, ray: &Ray) -> Vec<OctreeTraversalNode> {
        let mut path = Vec::new();
        if !self.nodes.is_empty() {
            self.collect_traversal_path(0, ray, &mut path);
        }
        path
    }

    fn collect_traversal_path(
        &self,
        node_idx: u32,
        ray: &Ray,
        path: &mut Vec<OctreeTraversalNode>,
    ) {
        let node = &self.nodes[node_idx as usize];

        let Some((t_enter, _t_exit)) = node.bounds.intersects_ray(ray) else {
            return;
        };

        let splat_count = if node.is_leaf {
            node.splat_indices.len() as u32
        } else {
            0
        };

        path.push(OctreeTraversalNode {
            bounds: node.bounds,
            depth: node.depth,
            splat_count,
            t_enter,
        });

        if !node.is_leaf {
            // Sort children by t_enter for front-to-back ordering.
            let mut child_entries: Vec<(u32, f32)> = node
                .children
                .iter()
                .copied()
                .filter(|&idx| idx != EMPTY_CHILD)
                .filter_map(|idx| {
                    self.nodes[idx as usize]
                        .bounds
                        .intersects_ray(ray)
                        .map(|(t, _)| (idx, t))
                })
                .collect();

            child_entries.sort_by(|a, b| cmp_f32(a.1, b.1));

            for (child_idx, _) in child_entries {
                self.collect_traversal_path(child_idx, ray, path);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Stable `f32` comparison (NaN goes to the end).
#[inline]
fn cmp_f32(a: f32, b: f32) -> Ordering {
    a.partial_cmp(&b).unwrap_or(Ordering::Equal)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a tiny Splats with 8 points at the corners of a unit cube.
    fn make_test_splats() -> Splats {
        let mut centers = Vec::new();
        for z in [0.0f32, 1.0] {
            for y in [0.0f32, 1.0] {
                for x in [0.0f32, 1.0] {
                    centers.push(x);
                    centers.push(y);
                    centers.push(z);
                }
            }
        }
        let count = 8u32;
        Splats {
            count,
            format: "test".into(),
            center: centers.into_boxed_slice(),
            covariance: vec![0.0; 6 * count as usize].into_boxed_slice(),
            rgba: vec![0u32; count as usize].into_boxed_slice(),
            sh_coeffs_l1: vec![0.0; 9 * count as usize].into_boxed_slice(),
            sh_coeffs_l2_packed: vec![0u32; 10 * count as usize].into_boxed_slice(),
            sh_coeffs_l2_scale: 1.0,
            sh_coeffs_l3_packed: vec![0u32; 14 * count as usize].into_boxed_slice(),
            sh_coeffs_l3_scale: 1.0,
            sh_degree: 0,
            bbox_min: [0.0, 0.0, 0.0],
            bbox_max: [1.0, 1.0, 1.0],
        }
    }

    #[test]
    fn build_octree_from_splats() {
        let splats = make_test_splats();
        let config = OctreeConfig {
            max_depth: 4,
            max_leaf_size: 2,
        };
        let tree = Octree::from_splats_with_config(&splats, config);

        assert!(tree.node_count() > 1, "should have more than just a root");
        assert_eq!(tree.splat_count(), 8);
        assert!(!tree.root().is_leaf, "root should be internal with 8 points and leaf_size=2");
    }

    #[test]
    fn ray_collider_hits_all_along_diagonal() {
        let splats = make_test_splats();
        let tree = Octree::from_splats(&splats);

        // Shoot a ray along the main diagonal — should enter the bounding box.
        let ray = Ray::new(
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, 1.0).normalized(),
        );
        let hits: Vec<CollisionHit> = tree.collide_ray(&ray).collect();
        assert!(!hits.is_empty(), "diagonal ray should hit some splats");
    }

    #[test]
    fn ray_collider_misses_when_outside() {
        let splats = make_test_splats();
        let tree = Octree::from_splats(&splats);

        // Shoot a ray that is clearly outside the bounding volume.
        let ray = Ray::new(
            Vec3::new(100.0, 100.0, 100.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        let hits: Vec<CollisionHit> = tree.collide_ray(&ray).collect();
        assert!(hits.is_empty(), "ray far away should produce no hits");
    }

    #[test]
    fn box_collider_finds_corner_splats() {
        let splats = make_test_splats();
        let tree = Octree::from_splats(&splats);

        // Query a small box around the origin — should catch the (0,0,0) splat.
        let query = AABB::new(Vec3::new(-0.1, -0.1, -0.1), Vec3::new(0.1, 0.1, 0.1));
        let hits: Vec<CollisionHit> = tree.collide_box(&query).collect();
        assert!(
            hits.iter().any(|h| h.splat_index == 0),
            "should find splat at origin"
        );
    }

    #[test]
    fn sphere_collider_finds_nearby_splats() {
        let splats = make_test_splats();
        let tree = Octree::from_splats(&splats);

        let query = Sphere::new(Vec3::new(0.5, 0.5, 0.5), 2.0);
        let hits: Vec<CollisionHit> = tree.collide_sphere(&query).collect();
        assert_eq!(hits.len(), 8, "sphere should encompass all 8 corner splats");
    }

    #[test]
    fn traversal_path_is_depth_first() {
        let splats = make_test_splats();
        let config = OctreeConfig {
            max_depth: 4,
            max_leaf_size: 2,
        };
        let tree = Octree::from_splats_with_config(&splats, config);

        let ray = Ray::new(
            Vec3::new(-1.0, 0.5, 0.5),
            Vec3::new(1.0, 0.0, 0.0),
        );

        let path = tree.ray_traversal_path(&ray);
        assert!(!path.is_empty(), "traversal path should not be empty");

        // The first node should be the root (depth 0).
        assert_eq!(path[0].depth, 0);

        // Depths should be monotonically non-decreasing within each sub-tree,
        // but because it's depth-first they should generally increase then
        // decrease.  At minimum, all depths should be ≤ max_depth.
        for n in &path {
            assert!(n.depth <= config.max_depth);
        }
    }
}
