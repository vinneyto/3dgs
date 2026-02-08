//! Octree spatial index for gaussian splats.
//!
//! # Overview
//!
//! This module provides an [`Octree`] data structure that spatially indexes the
//! centre positions of gaussian splats.  Three collision traits —
//! [`RayCollider`], [`BoxCollider`], and [`SphereCollider`] — define a uniform
//! query interface and are implemented by `Octree`.
//!
//! ## Quick start
//!
//! ```rust,ignore
//! use rust_wasm::octree::*;
//! use rust_wasm::Splats;
//!
//! let splats: Splats = /* parse from PLY / SOGS … */;
//! let tree = Octree::from_splats(&splats);
//!
//! // Ray query
//! let ray = Ray::new(Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
//! for hit in tree.collide_ray(&ray) {
//!     println!("splat {} at t={}", hit.splat_index, hit.distance);
//! }
//!
//! // Debug visualisation — draw the path of nodes traversed by the ray.
//! for node in tree.ray_traversal_path(&ray) {
//!     println!("depth {} bbox {:?}", node.depth, node.bounds);
//! }
//! ```

pub mod colliders;
pub mod geometry;
pub mod tree;

// Re-export the most commonly used types at crate level.
pub use colliders::{BoxCollider, CollisionHit, RayCollider, SphereCollider};
pub use geometry::{Ray, Sphere, Vec3, AABB};
pub use tree::{Octree, OctreeConfig, OctreeNode, OctreeTraversalNode, EMPTY_CHILD};
