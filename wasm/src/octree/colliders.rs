//! Collision query traits and their result type.
//!
//! Three traits — [`RayCollider`], [`BoxCollider`], [`SphereCollider`] — define
//! a uniform interface for spatial queries.  Any data structure (e.g. [`super::Octree`])
//! that implements these traits returns an iterator of [`CollisionHit`] items, each
//! carrying a splat index and a distance metric appropriate to the query kind.

use super::geometry::{Ray, Vec3, AABB, Sphere};

// ---------------------------------------------------------------------------
// CollisionHit
// ---------------------------------------------------------------------------

/// A single hit produced by a collision query.
#[derive(Debug, Clone, Copy)]
pub struct CollisionHit {
    /// Index of the splat in the original data array.
    pub splat_index: u32,

    /// Contextual distance metric whose semantics depend on the query kind:
    ///
    /// | Query                          | Meaning                                                       |
    /// |--------------------------------|---------------------------------------------------------------|
    /// | [`RayCollider::collide_ray`]   | Parametric `t` — projection of the splat centre onto the ray. |
    /// | [`BoxCollider::collide_box`]   | Squared distance from the AABB centre to the splat centre.    |
    /// | [`SphereCollider::collide_sphere`] | Distance from the sphere centre to the splat centre.      |
    pub distance: f32,

    /// Centre position of the hit splat (copied for convenience).
    pub position: Vec3,
}

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

/// Query a spatial structure with a [`Ray`].
///
/// Returns an iterator over all splats whose containing leaf node is
/// intersected by the ray, sorted by ascending `t` (parametric distance along
/// the ray).
pub trait RayCollider {
    fn collide_ray(&self, ray: &Ray) -> impl Iterator<Item = CollisionHit>;
}

/// Query a spatial structure with an axis-aligned bounding box ([`AABB`]).
///
/// Returns an iterator over all splats whose containing leaf node intersects
/// the query box.
pub trait BoxCollider {
    fn collide_box(&self, aabb: &AABB) -> impl Iterator<Item = CollisionHit>;
}

/// Query a spatial structure with a [`Sphere`].
///
/// Returns an iterator over all splats whose containing leaf node intersects
/// the query sphere.
pub trait SphereCollider {
    fn collide_sphere(&self, sphere: &Sphere) -> impl Iterator<Item = CollisionHit>;
}
