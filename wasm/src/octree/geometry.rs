//! Geometric primitives used by the octree: [`Vec3`], [`Ray`], [`AABB`], [`Sphere`].

use std::ops::{Add, Mul, Neg, Sub};

// ---------------------------------------------------------------------------
// Vec3
// ---------------------------------------------------------------------------

/// Minimal 3-component vector for spatial math.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub const fn zero() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    #[inline]
    pub const fn splat(v: f32) -> Self {
        Self { x: v, y: v, z: v }
    }

    #[inline]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    #[inline]
    pub fn length_sq(self) -> f32 {
        self.dot(self)
    }

    #[inline]
    pub fn length(self) -> f32 {
        self.length_sq().sqrt()
    }

    #[inline]
    pub fn normalized(self) -> Self {
        let len = self.length();
        if len > 0.0 {
            self * (1.0 / len)
        } else {
            Self::zero()
        }
    }

    #[inline]
    pub fn min_components(self, rhs: Self) -> Self {
        Self {
            x: self.x.min(rhs.x),
            y: self.y.min(rhs.y),
            z: self.z.min(rhs.z),
        }
    }

    #[inline]
    pub fn max_components(self, rhs: Self) -> Self {
        Self {
            x: self.x.max(rhs.x),
            y: self.y.max(rhs.y),
            z: self.z.max(rhs.z),
        }
    }

    /// Per-component clamp.
    #[inline]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        Self {
            x: self.x.clamp(lo.x, hi.x),
            y: self.y.clamp(lo.y, hi.y),
            z: self.z.clamp(lo.z, hi.z),
        }
    }

    /// Access by axis index (0=x, 1=y, 2=z).
    #[inline]
    pub fn axis(&self, i: usize) -> f32 {
        match i {
            0 => self.x,
            1 => self.y,
            _ => self.z,
        }
    }
}

// --- Operator impls --------------------------------------------------------

impl Add for Vec3 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Sub for Vec3 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Mul<f32> for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f32) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl Neg for Vec3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

// ---------------------------------------------------------------------------
// Ray
// ---------------------------------------------------------------------------

/// A ray defined by an origin and a direction (not necessarily normalised).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    #[inline]
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self { origin, direction }
    }

    /// Point along the ray at parameter `t`: `origin + direction * t`.
    #[inline]
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }
}

// ---------------------------------------------------------------------------
// AABB (Axis-Aligned Bounding Box)
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box defined by its minimum and maximum corners.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    #[inline]
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    #[inline]
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    #[inline]
    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    #[inline]
    pub fn contains_point(&self, p: Vec3) -> bool {
        p.x >= self.min.x
            && p.x <= self.max.x
            && p.y >= self.min.y
            && p.y <= self.max.y
            && p.z >= self.min.z
            && p.z <= self.max.z
    }

    /// Test overlap between two AABBs.
    #[inline]
    pub fn intersects_aabb(&self, other: &AABB) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    /// Test intersection between this AABB and a [`Sphere`].
    pub fn intersects_sphere(&self, sphere: &Sphere) -> bool {
        let closest = sphere.center.clamp(self.min, self.max);
        (closest - sphere.center).length_sq() <= sphere.radius * sphere.radius
    }

    /// Robust slab-method ray–AABB intersection.
    ///
    /// Returns `Some((t_enter, t_exit))` when the ray hits the box, `None` otherwise.
    /// `t_enter` is clamped to `≥ 0` so that intersections starting behind the
    /// origin are still reported as long as the ray exits in front of the origin.
    pub fn intersects_ray(&self, ray: &Ray) -> Option<(f32, f32)> {
        let mut t_min = f32::NEG_INFINITY;
        let mut t_max = f32::INFINITY;

        for axis in 0..3 {
            let origin = ray.origin.axis(axis);
            let dir = ray.direction.axis(axis);
            let min_v = self.min.axis(axis);
            let max_v = self.max.axis(axis);

            if dir.abs() < 1e-30 {
                // Ray is parallel to this slab.
                if origin < min_v || origin > max_v {
                    return None;
                }
                // No constraint from this axis — leave t_min/t_max unchanged.
            } else {
                let inv_d = 1.0 / dir;
                let mut t1 = (min_v - origin) * inv_d;
                let mut t2 = (max_v - origin) * inv_d;
                if t1 > t2 {
                    std::mem::swap(&mut t1, &mut t2);
                }
                t_min = t_min.max(t1);
                t_max = t_max.min(t2);
                if t_min > t_max {
                    return None;
                }
            }
        }

        if t_max < 0.0 {
            None
        } else {
            Some((t_min.max(0.0), t_max))
        }
    }

    /// Return the sub-AABB for the given octant index (`0..8`).
    ///
    /// Octant bit layout: `x = bit 0`, `y = bit 1`, `z = bit 2`.
    pub fn child_octant(&self, index: u8) -> AABB {
        let mid = self.center();

        let min_x = if index & 1 != 0 { mid.x } else { self.min.x };
        let max_x = if index & 1 != 0 { self.max.x } else { mid.x };
        let min_y = if index & 2 != 0 { mid.y } else { self.min.y };
        let max_y = if index & 2 != 0 { self.max.y } else { mid.y };
        let min_z = if index & 4 != 0 { mid.z } else { self.min.z };
        let max_z = if index & 4 != 0 { self.max.z } else { mid.z };

        AABB::new(
            Vec3::new(min_x, min_y, min_z),
            Vec3::new(max_x, max_y, max_z),
        )
    }
}

// ---------------------------------------------------------------------------
// Sphere
// ---------------------------------------------------------------------------

/// A sphere defined by centre and radius.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f32,
}

impl Sphere {
    #[inline]
    pub fn new(center: Vec3, radius: f32) -> Self {
        Self { center, radius }
    }

    /// Test whether a point lies inside (or on the surface of) this sphere.
    #[inline]
    pub fn contains_point(&self, p: Vec3) -> bool {
        (p - self.center).length_sq() <= self.radius * self.radius
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec3_basics() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        let sum = a + b;
        assert_eq!(sum, Vec3::new(5.0, 7.0, 9.0));
        assert!((a.dot(b) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn aabb_contains_point() {
        let bb = AABB::new(Vec3::zero(), Vec3::new(2.0, 2.0, 2.0));
        assert!(bb.contains_point(Vec3::new(1.0, 1.0, 1.0)));
        assert!(!bb.contains_point(Vec3::new(3.0, 1.0, 1.0)));
    }

    #[test]
    fn aabb_ray_hit() {
        let bb = AABB::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let ray = Ray::new(Vec3::new(0.0, 0.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        let hit = bb.intersects_ray(&ray);
        assert!(hit.is_some());
        let (t_enter, t_exit) = hit.unwrap();
        assert!((t_enter - 4.0).abs() < 1e-5);
        assert!((t_exit - 6.0).abs() < 1e-5);
    }

    #[test]
    fn aabb_ray_miss() {
        let bb = AABB::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let ray = Ray::new(Vec3::new(5.0, 5.0, 0.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(bb.intersects_ray(&ray).is_none());
    }

    #[test]
    fn aabb_sphere_intersection() {
        let bb = AABB::new(Vec3::zero(), Vec3::new(2.0, 2.0, 2.0));
        let inside = Sphere::new(Vec3::new(1.0, 1.0, 1.0), 0.5);
        assert!(bb.intersects_sphere(&inside));

        let touching = Sphere::new(Vec3::new(3.0, 1.0, 1.0), 1.0);
        assert!(bb.intersects_sphere(&touching));

        let outside = Sphere::new(Vec3::new(5.0, 5.0, 5.0), 0.5);
        assert!(!bb.intersects_sphere(&outside));
    }

    #[test]
    fn child_octant_covers_parent() {
        let parent = AABB::new(Vec3::zero(), Vec3::new(4.0, 4.0, 4.0));
        let mid = parent.center();
        assert_eq!(mid, Vec3::new(2.0, 2.0, 2.0));

        // Octant 0 = low x, low y, low z
        let c0 = parent.child_octant(0);
        assert_eq!(c0.min, parent.min);
        assert_eq!(c0.max, mid);

        // Octant 7 = high x, high y, high z
        let c7 = parent.child_octant(7);
        assert_eq!(c7.min, mid);
        assert_eq!(c7.max, parent.max);
    }

    #[test]
    fn ray_parallel_to_axis_inside() {
        // Ray along X, origin inside box
        let bb = AABB::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let hit = bb.intersects_ray(&ray);
        assert!(hit.is_some());
    }
}
