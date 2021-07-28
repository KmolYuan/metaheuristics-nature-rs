//! The random function for building algorithm.
use rand::{distributions::uniform::SampleUniform, thread_rng, Rng};

/// Generate random boolean by positive factor.
#[inline]
pub fn maybe(v: f64) -> bool {
    thread_rng().gen_bool(v)
}

/// Generate random values between [0., 1.).
#[inline]
pub fn rand() -> f64 {
    rand_rng(0., 1.)
}

/// Generate random values by range.
#[inline]
pub fn rand_rng<T>(lb: T, ub: T) -> T
where
    T: SampleUniform + PartialOrd,
{
    thread_rng().gen_range(lb..ub)
}
