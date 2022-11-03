use super::Rand64;
use core::ops::{Range, Sub};
use num_traits::AsPrimitive;

pub trait Rand {
    type Result;
    fn rand(self, rng: &mut Rand64) -> Self::Result;
}

// TODO: Need to Optimize
impl<N> Rand for Range<N>
where
    N: AsPrimitive<f64> + Sub<Output = N>,
    f64: AsPrimitive<N>,
{
    type Result = N;
    fn rand(self, rng: &mut Rand64) -> Self::Result {
        (rng.rand_float() * (self.end - self.start).as_() + self.start.as_()).as_()
    }
}
