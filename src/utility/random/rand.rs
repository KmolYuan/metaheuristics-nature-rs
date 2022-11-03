use core::ops::{Range, Sub};
use num_traits::AsPrimitive;

pub trait Rand {
    type Result;
    fn rand(self, base: f64) -> Self::Result;
}

impl<N> Rand for Range<N>
where
    N: AsPrimitive<f64> + Sub<Output = N>,
    f64: AsPrimitive<N>,
{
    type Result = N;
    fn rand(self, base: f64) -> Self::Result {
        (base * (self.end - self.start).as_() + self.start.as_()).as_()
    }
}
