use crate::prelude::*;
use alloc::boxed::Box;

/// A quick interface help to create objective function from a callable object.
///
/// See also [`ObjFunc`] for implementing the full definitions.
///
/// ```
/// use metaheuristics_nature::{Fx, Rga, Solver};
///
/// let bound = [[-50., 50.]; 4];
/// let f = Fx::new(&bound, |&[a, b, c, d]| a * a + 8. * b * b + c * c + d * d);
/// let s = Solver::build(Rga::default(), f)
///     .seed(0)
///     .task(|ctx| ctx.gen == 20)
///     .solve();
/// ```
pub struct Fx<'b, 'f, Y: Fitness, const DIM: usize> {
    bound: &'b [[f64; 2]; DIM],
    #[allow(clippy::type_complexity)]
    func: Box<dyn Fn(&[f64; DIM]) -> Y + Sync + Send + 'f>,
}

impl<'b, 'f, Y: Fitness, const DIM: usize> Fx<'b, 'f, Y, DIM> {
    /// Create objective function from a callable object.
    pub fn new<F>(bound: &'b [[f64; 2]; DIM], func: F) -> Self
    where
        F: Fn(&[f64; DIM]) -> Y + Sync + Send + 'f,
    {
        Self { func: Box::new(func), bound }
    }
}

impl<Y: Fitness, const DIM: usize> Bounded for Fx<'_, '_, Y, DIM> {
    #[inline]
    fn bound(&self) -> &[[f64; 2]] {
        self.bound
    }
}

impl<Y: Fitness, const DIM: usize> ObjFunc for Fx<'_, '_, Y, DIM> {
    type Ys = Y;
    fn fitness(&self, xs: &[f64]) -> Self::Ys {
        (self.func)(xs.try_into().unwrap_or_else(|_| unreachable!()))
    }
}
