use crate::prelude::*;
use alloc::boxed::Box;

/// A quick interface help to create objective function from a callable object.
///
/// ```
/// use metaheuristics_nature::{Fx, Rga, Solver};
///
/// let bound = [[-50., 50.]; 4];
/// let f = Fx::new(&bound, |[a, b, c, d]| a * a + 8. * b * b + c * c + d * d);
/// let s = Solver::build(Rga::default(), f)
///     .task(|ctx| ctx.gen == 20)
///     .solve();
/// ```
pub struct Fx<'b, 'f, R: Fitness, const DIM: usize> {
    bound: &'b [[f64; 2]; DIM],
    func: Box<dyn Fn([f64; DIM]) -> R + Sync + Send + 'f>,
}

impl<'b, 'f, R: Fitness, const DIM: usize> Fx<'b, 'f, R, DIM> {
    /// Create objective function from a callable object.
    pub fn new<F>(bound: &'b [[f64; 2]; DIM], func: F) -> Self
    where
        F: Fn([f64; DIM]) -> R + Sync + Send + 'f,
    {
        Self { func: Box::new(func), bound }
    }
}

impl<R: Fitness, const DIM: usize> Bounded for Fx<'_, '_, R, DIM> {
    fn bound(&self) -> &[[f64; 2]] {
        self.bound
    }
}

impl<R: Fitness, const DIM: usize> ObjFunc for Fx<'_, '_, R, DIM> {
    type Ys = R;

    fn fitness(&self, xs: &[f64]) -> Self::Ys {
        (self.func)(xs.try_into().unwrap())
    }
}
