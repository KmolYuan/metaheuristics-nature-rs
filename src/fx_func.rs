use crate::utility::prelude::*;
use alloc::boxed::Box;

/// A quick interface help to create objective function from a callable object.
///
/// ```
/// use metaheuristics_nature::{Fx, Rga, Solver};
///
/// let bound = [[-50., 50.]; 4];
/// let f = Fx::new(&bound, |[a, b, c, d]| a * a + 8. * b * b + c * c + d * d);
/// let s = Solver::build(Rga::default())
///     .task(|ctx| ctx.gen == 20)
///     .solve(f)
///     .unwrap();
/// ```
pub struct Fx<'f, 'b, R: Fitness, const DIM: usize> {
    func: Box<dyn Fn([f64; DIM]) -> R + Sync + Send + 'f>,
    bound: &'b [[f64; 2]; DIM],
}

impl<'f, 'b, R: Fitness, const DIM: usize> Fx<'f, 'b, R, DIM> {
    /// Create objective function from a callable object.
    pub fn new<F>(bound: &'b [[f64; 2]; DIM], func: F) -> Self
    where
        F: Fn([f64; DIM]) -> R + Sync + Send + 'f,
    {
        Self { bound, func: Box::new(func) }
    }
}

impl<R: Fitness, const DIM: usize> Bounded for Fx<'_, '_, R, DIM> {
    fn bound(&self) -> &[[f64; 2]] {
        self.bound
    }
}

impl<R: Fitness, const DIM: usize> ObjFunc for Fx<'_, '_, R, DIM> {
    type Fitness = R;

    fn fitness(&self, xs: &[f64]) -> Self::Fitness {
        (self.func)(xs.try_into().unwrap())
    }
}
