use crate::utility::prelude::*;
use alloc::vec::Vec;

/// A public API for using optimization methods.
///
/// Users can simply obtain their solution and see the result.
///
/// + The method is a type that implemented [`Algorithm`].
/// + The objective function is a type that implement [`ObjFunc`].
/// + A basic algorithm data is hold by [`Context`].
///
/// The builder of this type can infer the algorithm by [`Setting::Algorithm`].
///
/// ```
/// use metaheuristics_nature::{Rga, Solver};
/// # use metaheuristics_nature::tests::TestObj as MyFunc;
///
/// // Build and run the solver
/// let s = Solver::build(Rga::default())
///     .task(|ctx| ctx.gen == 20)
///     .solve(MyFunc::new());
/// // Get the result from objective function
/// let ans = s.result();
/// // Get the optimized XY value of your function
/// let xs = s.best_parameters();
/// let y = s.best_fitness();
/// // Get the history reports
/// let report = s.report();
/// ```
pub struct Solver<F: ObjFunc, R> {
    ctx: Context<F>,
    report: Vec<R>,
}

impl<F: ObjFunc, R> Solver<F, R> {
    pub(crate) fn new(ctx: Context<F>, report: Vec<R>) -> Self {
        Self { ctx, report }
    }

    /// Get the reference of the objective function.
    ///
    /// It's useful when you need to get the preprocessed data from the initialization process,
    /// which is stored in the objective function.
    pub fn func(&self) -> &F {
        &self.ctx.func
    }

    /// Get the history report returned by record function.
    pub fn report(&self) -> &[R] {
        &self.report
    }

    /// Get the best parameters.
    pub fn best_parameters(&self) -> &[f64] {
        self.ctx.best.as_slice().unwrap()
    }

    /// Get the best fitness.
    pub fn best_fitness(&self) -> F::Fitness {
        self.ctx.best_f.clone()
    }

    /// Get the result of the objective function.
    pub fn result(&self) -> F::Result {
        self.func().result(self.best_parameters())
    }

    /// Seed of the random number generator.
    pub fn seed(&self) -> u128 {
        self.ctx.rng.seed()
    }
}
