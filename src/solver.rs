use crate::utility::prelude::*;

/// A public API for using optimization methods.
///
/// Users can simply obtain their solution and see the result.
///
/// + The method is a type that implemented [`Algorithm`].
/// + The objective function is a type that implement [`ObjFunc`].
/// + A basic algorithm data is hold by [`Ctx`].
///
/// The builder of this type can infer the algorithm by [`Setting::Algorithm`].
///
/// ```
/// use metaheuristics_nature::{Rga, Solver};
/// # use metaheuristics_nature::tests::TestObj as MyFunc;
///
/// let mut report = Vec::with_capacity(20);
///
/// // Build and run the solver
/// let s = Solver::build(Rga::default(), MyFunc::new())
///     .task(|ctx| ctx.gen == 20)
///     .callback(|ctx| report.push(ctx.best_f))
///     .solve()
///     .unwrap();
/// // Get the result from objective function
/// let ans = s.result();
/// // Get the optimized XY value of your function
/// let xs = s.best_parameters();
/// let y = s.best_fitness();
/// // Get the history reports
/// let y2 = report[2];
/// ```
#[must_use = "please call `Solver::best_parameters()` or other methods to get the answer"]
pub struct Solver<F: ObjFunc> {
    ctx: Ctx<F>,
}

impl<F: ObjFunc> Solver<F> {
    pub(crate) fn new(ctx: Ctx<F>) -> Self {
        Self { ctx }
    }

    /// Get the reference of the objective function.
    ///
    /// It's useful when you need to get the preprocessed data from the
    /// initialization process, which is stored in the objective function.
    pub fn func(&self) -> &F {
        &self.ctx.func
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
    pub fn result(&self) -> F::Product
    where
        F: ObjFactory,
    {
        self.ctx.result()
    }

    /// Seed of the random number generator.
    pub fn seed(&self) -> u128 {
        self.ctx.rng.seed()
    }

    /// Get the pool from the last status.
    pub fn pool(&self) -> ArrayView2<f64> {
        self.ctx.pool.view()
    }
}
