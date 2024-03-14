use crate::utility::prelude::*;
pub use crate::utility::{gaussian_pool, uniform_pool, Pool};

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
/// Please use [`Solver::build()`] method to start a task.
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
///     .callback(|ctx| report.push(ctx.best_f.clone()))
///     .solve();
/// // Get the result from objective function
/// let ans = s.as_result();
/// // Get the optimized XY value of your function
/// let xs = s.best_parameters();
/// let (err, ans) = s.into_err_result();
/// // Get the history reports
/// let y2 = &report[2];
/// ```
#[must_use = "please call `Solver::best_parameters()` or other methods to get the answer"]
pub struct Solver<F: ObjFunc> {
    ctx: Ctx<F>,
    seed: Seed,
}

impl<F: ObjFunc> Solver<F> {
    pub(crate) fn new(ctx: Ctx<F>, seed: Seed) -> Self {
        Self { ctx, seed }
    }

    /// Get the reference of the objective function.
    ///
    /// It's useful when you need to get the preprocessed data from the
    /// initialization process, which is stored in the objective function.
    pub fn func(&self) -> &F {
        &self.ctx.func
    }

    /// Get the best parameters.
    ///
    /// See also [`Solver::as_best_fitness()`], [`Solver::as_result()`].
    pub fn best_parameters(&self) -> &[f64] {
        &self.ctx.best
    }

    /// Get the best fitness.
    ///
    /// See also [`Solver::as_best_fitness()`].
    pub fn best_fitness(&self) -> F::Fitness
    where
        F::Fitness: Copy,
    {
        self.ctx.best_f
    }

    /// Get the reference to the best fitness.
    ///
    /// See also [`Solver::best_fitness()`].
    pub fn as_best_fitness(&self) -> &F::Fitness {
        &self.ctx.best_f
    }

    /// Get the result of the objective function.
    ///
    /// See also [`Solver::into_result()`], [`Solver::into_err_result()`].
    pub fn as_result<P, Fit>(&self) -> &P
    where
        Fit: Fitness,
        F: ObjFunc<Fitness = Product<P, Fit>>,
    {
        self.ctx.best_f.as_result()
    }

    /// Unwrap and get the final result.
    ///
    /// See also [`Solver::as_result()`], [`Solver::into_err_result()`].
    pub fn into_result<P, Fit>(self) -> P
    where
        Fit: Fitness,
        F: ObjFunc<Fitness = Product<P, Fit>>,
    {
        self.ctx.best_f.into_result()
    }

    /// Unwrap and get the final error and result.
    ///
    /// See also [`Solver::as_result()`], [`Solver::into_result()`].
    pub fn into_err_result<P, Fit>(self) -> (Fit, P)
    where
        Fit: Fitness,
        F: ObjFunc<Fitness = Product<P, Fit>>,
    {
        self.ctx.best_f.into_inner()
    }

    /// Seed of the random number generator.
    pub fn seed(&self) -> Seed {
        self.seed
    }

    /// Get the pool from the last status.
    pub fn pool(&self) -> &[Vec<f64>] {
        &self.ctx.pool
    }
}
