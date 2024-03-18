use crate::prelude::*;
use alloc::vec::Vec;

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
///     .seed(0)
///     .task(|ctx| ctx.gen == 20)
///     .callback(|ctx| report.push(ctx.best.get_eval()))
///     .solve();
/// // Get the optimized XY value of your function
/// let (xs, ys) = s.as_best();
/// // If `ys` is a `Product` type
/// let err = ys.fitness();
/// let result = ys.as_result();
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

    /// Get the reference of the best set.
    ///
    /// Use [`Solver::as_best()`] to get the best parameters and the fitness
    /// value directly.
    pub fn as_best_set(&self) -> &BestCon<F::Ys> {
        &self.ctx.best
    }

    /// Get the reference of the best parameters and the fitness value.
    pub fn as_best(&self) -> (&[f64], &F::Ys) {
        self.ctx.best.as_result()
    }

    /// Get the reference of the best fitness value.
    pub fn as_best_xs(&self) -> &[f64] {
        self.as_best().0
    }

    /// Get the reference of the best fitness value.
    pub fn as_best_fit(&self) -> &F::Ys {
        self.as_best().1
    }

    /// Get the final best fitness value.
    pub fn get_best_eval(&self) -> <F::Ys as Fitness>::Eval {
        self.as_best_fit().eval()
    }

    /// Get the final best element.
    pub fn into_result<P, Fit: Fitness>(self) -> P
    where
        F: ObjFunc<Ys = Product<Fit, P>>,
        P: MaybeParallel + Clone + 'static,
    {
        self.ctx.best.into_result_fit().into_result()
    }

    /// Get the fitness value and the final result.
    pub fn into_err_result<P, Fit: Fitness>(self) -> (Fit, P)
    where
        F: ObjFunc<Ys = Product<Fit, P>>,
        P: MaybeParallel + Clone + 'static,
    {
        self.ctx.best.into_result_fit().into_err_result()
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
