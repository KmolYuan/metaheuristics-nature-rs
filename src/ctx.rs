use crate::prelude::*;
use alloc::vec::Vec;

pub(crate) type BestCon<F> = <F as Fitness>::Best<F>;

/// A basic context type of the algorithms.
///
/// This type provides a shared dataset if you want to implement a new method.
/// The fields maybe expanded in the future, so it marked as non-exhaustive.
///
/// # View the Progress
///
/// You can view the progress from the [`SolverBuilder::task()`] and
/// [`SolverBuilder::callback()`].
///
/// + `ctx.gen` - Get generation number.
/// + `ctx.pop_num()` - Get population number.
/// + `ctx.best.get_eval()` - Get the current best evaluation value.
/// + `ctx.best.get_xs()` - Get the current best variables.
///
/// # Implement an Algorithm
///
/// Do everything you want to do with the context. Please see [`Algorithm`] for
/// the implementation.
#[non_exhaustive]
pub struct Ctx<F: ObjFunc> {
    /// Best container
    pub best: BestCon<F::Ys>,
    /// Current variables of all individuals
    pub pool: Vec<Vec<f64>>,
    /// Current fitness values of all individuals
    pub pool_y: Vec<F::Ys>,
    /// Objective function object
    pub func: F,
    /// Generation (iteration) number
    pub gen: u64,
}

impl<F: ObjFunc> Ctx<F> {
    pub(crate) fn from_parts(
        func: F,
        limit: usize,
        pool: Vec<Vec<f64>>,
        pool_y: Vec<F::Ys>,
    ) -> Self {
        let mut best = BestCon::<F::Ys>::from_limit(limit);
        best.update_all(&pool, &pool_y);
        Self { best, pool, pool_y, func, gen: 0 }
    }

    pub(crate) fn from_pool(func: F, limit: usize, pool: Vec<Vec<f64>>) -> Self {
        #[cfg(not(feature = "rayon"))]
        let iter = pool.iter();
        #[cfg(feature = "rayon")]
        let iter = pool.par_iter();
        let pool_y = iter.map(|xs| func.fitness(xs)).collect();
        Self::from_parts(func, limit, pool, pool_y)
    }

    /// Get population number.
    #[inline]
    pub fn pop_num(&self) -> usize {
        self.pool.len()
    }

    /// Assign the index from source.
    pub fn set_from(&mut self, i: usize, xs: Vec<f64>, ys: F::Ys) {
        self.pool[i] = xs;
        self.pool_y[i] = ys;
    }

    /// Find the best, and set it globally.
    pub fn find_best(&mut self) {
        self.best.update_all(&self.pool, &self.pool_y);
    }
}

impl<F: ObjFunc> core::ops::Deref for Ctx<F> {
    type Target = F;
    fn deref(&self) -> &Self::Target {
        &self.func
    }
}
