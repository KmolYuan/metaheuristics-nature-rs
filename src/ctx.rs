use crate::prelude::*;
use alloc::vec::Vec;

pub(crate) type BestCon<F> = <F as Fitness>::Best<F>;

/// A basic context type of the algorithms.
///
/// This type provides a shared dataset if you want to implement a new method.
///
/// Please see [`Algorithm`] for the implementation.
#[non_exhaustive]
pub struct Ctx<F: ObjFunc> {
    /// Best container
    pub best: BestCon<F::Fitness>,
    /// Current variables of all individuals
    pub pool: Vec<Vec<f64>>,
    /// Current fitness values of all individuals
    pub pool_y: Vec<F::Fitness>,
    /// Generation
    pub gen: u64,
    /// Objective function object
    pub func: F,
}

impl<F: ObjFunc> Ctx<F> {
    pub(crate) fn from_parts(
        func: F,
        mut best: BestCon<F::Fitness>,
        pool: Vec<Vec<f64>>,
        mut pool_y: Vec<F::Fitness>,
    ) -> Self {
        best.update_all(&pool, &pool_y);
        pool_y.iter_mut().for_each(|ys| ys.mark_not_best());
        Self { best, pool, pool_y, gen: 0, func }
    }

    pub(crate) fn from_pool(func: F, best: BestCon<F::Fitness>, pool: Vec<Vec<f64>>) -> Self {
        #[cfg(not(feature = "rayon"))]
        let iter = pool.iter();
        #[cfg(feature = "rayon")]
        let iter = pool.par_iter();
        let pool_y = iter.map(|xs| func.fitness(xs)).collect();
        Self::from_parts(func, best, pool, pool_y)
    }

    /// Get population number.
    pub fn pop_num(&self) -> usize {
        self.pool_y.len()
    }

    /// Get the current best element.
    pub fn as_best_result(&self) -> (&[f64], &F::Fitness) {
        self.best.as_result()
    }

    /// Get the current best fitness value.
    pub fn as_best_fitness(&self) -> &F::Fitness {
        self.best.as_result_fit()
    }

    /// Prune the pool fitness object to reduce memory in some cases.
    pub fn prune_fitness(&mut self) {
        self.pool_y.iter_mut().for_each(|ys| ys.mark_not_best());
    }

    /// Assign the index from source.
    pub fn set_from(&mut self, i: usize, xs: Vec<f64>, ys: F::Fitness) {
        self.pool[i] = xs;
        self.pool_y[i] = ys;
    }

    /// Find the best, and set it globally.
    pub fn find_best(&mut self) {
        self.best.update_all(&self.pool, &self.pool_y);
        self.prune_fitness();
    }
}

impl<F: ObjFunc> core::ops::Deref for Ctx<F> {
    type Target = F;
    fn deref(&self) -> &Self::Target {
        &self.func
    }
}
