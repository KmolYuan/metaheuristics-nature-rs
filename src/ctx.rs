use crate::prelude::*;
use alloc::vec::Vec;
use core::iter::zip;

/// A basic context type of the algorithms.
///
/// This type provides a shared dataset if you want to implement a new method.
///
/// Please see [`Algorithm`] for the implementation.
#[non_exhaustive]
pub struct Ctx<F: ObjFunc> {
    /// The best variables
    pub best: Vec<f64>,
    /// Best fitness
    pub best_f: F::Fitness,
    /// Current variables of all individuals
    pub pool: Vec<Vec<f64>>,
    /// Current fitness values of all individuals
    pub pool_f: Vec<F::Fitness>,
    /// Generation
    pub gen: u64,
    /// Objective function object
    pub func: F,
}

impl<F: ObjFunc> Ctx<F> {
    pub(crate) fn from_parts(func: F, pool: Vec<Vec<f64>>, mut pool_f: Vec<F::Fitness>) -> Self {
        let (best_f, best) = zip(&pool_f, &pool)
            .min_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap())
            .map(|(f, xs)| (f.clone(), xs.clone()))
            .unwrap();
        pool_f.iter_mut().for_each(|f| f.mark_not_best());
        Self { best, best_f, pool, pool_f, gen: 0, func }
    }

    pub(crate) fn from_pool(func: F, pool: Vec<Vec<f64>>) -> Self {
        #[cfg(not(feature = "rayon"))]
        let iter = pool.iter();
        #[cfg(feature = "rayon")]
        let iter = pool.par_iter();
        let pool_f = iter.map(|xs| func.fitness(xs)).collect();
        Self::from_parts(func, pool, pool_f)
    }

    /// Get population number.
    #[inline]
    #[must_use = "the population number should be used"]
    pub fn pop_num(&self) -> usize {
        self.pool_f.len()
    }

    /// Get dimension (number of variables).
    #[inline]
    #[must_use = "the dimension value should be used"]
    pub fn dim(&self) -> usize {
        self.best.len()
    }

    /// Get pool shape.
    #[inline]
    #[must_use = "the pool size should be used"]
    pub fn pool_size(&self) -> [usize; 2] {
        [self.pop_num(), self.dim()]
    }

    /// Get the result of the objective function.
    pub fn as_result<P, Fit>(&self) -> &P
    where
        Fit: Fitness + 'static,
        F: ObjFunc<Fitness = Product<P, Fit>>,
    {
        self.best_f.as_result()
    }

    /// Prune the pool fitness object to reduce memory in some cases.
    pub fn prune_fitness(&mut self) {
        self.pool_f.iter_mut().for_each(|f| f.mark_not_best());
    }

    /// Assign the index from best.
    pub fn set_from_best(&mut self, i: usize) {
        self.pool[i] = self.best.clone();
        self.pool_f[i] = self.best_f.clone();
    }

    /// Assign the index from source.
    pub fn set_from(&mut self, i: usize, xs: Vec<f64>, f: F::Fitness) {
        self.pool[i] = xs;
        self.pool_f[i] = f;
    }

    /// Find the best, and set it globally.
    pub fn find_best(&mut self) {
        let (f, xs) = zip(&self.pool_f, &self.pool)
            .min_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap())
            .unwrap();
        if *f < self.best_f {
            self.best_f = f.clone();
            self.best = xs.clone();
        }
        self.prune_fitness();
    }
}

impl<F: ObjFunc> core::ops::Deref for Ctx<F> {
    type Target = F;
    fn deref(&self) -> &Self::Target {
        &self.func
    }
}
