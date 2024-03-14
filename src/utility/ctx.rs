use crate::utility::prelude::*;
use alloc::{vec, vec::Vec};

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
    pub(crate) fn new(func: F, pop_num: usize) -> Self {
        let dim = func.bound().len();
        Self {
            best: vec![0.; dim],
            best_f: Default::default(),
            pool: vec![vec![0.; dim]; pop_num],
            pool_f: vec![Default::default(); pop_num],
            gen: 0,
            func,
        }
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

    pub(crate) fn init_pop(&mut self, pool: Vec<Vec<f64>>) {
        let mut fitness = self.pool_f.clone();
        #[cfg(feature = "rayon")]
        let iter = fitness.par_iter_mut();
        #[cfg(not(feature = "rayon"))]
        let iter = fitness.iter_mut();
        let (f, xs) = iter
            .zip(&pool)
            .map(|(f, xs)| {
                *f = self.func.fitness(xs);
                (f, xs)
            })
            .min_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap())
            .unwrap();
        self.set_best_from(f.clone(), xs.clone());
        self.pool = pool;
        self.pool_f = fitness;
    }

    /// Prune the pool fitness object to reduce memory in some cases.
    pub fn prune_fitness(&mut self) {
        self.pool_f.iter_mut().for_each(|f| f.mark_not_best());
    }

    /// Set the fitness and variables to best.
    pub fn set_best_from(&mut self, f: F::Fitness, xs: Vec<f64>) {
        self.best_f = f;
        self.best = xs;
    }

    /// Assign the index from best.
    pub fn assign_from_best(&mut self, i: usize) {
        self.pool_f[i] = self.best_f.clone();
        self.pool[i] = self.best.clone();
    }

    /// Assign the index from source.
    pub fn assign_from(&mut self, i: usize, f: F::Fitness, xs: Vec<f64>) {
        self.pool_f[i] = f;
        self.pool[i] = xs;
    }

    /// Find the best, and set it globally.
    pub fn find_best(&mut self) {
        self.find_best_inner(false);
    }

    /// Find the best and override it.
    pub fn find_best_override(&mut self) {
        self.find_best_inner(true);
    }

    fn find_best_inner(&mut self, overrided: bool) {
        let (f, xs) = core::iter::zip(&self.pool_f, &self.pool)
            .min_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap())
            .unwrap();
        if overrided || *f < self.best_f {
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
