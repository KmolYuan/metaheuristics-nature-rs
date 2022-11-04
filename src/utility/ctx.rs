use crate::utility::prelude::*;
use alloc::{vec, vec::Vec};

/// A basic context type of the algorithms.
///
/// This type provides a shared dataset if you want to implement a new method.
///
/// Please see [`Algorithm`] for the implementation.
#[non_exhaustive]
pub struct Ctx<F: ObjFunc> {
    /// Random number generator
    pub rng: Rng,
    /// The best variables
    pub best: Array1<f64>,
    /// Best fitness
    pub best_f: F::Fitness,
    /// Current variables of all individuals
    pub pool: Array2<f64>,
    /// Current fitness values of all individuals
    pub pool_f: Vec<F::Fitness>,
    /// Generation
    pub gen: u64,
    /// Objective function object
    pub func: F,
}

impl<F: ObjFunc> Ctx<F> {
    pub(crate) fn new(func: F, seed: Option<Seed>, pop_num: usize) -> Self {
        let dim = func.bound().len();
        Self {
            rng: Rng::new(seed),
            best: Array1::zeros(dim),
            best_f: Default::default(),
            pool: Array2::zeros((pop_num, dim)),
            pool_f: vec![Default::default(); pop_num],
            gen: 0,
            func,
        }
    }

    /// Get the upper bound and the lower bound values.
    pub fn bound(&self, s: usize) -> [f64; 2] {
        self.func.bound()[s]
    }

    /// Get the width of the upper bound and the lower bound.
    pub fn bound_width(&self, s: usize) -> f64 {
        let [min, max] = self.bound(s);
        max - min
    }

    /// Get the upper bound and the lower bound as a range.
    pub fn bound_range(&self, s: usize) -> core::ops::Range<f64> {
        let [min, max] = self.bound(s);
        min..max
    }

    /// Get the lower bound.
    #[inline(always)]
    #[must_use = "the bound value should be used"]
    pub fn lb(&self, i: usize) -> f64 {
        self.func.bound()[i][0]
    }

    /// Get the upper bound.
    #[inline(always)]
    #[must_use = "the bound value should be used"]
    pub fn ub(&self, i: usize) -> f64 {
        self.func.bound()[i][1]
    }

    /// Get population number.
    #[inline(always)]
    #[must_use = "the population number should be used"]
    pub fn pop_num(&self) -> usize {
        self.pool_f.len()
    }

    /// Get dimension (number of variables).
    #[inline(always)]
    #[must_use = "the dimension value should be used"]
    pub fn dim(&self) -> usize {
        self.best.len()
    }

    /// Get pool shape.
    #[inline(always)]
    #[must_use = "the pool size should be used"]
    pub fn pool_size(&self) -> [usize; 2] {
        [self.pop_num(), self.dim()]
    }

    /// Get fitness from individual `i`.
    pub fn fitness(&mut self, i: usize) {
        self.pool_f[i] = self
            .func
            .fitness(self.pool.slice(s![i, ..]).as_slice().unwrap());
    }

    /// Get the current best result of the objective function.
    ///
    /// This method can generate midway product of convergence if necessary.
    pub fn result(&self) -> F::Product
    where
        F: ObjFactory,
    {
        self.func.produce(self.best.as_slice().unwrap())
    }

    pub(crate) fn init_pop(&mut self, pool: Array2<f64>) {
        let mut fitness = self.pool_f.clone();
        #[cfg(feature = "rayon")]
        let iter = fitness.par_iter_mut();
        #[cfg(not(feature = "rayon"))]
        let iter = fitness.iter_mut();
        let (f, v) = iter
            .zip(pool.axis_iter(Axis(0)))
            .map(|(f, v)| {
                *f = self.func.fitness(v.to_slice().unwrap());
                (f.clone(), v)
            })
            .min_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap())
            .unwrap();
        self.set_best_from(f, v);
        self.pool = pool;
        self.pool_f = fitness;
    }

    /// Set the index to best.
    #[inline(always)]
    pub fn set_best(&mut self, i: usize) {
        self.best_f = self.pool_f[i].clone();
        self.best.assign(&self.pool.slice(s![i, ..]));
    }

    /// Set the fitness and variables to best.
    #[inline(always)]
    pub fn set_best_from<'a, A>(&mut self, f: F::Fitness, v: A)
    where
        A: AsArray<'a, f64>,
    {
        self.best_f = f;
        self.best.assign(&v.into());
    }

    /// Assign the index from best.
    #[inline(always)]
    pub fn assign_from_best(&mut self, i: usize) {
        self.pool_f[i] = self.best_f.clone();
        self.pool.slice_mut(s![i, ..]).assign(&self.best);
    }

    /// Assign the index from source.
    #[inline(always)]
    pub fn assign_from<'a, A>(&mut self, i: usize, f: F::Fitness, v: A)
    where
        A: AsArray<'a, f64>,
    {
        self.pool_f[i] = f;
        self.pool.slice_mut(s![i, ..]).assign(&v.into());
    }

    /// Find the best, and set it globally.
    pub fn find_best(&mut self) {
        self.find_best_inner(false);
    }

    /// Find the best and force override it.
    pub fn find_best_force(&mut self) {
        self.find_best_inner(true);
    }

    fn find_best_inner(&mut self, force: bool) {
        let (i, f) = self
            .pool_f
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        if force || f < &self.best_f {
            self.set_best(i);
        }
    }

    /// Check the bounds of the index `s` with the value `v`.
    #[inline(always)]
    pub fn clamp(&self, s: usize, v: f64) -> f64 {
        let [min, max] = self.func.bound()[s];
        v.clamp(min, max)
    }
}
