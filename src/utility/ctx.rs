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
    pub(crate) fn new(func: F, pop_num: usize) -> Self {
        let dim = func.bound().len();
        Self {
            best: Array1::zeros(dim),
            best_f: Default::default(),
            pool: Array2::zeros((pop_num, dim)),
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

    pub(crate) fn init_pop(&mut self, pool: Array2<f64>) {
        let mut fitness = self.pool_f.clone();
        #[cfg(feature = "rayon")]
        let iter = fitness.par_iter_mut();
        #[cfg(not(feature = "rayon"))]
        let iter = fitness.iter_mut();
        let (f, xs) = iter
            .zip(pool.axis_iter(Axis(0)))
            .map(|(f, xs)| {
                *f = self.func.fitness(xs.to_slice().unwrap());
                (f, xs)
            })
            .min_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap())
            .unwrap();
        self.set_best_from(f.clone(), &xs);
        self.pool = pool;
        self.pool_f = fitness;
    }

    /// Prune the pool fitness object to reduce memory in some cases.
    pub fn prune_fitness(&mut self) {
        self.pool_f.iter_mut().for_each(|f| f.mark_not_best());
    }

    /// Set the fitness and variables to best.
    pub fn set_best_from<S>(&mut self, f: F::Fitness, xs: &ArrayBase<S, Ix1>)
    where
        S: ndarray::Data<Elem = f64>,
    {
        self.best_f = f;
        self.best.assign(xs);
    }

    /// Assign the index from best.
    pub fn assign_from_best(&mut self, i: usize) {
        self.pool_f[i] = self.best_f.clone();
        self.pool.slice_mut(s![i, ..]).assign(&self.best);
    }

    /// Assign the index from source.
    pub fn assign_from<'a, A>(&mut self, i: usize, f: F::Fitness, xs: A)
    where
        A: AsArray<'a, f64>,
    {
        self.pool_f[i] = f;
        self.pool.slice_mut(s![i, ..]).assign(&xs.into());
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
        let (f, xs) = self
            .pool_f
            .iter()
            .zip(self.pool.axis_iter(Axis(0)))
            .min_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap())
            .unwrap();
        if force || *f < self.best_f {
            self.best_f = f.clone();
            self.best.assign(&xs);
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
