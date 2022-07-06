use crate::utility::prelude::*;
use alloc::{vec, vec::Vec};

/// The base class of algorithms.
///
/// This type provides a shared dataset if you want to implement a new method.
///
/// Please see [`Algorithm`] for the implementation.
#[non_exhaustive]
pub struct Context<F: ObjFunc> {
    /// Random number generator.
    pub rng: Rng,
    /// The best variables.
    pub best: Array1<f64>,
    /// Best fitness.
    pub best_f: F::Fitness,
    /// Current variables of all individuals.
    pub pool: Array2<f64>,
    /// Current fitness of all individuals.
    pub fitness: Vec<F::Fitness>,
    /// Adaptive factor.
    pub adaptive: f64,
    /// Generation.
    pub gen: u64,
    /// The objective function.
    pub func: F,
}

impl<F: ObjFunc> Context<F> {
    pub(crate) fn new(func: F, seed: Option<u128>, pop_num: usize) -> Self {
        let dim = func.lb().len();
        assert_eq!(
            dim,
            func.ub().len(),
            "different dimension of the variables!"
        );
        Self {
            rng: Rng::new(seed),
            best: Array1::zeros(dim),
            best_f: Default::default(),
            pool: Array2::zeros((pop_num, dim)),
            fitness: vec![Default::default(); pop_num],
            adaptive: 0.,
            gen: 0,
            func,
        }
    }

    /// Get lower bound.
    #[inline(always)]
    #[must_use = "the bound value should be used"]
    pub fn lb(&self, i: usize) -> f64 {
        self.func.lb()[i]
    }

    /// Get upper bound.
    #[inline(always)]
    #[must_use = "the bound value should be used"]
    pub fn ub(&self, i: usize) -> f64 {
        self.func.ub()[i]
    }

    /// Get population number.
    #[inline(always)]
    #[must_use = "the population number should be used"]
    pub fn pop_num(&self) -> usize {
        self.fitness.len()
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
        self.fitness[i] = self.func.fitness(
            self.pool.slice(s![i, ..]).as_slice().unwrap(),
            self.adaptive,
        );
    }

    pub(crate) fn init_pop(&mut self, pool: Array2<f64>) {
        let mut fitness = self.fitness.clone();
        let zip = Zip::from(&mut fitness).and(pool.axis_iter(Axis(0)));
        #[cfg(not(feature = "rayon"))]
        let _ = zip.for_each(|f, v| *f = self.func.fitness(v.to_slice().unwrap(), self.adaptive));
        #[cfg(feature = "rayon")]
        {
            use crate::rayon::prelude::*;
            let (f, v) = zip
                .into_par_iter()
                .map(|(f, v)| {
                    *f = self.func.fitness(v.to_slice().unwrap(), self.adaptive);
                    (f, v)
                })
                .min_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap())
                .unwrap();
            self.set_best_from(f.clone(), v);
        }
        self.pool = pool;
        self.fitness = fitness;
        #[cfg(not(feature = "rayon"))]
        {
            let mut best = 0;
            for i in 1..self.pop_num() {
                if self.fitness[i] < self.fitness[best] {
                    best = i;
                }
            }
            self.set_best(best);
        }
    }

    /// Set the index to best.
    #[inline(always)]
    pub fn set_best(&mut self, i: usize) {
        self.best_f = self.fitness[i].clone();
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
        self.fitness[i] = self.best_f.clone();
        self.pool.slice_mut(s![i, ..]).assign(&self.best);
    }

    /// Assign the index from source.
    #[inline(always)]
    pub fn assign_from<'a, A>(&mut self, i: usize, f: F::Fitness, v: A)
    where
        A: AsArray<'a, f64>,
    {
        self.fitness[i] = f;
        self.pool.slice_mut(s![i, ..]).assign(&v.into());
    }

    /// Find the best, and set it globally.
    pub fn find_best(&mut self) {
        let mut best = 0;
        for i in 1..self.pop_num() {
            if self.fitness[i] < self.fitness[best] {
                best = i;
            }
        }
        if self.fitness[best] < self.best_f {
            self.set_best(best);
        }
    }

    /// Check the bounds of the index `s` with the value `v`.
    #[inline(always)]
    pub fn check(&self, s: usize, v: f64) -> f64 {
        if v > self.ub(s) {
            self.ub(s)
        } else if v < self.lb(s) {
            self.lb(s)
        } else {
            v
        }
    }
}
