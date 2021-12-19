use crate::utility::prelude::*;
use alloc::{sync::Arc, vec, vec::Vec};

/// The base class of algorithms.
///
/// This type provides a shared dataset if you want to implement a new method.
///
/// Please see [`Algorithm`] for the implementation.
pub struct Context<F: ObjFunc> {
    /// Random number generator.
    pub rng: Rng,
    /// Termination condition.
    pub task: Task,
    /// The best variables.
    pub best: Array1<f64>,
    /// Current fitness of all individuals.
    pub fitness: Vec<F::Respond>,
    /// Current variables of all individuals.
    pub pool: Array2<f64>,
    /// Adaptive factor.
    pub adaptive: f64,
    /// Time duration.
    #[cfg(feature = "std")]
    #[cfg_attr(doc_cfg, doc(cfg(feature = "std")))]
    pub time: f64,
    /// Generation.
    pub gen: u64,
    /// Best fitness.
    pub best_f: f64,
    /// Is the best fitness feasible.
    pub best_feasible: bool,
    /// Gradient of the best fitness, between the current and the previous.
    pub diff: f64,
    /// The objective function.
    pub func: Arc<F>,
}

impl<F: ObjFunc> Context<F> {
    pub(crate) fn new(func: F, s: BasicSetting) -> Self {
        let dim = func.lb().len();
        assert_eq!(
            dim,
            func.ub().len(),
            "different dimension of the variables!"
        );
        Self {
            rng: Rng::new(s.seed),
            task: s.task,
            best: Array1::zeros(dim),
            fitness: vec![F::Respond::INFINITY; s.pop_num],
            pool: Array2::zeros((s.pop_num, dim)),
            adaptive: 0.,
            #[cfg(feature = "std")]
            time: 0.,
            gen: 0,
            best_f: f64::INFINITY,
            best_feasible: false,
            diff: 0.0,
            func: Arc::new(func),
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

    /// Get dimension (number of variables).
    #[inline(always)]
    #[must_use = "the dimension value should be used"]
    pub fn dim(&self) -> usize {
        self.best.len()
    }

    /// Get population number.
    #[inline(always)]
    #[must_use = "the population number should be used"]
    pub fn pop_num(&self) -> usize {
        self.fitness.len()
    }

    /// Get fitness from individual `i`.
    pub fn fitness(&mut self, i: usize) {
        self.fitness[i] = self.func.fitness(
            self.pool.slice(s![i, ..]).as_slice().unwrap(),
            self.adaptive,
        );
    }

    pub(crate) fn init_pop(&mut self) {
        let pool = Array2::from_shape_fn([self.pop_num(), self.dim()], |(_, s)| {
            self.rng.float(self.lb(s)..self.ub(s))
        });
        let mut fitness = self.fitness.clone();
        let zip = Zip::from(&mut fitness).and(pool.axis_iter(Axis(0)));
        #[cfg(not(feature = "parallel"))]
        {
            zip.for_each(|f, v| {
                *f = self.func.fitness(v.to_slice().unwrap(), self.adaptive);
            });
            self.find_best();
        }
        #[cfg(feature = "parallel")]
        {
            let (f, v) = zip
                .into_par_iter()
                .map(|(f, v)| {
                    *f = self.func.fitness(v.to_slice().unwrap(), self.adaptive);
                    (f.value(), v)
                })
                .min_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap())
                .unwrap();
            self.set_best_from(f, v);
        }
        self.pool = pool;
        self.fitness = fitness;
    }

    /// Set the index to best.
    #[inline(always)]
    pub fn set_best(&mut self, i: usize) {
        self.best_f = self.fitness[i].value();
        self.best_feasible = self.fitness[i].feasible();
        self.best.assign(&self.pool.slice(s![i, ..]));
    }

    /// Set the fitness and variables to best.
    #[inline(always)]
    pub fn set_best_from<'a, A>(&mut self, f: f64, v: A)
    where
        A: AsArray<'a, f64>,
    {
        self.best_f = f;
        self.best.assign(&v.into());
    }

    /// Assign the index from best.
    #[inline(always)]
    pub fn assign_from_best(&mut self, i: usize) {
        self.fitness[i] = F::Respond::from_value(self.best_f, self.best_feasible);
        self.pool.slice_mut(s![i, ..]).assign(&self.best);
    }

    /// Assign the index from source.
    #[inline(always)]
    pub fn assign_from<'a, A>(&mut self, i: usize, f: F::Respond, v: A)
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
            if self.fitness[i].value() < self.fitness[best].value() {
                best = i;
            }
        }
        if self.fitness[best].value() < self.best_f {
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
