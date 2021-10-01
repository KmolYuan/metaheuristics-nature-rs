use crate::{thread_pool::ThreadPool, utility::*, *};
use alloc::{sync::Arc, vec::Vec};

/// The base class of algorithms.
///
/// This type provides a shared dataset if you want to implement a new method.
///
/// Please see [`Algorithm`] for the implementation.
pub struct Context<F> {
    /// Termination condition.
    pub task: Task,
    /// The best variables.
    pub best: Array1<f64>,
    /// Current fitness of all individuals.
    pub fitness: Array1<f64>,
    /// Current variables of all individuals.
    pub pool: Array2<f64>,
    /// The current information of the algorithm.
    pub report: Report,
    pub(crate) reports: Vec<Report>,
    pub(crate) rpt: u32,
    pub(crate) average: bool,
    pub(crate) adaptive: Adaptive,
    /// The objective function.
    pub func: Arc<F>,
}

impl<F: ObjFunc> Context<F> {
    pub(crate) fn new(func: F, s: &BasicSetting) -> Self {
        let dim = func.lb().len();
        assert_eq!(
            dim,
            func.ub().len(),
            "different dimension of the variables!"
        );
        Self {
            task: s.task.clone(),
            best: Array1::zeros(dim),
            fitness: Array1::ones(s.pop_num) * f64::INFINITY,
            pool: Array2::zeros((s.pop_num, dim)),
            report: Report::default(),
            reports: Vec::new(),
            rpt: s.rpt,
            average: s.average,
            adaptive: s.adaptive.clone(),
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
        self.fitness[i] = self
            .func
            .fitness(self.pool.slice(s![i, ..]).as_slice().unwrap(), &self.report);
    }

    pub(crate) fn init_pop(&mut self) {
        let mut tasks = ThreadPool::new();
        let mut best = 0;
        for i in 0..self.pop_num() {
            for s in 0..self.dim() {
                self.pool[[i, s]] = rand_float(self.lb(s), self.ub(s));
            }
            tasks.insert(
                i,
                self.func.clone(),
                self.report.clone(),
                self.pool.slice(s![i, ..]),
            );
        }
        for (i, f) in tasks {
            self.fitness[i] = f;
            if self.fitness[i] < self.fitness[best] {
                best = i;
            }
        }
        self.set_best(best);
    }

    /// Set the index to best.
    #[inline(always)]
    pub fn set_best(&mut self, i: usize) {
        self.report.best_f = self.fitness[i];
        self.best.assign(&self.pool.slice(s![i, ..]));
    }

    /// Assign the index from best.
    #[inline(always)]
    pub fn assign_from_best(&mut self, i: usize) {
        self.fitness[i] = self.report.best_f;
        self.pool.slice_mut(s![i, ..]).assign(&self.best);
    }

    /// Assign the index from source.
    #[inline(always)]
    pub fn assign_from<'a, A>(&mut self, i: usize, f: f64, v: A)
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
        if self.fitness[best] < self.report.best_f {
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

    /// Record the performance.
    pub(crate) fn report(&mut self) {
        self.reports.push(self.report.clone());
    }
}
