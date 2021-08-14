use crate::{random::*, thread_pool::ThreadPool, *};
use alloc::{sync::Arc, vec, vec::Vec};
use core::ops::{Deref, DerefMut};
use ndarray::{s, Array1, Array2, AsArray};
#[cfg(feature = "std")]
use std::time::Instant;

/// The data of generation sampling.
#[derive(Clone, Debug)]
pub struct Report {
    /// Generation.
    pub gen: u32,
    /// The best fitness.
    pub best_f: f64,
    /// Time duration.
    pub time: f64,
}

impl Default for Report {
    fn default() -> Self {
        Self {
            gen: 0,
            best_f: f64::INFINITY,
            time: 0.,
        }
    }
}

impl Report {
    /// Go into next generation.
    pub fn next_gen(&mut self) {
        self.gen += 1;
    }

    /// Update time by a starting point.
    #[cfg(feature = "std")]
    #[cfg_attr(doc_cfg, doc(cfg(feature = "std")))]
    pub fn update_time(&mut self, time: Instant) {
        self.time = (Instant::now() - time).as_secs_f64();
    }
}

/// The terminal condition of the algorithm setting.
pub enum Task {
    /// Max generation.
    MaxGen(u32),
    /// Minimum fitness.
    MinFit(f64),
    /// Max time in second.
    #[cfg(feature = "std")]
    #[cfg_attr(doc_cfg, doc(cfg(feature = "std")))]
    MaxTime(f32),
    /// Minimum delta value.
    SlowDown(f64),
}

setting_builder! {
    /// Base settings.
    pub struct Setting {
        /// Termination condition.
        task: Task = Task::MaxGen(200),
        /// Population number.
        pop_num: usize = 200,
        /// The report frequency. (per generation)
        rpt: u32 = 1,
    }
}

/// The base class of algorithms.
/// Please see [`Algorithm`] for more information.
pub struct AlgorithmBase<F: ObjFunc> {
    /// Population number.
    pub pop_num: usize,
    /// Dimension, the variable number of the problem.
    pub dim: usize,
    rpt: u32,
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
    reports: Vec<Report>,
    /// The objective function.
    pub func: Arc<F>,
}

impl<F: ObjFunc> AlgorithmBase<F> {
    pub fn new(func: F, settings: Setting) -> Self {
        let dim = {
            let lb = func.lb();
            let ub = func.ub();
            assert_eq!(lb.len(), ub.len(), "different dimension of the variables!");
            lb.len()
        };
        Self {
            pop_num: settings.pop_num,
            dim,
            rpt: settings.rpt,
            task: settings.task,
            best: Array1::zeros(dim),
            fitness: Array1::ones(settings.pop_num) * f64::INFINITY,
            pool: Array2::zeros((settings.pop_num, dim)),
            report: Default::default(),
            reports: vec![],
            func: Arc::new(func),
        }
    }

    #[inline(always)]
    pub fn lb(&self, i: usize) -> f64 {
        self.func.lb()[i]
    }

    #[inline(always)]
    pub fn ub(&self, i: usize) -> f64 {
        self.func.ub()[i]
    }

    /// Get fitness from individual `i`.
    pub fn fitness(&mut self, i: usize) {
        self.fitness[i] = self.func.fitness(self.pool.slice(s![i, ..]), &self.report);
    }

    /// Set the index to best.
    pub fn set_best(&mut self, i: usize) {
        self.report.best_f = self.fitness[i];
        self.best.assign(&self.pool.slice(s![i, ..]));
    }

    /// Assign from source.
    #[inline(always)]
    pub fn assign_from<'a, A>(&mut self, i: usize, f: f64, v: A)
    where
        A: AsArray<'a, f64>,
    {
        self.fitness[i] = f;
        self.pool.slice_mut(s![i, ..]).assign(&v.into());
    }

    /// Record the performance.
    fn report(&mut self) {
        self.reports.push(self.report.clone());
    }
}

/// The methods of the metaheuristic algorithms.
///
/// This trait is extendable.
/// Create a structure and store a `AlgorithmBase` member to implement it.
/// ```
/// use metaheuristics_nature::{Algorithm, AlgorithmBase, ObjFunc, Setting};
/// use std::ops::{Deref, DerefMut};
///
/// struct MyAlgorithm<F: ObjFunc> {
///     tmp: Vec<f64>,
///     base: AlgorithmBase<F>,
/// }
///
/// impl<F: ObjFunc> Deref for MyAlgorithm<F> {
///     type Target = AlgorithmBase<F>;
///     fn deref(&self) -> &Self::Target {
///         &self.base
///     }
/// }
///
/// impl<F: ObjFunc> DerefMut for MyAlgorithm<F> {
///     fn deref_mut(&mut self) -> &mut Self::Target {
///         &mut self.base
///     }
/// }
///
/// impl<F: ObjFunc> Algorithm<F> for MyAlgorithm<F> {
///     type Setting = Setting;
///     fn create(func: F, settings: Self::Setting) -> Self {
///         Self {
///             tmp: vec![],
///             base: AlgorithmBase::new(func, settings),
///         }
///     }
///     fn generation(&mut self) {
///         todo!()
///     }
/// }
/// ```
/// Your algorithm will be implemented [Solver](trait.Solver.html) automatically.
pub trait Algorithm<F: ObjFunc>: Deref<Target = AlgorithmBase<F>> + DerefMut + Sized {
    /// The setting type of the algorithm.
    type Setting;

    /// Create the task.
    fn create(func: F, settings: Self::Setting) -> Self;

    /// Initialization implementation.
    fn init(&mut self) {}

    /// Processing implementation of each generation.
    fn generation(&mut self);

    /// Find the best, and set it globally.
    fn find_best(&mut self) {
        let mut best = 0;
        for i in 1..self.pop_num {
            if self.fitness[i] < self.fitness[best] {
                best = i;
            }
        }
        if self.fitness[best] < self.report.best_f {
            self.set_best(best);
        }
    }

    /// Initialize population.
    fn init_pop(&mut self) {
        let mut tasks = ThreadPool::new();
        let mut best = 0;
        for i in 0..self.pop_num {
            for s in 0..self.dim {
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

    /// Check the bounds of the index `s` with the value `v`.
    #[inline(always)]
    fn check(&self, s: usize, v: f64) -> f64 {
        if v > self.ub(s) {
            self.ub(s)
        } else if v < self.lb(s) {
            self.lb(s)
        } else {
            v
        }
    }

    #[doc(hidden)]
    fn run(mut self, mut callback: impl FnMut(Report) -> bool) -> Self {
        #[cfg(feature = "std")]
        let time_start = Instant::now();
        self.init_pop();
        #[cfg(feature = "std")]
        {
            self.report.update_time(time_start);
        }
        self.init();
        if !callback(self.report.clone()) {
            return self;
        }
        self.report();
        let mut last_diff = 0.;
        loop {
            let best_f = {
                self.report.next_gen();
                #[cfg(feature = "std")]
                {
                    self.report.update_time(time_start);
                }
                self.report.best_f
            };
            self.generation();
            if self.report.gen % self.rpt == 0 {
                if !callback(self.report.clone()) {
                    break;
                }
                self.report();
            }
            match self.task {
                Task::MaxGen(v) => {
                    if self.report.gen >= v {
                        break;
                    }
                }
                Task::MinFit(v) => {
                    if self.report.best_f <= v {
                        break;
                    }
                }
                #[cfg(feature = "std")]
                Task::MaxTime(v) => {
                    if (Instant::now() - time_start).as_secs_f32() >= v {
                        break;
                    }
                }
                Task::SlowDown(v) => {
                    let diff = best_f - self.report.best_f;
                    if last_diff > 0. && diff / last_diff >= v {
                        break;
                    }
                    last_diff = diff;
                }
            }
        }
        self
    }
}

/// A public API for [`Algorithm`].
///
/// Users can simply obtain their solution and see the result.
pub trait Solver<F: ObjFunc>: Algorithm<F> {
    /// Create the task and run the algorithm.
    ///
    /// Argument `callback` is a progress feedback function,
    /// returns true to keep algorithm running, same as the behavior of the while-loop.
    fn solve(func: F, settings: Self::Setting, callback: impl FnMut(Report) -> bool) -> Self {
        Self::create(func, settings).run(callback)
    }

    /// Get the history for plotting.
    #[inline(always)]
    fn history(&self) -> Vec<Report> {
        self.reports.clone()
    }

    /// Return the x and y of function.
    /// The algorithm must be executed once.
    #[inline(always)]
    fn parameters(&self) -> (Array1<f64>, f64) {
        (self.best.to_owned(), self.report.best_f)
    }

    /// Get the result of the objective function.
    #[inline(always)]
    fn result(&self) -> F::Result {
        self.func.result(&self.best)
    }
}

impl<F, T> Solver<F> for T
where
    F: ObjFunc,
    T: Algorithm<F>,
{
}
