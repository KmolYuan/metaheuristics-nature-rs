use crate::*;
use ndarray::{s, Array1, Array2, AsArray};
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
        rpt: u32 = 50,
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
    pub func: F,
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
            fitness: Array1::zeros(settings.pop_num),
            pool: Array2::zeros((settings.pop_num, dim)),
            report: Report::default(),
            reports: vec![],
            func,
        }
    }

    /// Get fitness from individual `i`.
    pub fn fitness(&mut self, i: usize) {
        self.fitness[i] = self.func.fitness(self.pool.slice(s![i, ..]), &self.report);
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
/// use metaheuristics_nature::{AlgorithmBase, Algorithm, ObjFunc, Setting};
/// struct MyAlgorithm<F: ObjFunc> {
///     tmp: Vec<f64>,
///     base: AlgorithmBase<F>,
/// }
/// impl<F: ObjFunc> Algorithm<F> for MyAlgorithm<F> {
///     type Setting = Setting;
///     fn create(func: F, settings: Self::Setting) -> Self {
///         Self {
///             tmp: vec![],
///             base: AlgorithmBase::new(func, settings),
///         }
///     }
///     fn base(&self) -> &AlgorithmBase<F> { &self.base }
///     fn base_mut(&mut self) -> &mut AlgorithmBase<F> { &mut self.base }
///     fn generation(&mut self) { unimplemented!() }
/// }
/// ```
/// Your algorithm will be implemented [Solver](trait.Solver.html) automatically.
pub trait Algorithm<F: ObjFunc>: Sized {
    /// The setting type of the algorithm.
    type Setting;

    /// Create the task.
    fn create(func: F, settings: Self::Setting) -> Self;

    /// Return a base handle.
    fn base(&self) -> &AlgorithmBase<F>;

    /// Return a mutable base handle.
    fn base_mut(&mut self) -> &mut AlgorithmBase<F>;

    /// Initialization implementation.
    fn init(&mut self) {}

    /// Processing implementation of each generation.
    fn generation(&mut self);

    /// Get lower bound with index.
    #[inline(always)]
    fn lb(&self, i: usize) -> f64 {
        self.base().func.lb()[i]
    }

    /// Get upper bound with index.
    #[inline(always)]
    fn ub(&self, i: usize) -> f64 {
        self.base().func.ub()[i]
    }

    /// Assign from source.
    fn assign_from<'a, A>(&mut self, i: usize, f: f64, v: A)
    where
        A: AsArray<'a, f64>,
    {
        let b = self.base_mut();
        b.fitness[i] = f;
        b.pool.slice_mut(s![i, ..]).assign(&v.into());
    }

    /// Set the index to best.
    fn set_best(&mut self, i: usize) {
        let b = self.base_mut();
        b.report.best_f = b.fitness[i];
        b.best.assign(&b.pool.slice(s![i, ..]));
    }

    /// Find the best, and set it globally.
    fn find_best(&mut self) {
        let b = self.base_mut();
        let mut best = 0;
        for i in 0..b.pop_num {
            if b.fitness[i] < b.fitness[best] {
                best = i;
            }
        }
        if b.fitness[best] < b.report.best_f {
            self.set_best(best);
        }
    }

    /// Initialize population.
    fn init_pop(&mut self) {
        let mut best = 0;
        for i in 0..self.base().pop_num {
            for s in 0..self.base().dim {
                self.base_mut().pool[[i, s]] = rand!(self.lb(s), self.ub(s));
            }
            self.base_mut().fitness(i);
            if self.base().fitness[i] < self.base().fitness[best] {
                best = i;
            }
        }
        if self.base().fitness[best] < self.base().report.best_f {
            self.set_best(best);
        }
    }

    /// Check the bounds of the index `s` with the value `v`.
    fn check(&self, s: usize, v: f64) -> f64 {
        if v > self.ub(s) {
            self.ub(s)
        } else if v < self.lb(s) {
            self.lb(s)
        } else {
            v
        }
    }

    /// Start the algorithm process.
    ///
    /// Support a callback function, such as progress bar.
    /// To suppress it, just using a unit type `()`.
    fn run<C>(mut self, callback: impl Callback<C>) -> Self {
        let time_start = Instant::now();
        self.init_pop();
        self.base_mut().report.update_time(time_start);
        self.init();
        callback.call(&self.base().report);
        self.base_mut().report();
        let mut last_diff = 0.;
        loop {
            let best_f = {
                let r = &mut self.base_mut().report;
                r.next_gen();
                r.update_time(time_start);
                r.best_f
            };
            self.generation();
            let b = self.base_mut();
            if b.report.gen % b.rpt == 0 {
                callback.call(&b.report);
                b.report();
            }
            match b.task {
                Task::MaxGen(v) => {
                    if b.report.gen >= v {
                        break;
                    }
                }
                Task::MinFit(v) => {
                    if b.report.best_f <= v {
                        break;
                    }
                }
                Task::MaxTime(v) => {
                    if (Instant::now() - time_start).as_secs_f32() >= v {
                        break;
                    }
                }
                Task::SlowDown(v) => {
                    let diff = best_f - b.report.best_f;
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
    /// Create the task and calling [`Algorithm::run`].
    fn solve<C>(func: F, settings: Self::Setting, callback: impl Callback<C>) -> Self {
        Self::create(func, settings).run(callback)
    }

    /// Get the history for plotting.
    fn history(&self) -> Vec<Report> {
        self.base().reports.clone()
    }

    /// Return the x and y of function.
    /// The algorithm must be executed once.
    fn parameters(&self) -> (Array1<f64>, f64) {
        let b = self.base();
        (b.best.clone(), b.report.best_f)
    }

    /// Get the result of the objective function.
    fn result(&self) -> F::Result {
        let b = self.base();
        b.func.result(&b.best)
    }
}

impl<F, T> Solver<F> for T
where
    F: ObjFunc,
    T: Algorithm<F>,
{
}

/// A trait for fitting different callback functions.
///
/// + Empty callback `()`.
/// + None argument callback `Fn()`.
/// + One argument callback `Fn(&Report)`.
pub trait Callback<C> {
    fn call(&self, report: &Report);
}

impl Callback<()> for () {
    #[inline(always)]
    fn call(&self, _report: &Report) {}
}

impl<T: Fn()> Callback<()> for T {
    #[inline(always)]
    fn call(&self, _report: &Report) {
        self();
    }
}

impl<T: Fn(&Report)> Callback<Report> for T {
    #[inline(always)]
    fn call(&self, report: &Report) {
        self(report);
    }
}
