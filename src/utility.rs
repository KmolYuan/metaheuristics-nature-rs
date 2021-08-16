use crate::{random::*, thread_pool::ThreadPool, *};
use alloc::{sync::Arc, vec, vec::Vec};
use ndarray::s;
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
    /// Setting base.
    pub struct BasicSetting {
        /// Termination condition.
        task: Task = Task::MaxGen(200),
        /// Population number.
        pop_num: usize = 200,
        /// The report frequency. (per generation)
        rpt: u32 = 1,
    }
}

/// A trait that provides a conversion to original setting.
///
/// Can be auto implemented through [`setting_builder!`].
pub trait Setting {
    /// Associated algorithm.
    type Algorithm: Algorithm<Setting = Self>;
    /// Convert to original setting.
    fn into_setting(self) -> BasicSetting;
}

/// The base class of algorithms.
/// Please see [`Solver`] for more information.
pub struct Context<F> {
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

impl<F: ObjFunc> Context<F> {
    pub fn new(func: F, settings: BasicSetting) -> Self {
        let dim = func.lb().len();
        assert_eq!(
            dim,
            func.ub().len(),
            "different dimension of the variables!"
        );
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
        for i in 1..self.pop_num {
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
    fn report(&mut self) {
        self.reports.push(self.report.clone());
    }
}

/// The methods of the metaheuristic algorithms.
///
/// This trait is extendable.
/// Create a structure and implement `Algorithm` member to implement it.
/// ```
/// use metaheuristics_nature::{Algorithm, ObjFunc, setting_builder, Context};
///
/// setting_builder! {
///     pub struct MySetting for MyAlgorithm {
///         @base,
///     }
/// }
///
/// pub struct MyAlgorithm;
///
/// impl Algorithm for MyAlgorithm {
///     type Setting = MySetting;
///     fn create(settings: &Self::Setting) -> Self {
///         unimplemented!()
///     }
///     fn generation<F: ObjFunc>(&mut self, ctx: &mut Context<F>) {
///         unimplemented!()
///     }
/// }
/// ```
/// Your algorithm will be implemented by [Solver] automatically.
pub trait Algorithm: Sized {
    /// The setting type of the algorithm.
    type Setting: Setting<Algorithm = Self>;

    /// Create the task.
    fn create(settings: &Self::Setting) -> Self;

    /// Initialization implementation.
    #[inline(always)]
    #[allow(unused_variables)]
    fn init<F: ObjFunc>(&mut self, ctx: &mut Context<F>) {}

    /// Processing implementation of each generation.
    fn generation<F: ObjFunc>(&mut self, ctx: &mut Context<F>);
}

/// A public API for [`Algorithm`].
///
/// Users can simply obtain their solution and see the result.
///
/// + The method is a type that implemented [`Algorithm`].
/// + The objective function is a type that implement [`ObjFunc`].
/// + A basic algorithm data is hold by [`Context`].
///
/// This type can infer the algorithm by [`Setting::Algorithm`].
pub struct Solver<M: Algorithm, F: ObjFunc> {
    method: M,
    ctx: Context<F>,
}

impl<S, M, F> Solver<M, F>
where
    S: Setting<Algorithm = M>,
    M: Algorithm<Setting = S>,
    F: ObjFunc,
{
    /// Create the task and run the algorithm.
    ///
    /// Argument `callback` is a progress feedback function,
    /// returns true to keep algorithm running, same as the behavior of the while-loop.
    pub fn solve(func: F, settings: S, callback: impl FnMut(Report) -> bool) -> Self {
        Self {
            method: M::create(&settings),
            ctx: Context::new(func, settings.into_setting()),
        }
        .run(callback)
    }
}

impl<M: Algorithm, F: ObjFunc> Solver<M, F> {
    fn run(mut self, mut callback: impl FnMut(Report) -> bool) -> Self {
        #[cfg(feature = "std")]
        let time_start = Instant::now();
        self.ctx.init_pop();
        #[cfg(feature = "std")]
        {
            self.ctx.report.update_time(time_start);
        }
        self.method.init(&mut self.ctx);
        if !callback(self.ctx.report.clone()) {
            return self;
        }
        self.ctx.report();
        let mut last_diff = 0.;
        loop {
            let best_f = {
                self.ctx.report.next_gen();
                #[cfg(feature = "std")]
                {
                    self.ctx.report.update_time(time_start);
                }
                self.ctx.report.best_f
            };
            self.method.generation(&mut self.ctx);
            if self.ctx.report.gen % self.ctx.rpt == 0 {
                if !callback(self.ctx.report.clone()) {
                    break;
                }
                self.ctx.report();
            }
            match self.ctx.task {
                Task::MaxGen(v) => {
                    if self.ctx.report.gen >= v {
                        break;
                    }
                }
                Task::MinFit(v) => {
                    if self.ctx.report.best_f <= v {
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
                    let diff = best_f - self.ctx.report.best_f;
                    if last_diff > 0. && diff / last_diff >= v {
                        break;
                    }
                    last_diff = diff;
                }
            }
        }
        self
    }

    /// Get the history for plotting.
    #[inline(always)]
    pub fn history(&self) -> Vec<Report> {
        self.ctx.reports.clone()
    }

    /// Return the x and y of function.
    /// The algorithm must be executed once.
    #[inline(always)]
    pub fn parameters(&self) -> (Array1<f64>, f64) {
        (self.ctx.best.to_owned(), self.ctx.report.best_f)
    }

    /// Get the result of the objective function.
    #[inline(always)]
    pub fn result(&self) -> F::Result {
        self.ctx.func.result(&self.ctx.best)
    }
}
