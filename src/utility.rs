#[cfg(feature = "cli")]
use indicatif::ProgressBar;
use ndarray::{s, Array1, Array2, ArrayView1, AsArray};
use std::time::Instant;

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

/// The data of generation sampling.
#[derive(Copy, Clone)]
pub struct Report {
    pub gen: u32,
    pub fitness: f64,
    pub time: f64,
}

/// The base of the objective function.
///
/// For example:
/// ```
/// use metaheuristics_nature::ObjFunc;
/// use ndarray::{AsArray, ArrayView1, Array1};
/// struct MyFunc(Array1<f64>, Array1<f64>);
/// impl MyFunc {
///     fn new() -> Self { Self(Array1::zeros(3), Array1::ones(3) * 50.) }
/// }
/// impl ObjFunc for MyFunc {
///     type Result = f64;
///     fn fitness<'a, A>(&self, gen: u32, v: A) -> f64
///     where
///         A: AsArray<'a, f64>,
///     {
///         let v = v.into();
///         v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
///     }
///     fn result<'a, A>(&self, v: A) -> Self::Result
///     where
///         A: AsArray<'a, f64>
///     {
///         self.fitness(0, v)
///     }
///     fn ub(&self) -> ArrayView1<f64> { self.1.view() }
///     fn lb(&self) -> ArrayView1<f64> { self.0.view() }
/// }
/// ```
/// The objective function returns fitness value that used to evaluate the objective.
///
/// The lower bound and upper bound represents the number of variables at the same time.
///
/// This trait is designed as immutable.
pub trait ObjFunc {
    /// The result type.
    type Result;
    /// Return fitness, the smaller value represents good.
    fn fitness<'a, A>(&self, gen: u32, v: A) -> f64
    where
        A: AsArray<'a, f64>;
    /// Return the final result of the problem.
    fn result<'a, A>(&self, v: A) -> Self::Result
    where
        A: AsArray<'a, f64>;
    /// Get upper bound.
    fn ub(&self) -> ArrayView1<f64>;
    /// Get lower bound.
    fn lb(&self) -> ArrayView1<f64>;
}

setting_builder! {
    /// Base settings.
    pub struct Setting {
        task: Task = Task::MaxGen(200),
        pop_num: usize = 200,
        rpt: u32 = 50,
    }
}

#[cfg(feature = "cli")]
struct PB(Option<ProgressBar>);
#[cfg(not(feature = "cli"))]
struct PB;

impl PB {
    fn new(_len: u32) -> Self {
        #[cfg(feature = "cli")]
        if _len > 0 {
            Self(Some(ProgressBar::new(_len as u64)))
        } else {
            Self(None)
        }
        #[cfg(not(feature = "cli"))]
        Self
    }

    fn inc(&self) {
        #[cfg(feature = "cli")]
        if let Some(pb) = &self.0 {
            pb.inc(1);
        }
    }

    fn finish(&self) {
        #[cfg(feature = "cli")]
        if let Some(pb) = &self.0 {
            pb.finish();
        }
    }
}

/// The base class of algorithms.
/// Please see [`Algorithm`] for more information.
pub struct AlgorithmBase<F: ObjFunc> {
    pub pop_num: usize,
    pub dim: usize,
    pub gen: u32,
    pb: PB,
    rpt: u32,
    pub task: Task,
    pub best_f: f64,
    pub best: Array1<f64>,
    pub fitness: Array1<f64>,
    pub pool: Array2<f64>,
    time_start: Instant,
    reports: Vec<Report>,
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
        let pb_gen = match settings.task {
            Task::MaxGen(gen) if cfg!(feature = "cli") => gen,
            _ => 0,
        };
        Self {
            pop_num: settings.pop_num,
            dim,
            gen: 0,
            pb: PB::new(pb_gen),
            rpt: settings.rpt,
            task: settings.task,
            best_f: f64::INFINITY,
            best: Array1::zeros(dim),
            fitness: Array1::zeros(settings.pop_num),
            pool: Array2::zeros((settings.pop_num, dim)),
            time_start: Instant::now(),
            reports: vec![],
            func,
        }
    }
    /// Get fitness from individual `i`.
    pub fn fitness(&mut self, i: usize) {
        self.fitness[i] = self.func.fitness(self.gen, self.pool.slice(s![i, ..]));
    }
    /// Record the performance.
    fn report(&mut self) {
        self.reports.push(Report {
            gen: self.gen,
            fitness: self.best_f,
            time: (Instant::now() - self.time_start).as_secs_f64(),
        });
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
/// impl<F: ObjFunc> MyAlgorithm<F> {
///     fn new(func: F, settings: Setting) -> Self {
///         let base = AlgorithmBase::new(func, settings);
///         Self {
///             tmp: vec![],
///             base,
///         }
///     }
/// }
/// impl<F: ObjFunc> Algorithm<F> for MyAlgorithm<F> {
///     fn base(&self) -> &AlgorithmBase<F> { &self.base }
///     fn base_mut(&mut self) -> &mut AlgorithmBase<F> { &mut self.base }
///     fn generation(&mut self) { unimplemented!() }
/// }
/// ```
/// Your algorithm will be implemented [Solver](trait.Solver.html) automatically.
pub trait Algorithm<F: ObjFunc> {
    /// Return a base handle.
    fn base(&self) -> &AlgorithmBase<F>;
    /// Return a mutable base handle.
    fn base_mut(&mut self) -> &mut AlgorithmBase<F>;
    /// Initialization implementation.
    fn init(&mut self) {}
    /// Processing implementation of each generation.
    fn generation(&mut self);
    /// Get lower bound with index.
    fn lb(&self, i: usize) -> f64 {
        self.base().func.lb()[i]
    }
    /// Get upper bound with index.
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
        b.best_f = b.fitness[i];
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
        if b.fitness[best] < b.best_f {
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
        if self.base().fitness[best] < self.base().best_f {
            self.set_best(best);
        }
    }
    /// Check the bounds.
    fn check(&self, s: usize, v: f64) -> f64 {
        if v > self.ub(s) {
            self.ub(s)
        } else if v < self.lb(s) {
            self.lb(s)
        } else {
            v
        }
    }
}

/// The public API for [`Algorithm`].
pub trait Solver<F: ObjFunc>: Algorithm<F> {
    /// Get the history for plotting.
    fn history(&self) -> Vec<Report> {
        self.base().reports.clone()
    }
    /// Return the x and y of function.
    /// The algorithm must be executed once.
    fn result(&self) -> (Array1<f64>, f64) {
        let b = self.base();
        (b.best.clone(), b.best_f)
    }
    /// Start the algorithm and return the final result.
    fn run(&mut self) -> F::Result {
        self.base_mut().gen = 0;
        self.base_mut().time_start = Instant::now();
        self.init_pop();
        self.init();
        self.base_mut().report();
        let mut last_diff = 0.;
        loop {
            let best_f = {
                let b = self.base_mut();
                b.gen += 1;
                b.best_f
            };
            self.generation();
            if self.base().gen % self.base().rpt == 0 {
                self.base_mut().report();
            }
            let b = self.base_mut();
            match b.task {
                Task::MaxGen(v) => {
                    b.pb.inc();
                    if b.gen >= v {
                        break;
                    }
                }
                Task::MinFit(v) => {
                    if b.best_f <= v {
                        break;
                    }
                }
                Task::MaxTime(v) => {
                    if (Instant::now() - b.time_start).as_secs_f32() >= v {
                        break;
                    }
                }
                Task::SlowDown(v) => {
                    let diff = best_f - b.best_f;
                    if last_diff > 0. && diff / last_diff >= v {
                        break;
                    }
                    last_diff = diff;
                }
            }
        }
        self.base().pb.finish();
        self.base_mut().report();
        self.base().func.result(&self.base().best)
    }
}

impl<F, T> Solver<F> for T
where
    F: ObjFunc,
    T: Algorithm<F>,
{
}
