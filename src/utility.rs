use std::time::Instant;

#[macro_export]
macro_rules! rand {
    ($v1:expr, $v2:expr) => {
        {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            rng.gen_range($v1..$v2)
        }
    };
    ($v: expr) => { rand!(0, $v) };
    () => { rand!(0., 1.) };
}

#[macro_export]
macro_rules! zeros {
    () => { 0. };
    ($w:expr $(, $h:expr)* $(,)?) => {
        {
            use std::iter::{repeat, FromIterator};
            Vec::from_iter(repeat(zeros!($($h,)*)).take($w))
        }
    };
}

/// The terminal condition of the algorithm setting.
#[derive(Eq, PartialEq)]
pub enum Task {
    MaxGen,
    MinFit,
    MaxTime,
    SlowDown,
}

/// The data of generation sampling.
#[derive(Copy, Clone)]
pub struct Report {
    pub gen: u32,
    pub fitness: f64,
    pub time: f64,
}

/// The base of the objective function. For example:
/// ```
/// use metaheuristics::ObjFunc;
/// struct MyFunc(Vec<f64>, Vec<f64>);
/// impl MyFunc {
///     fn new() -> Self { Self(vec![0., 0., 0.], vec![50., 50., 50.]) }
/// }
/// impl ObjFunc for MyFunc {
///     type Result = f64;
///     fn fitness(&self, _gen: u32, v: &Vec<f64>) -> f64 {
///         v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
///     }
///     fn result(&self, v: &Vec<f64>) -> f64 { self.fitness(0, v) }
///     fn ub(&self) -> &Vec<f64> { &self.1 }
///     fn lb(&self) -> &Vec<f64> { &self.0 }
/// }
/// ```
pub trait ObjFunc {
    type Result;
    /// Return fitness, the smaller value represents good.
    fn fitness(&self, gen: u32, v: &Vec<f64>) -> f64;
    /// Return the final result of the problem.
    fn result(&self, v: &Vec<f64>) -> Self::Result;
    /// Get upper bound.
    fn ub(&self) -> &Vec<f64>;
    /// Get lower bound.
    fn lb(&self) -> &Vec<f64>;
}

/// Base settings.
pub struct Setting {
    pub task: Task,
    pub stop_at: f64,
    pub pop_num: usize,
    pub rpt: u32,
}

impl Default for Setting {
    fn default() -> Self {
        Self {
            task: Task::MaxGen,
            stop_at: 200.,
            pop_num: 200,
            rpt: 50,
        }
    }
}

/// The base class of algorithms.
pub struct AlgorithmBase<F: ObjFunc> {
    pub pop_num: usize,
    pub dim: usize,
    pub gen: u32,
    rpt: u32,
    pub task: Task,
    pub stop_at: f64,
    pub best_f: f64,
    pub best: Vec<f64>,
    pub fitness: Vec<f64>,
    pub pool: Vec<Vec<f64>>,
    time_start: Instant,
    reports: Vec<Report>,
    pub func: F,
}

impl<F: ObjFunc> AlgorithmBase<F> {
    pub fn new(func: F, settings: Setting) -> Self {
        let dim = {
            let lb = func.lb();
            let ub = func.ub();
            assert_eq!(lb.len(), ub.len());
            lb.len()
        };
        Self {
            pop_num: settings.pop_num,
            dim,
            gen: 0,
            rpt: settings.rpt,
            task: settings.task,
            stop_at: settings.stop_at,
            best_f: f64::INFINITY,
            best: zeros!(dim),
            fitness: zeros!(settings.pop_num),
            pool: zeros!(settings.pop_num, dim),
            time_start: Instant::now(),
            reports: vec![],
            func,
        }
    }
}

/// The methods of the meta-heuristic algorithms.
pub trait Algorithm<F: ObjFunc> {
    fn base(&self) -> &AlgorithmBase<F>;
    fn base_mut(&mut self) -> &mut AlgorithmBase<F>;
    /// Initialization implementation.
    fn init(&mut self);
    /// Processing implementation of each generation.
    fn generation(&mut self);
    fn lb(&self, i: usize) -> f64 { self.base().func.lb()[i] }
    fn ub(&self, i: usize) -> f64 { self.base().func.ub()[i] }
    fn assign(&mut self, i: usize, j: usize) {
        let b = self.base_mut();
        b.fitness[i] = b.fitness[j];
        b.pool[i] = b.pool[j].clone();
    }
    fn assign_from(&mut self, i: usize, f: f64, v: Vec<f64>) {
        let b = self.base_mut();
        b.fitness[i] = f;
        b.pool[i] = v;
    }
    fn set_best(&mut self, i: usize) {
        let b = self.base_mut();
        b.best_f = b.fitness[i];
        b.best = b.pool[i].clone();
    }
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
    fn init_pop(&mut self) {
        for i in 0..self.base().pop_num {
            for s in 0..self.base().dim {
                self.base_mut().pool[i][s] = rand!(self.lb(s), self.ub(s));
            }
            let b = self.base_mut();
            b.fitness[i] = b.func.fitness(b.gen, &b.pool[i]);
        }
    }
    fn report(&mut self) {
        let b = self.base_mut();
        b.reports.push(Report {
            gen: b.gen,
            fitness: b.best_f,
            time: (Instant::now() - b.time_start).as_secs_f64(),
        });
    }
    /// Get the history for plotting.
    fn history(&self) -> Vec<Report> { self.base().reports.clone() }
    /// Return the x and y of function.
    fn result(&self) -> (Vec<f64>, f64) {
        let b = self.base();
        (b.best.clone(), b.best_f)
    }
    /// Start the algorithm.
    fn run(&mut self) -> F::Result {
        self.base_mut().gen = 0;
        self.base_mut().time_start = Instant::now();
        self.init();
        self.report();
        let mut last_diff = 0.;
        loop {
            let best_f = {
                let b = self.base_mut();
                b.gen += 1;
                b.best_f
            };
            self.generation();
            if self.base().gen % self.base().rpt == 0 {
                self.report();
            }
            let b = self.base_mut();
            match b.task {
                Task::MaxGen => if b.gen >= b.stop_at as u32 {
                    break;
                }
                Task::MinFit => if b.best_f <= b.stop_at {
                    break;
                }
                Task::MaxTime => if (Instant::now() - b.time_start).as_secs_f64() >= b.stop_at {
                    break;
                }
                Task::SlowDown => {
                    let diff = best_f - b.best_f;
                    if last_diff > 0. && diff / last_diff >= b.stop_at {
                        break;
                    }
                    last_diff = diff;
                }
            }
        }
        self.report();
        self.base().func.result(&self.base().best)
    }
}
