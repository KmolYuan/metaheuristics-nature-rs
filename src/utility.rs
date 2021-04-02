//! The definitions of objective function trait and algorithm base.
use std::{
    iter::{repeat, FromIterator},
    time::{Duration, Instant},
};
use rand::{thread_rng, Rng};

#[macro_export]
macro_rules! rand_v {
    ($v1: expr, $v2: expr) => {
        {
            let mut rng = rand::thread_rng();
            rng.gen_range($v1..$v2)
        }
    };
    () => { rand_v!(0., 1.) };
}

#[macro_export]
macro_rules! rand_i {
    ($v: expr) => { rand_v!(0, $v) };
}

#[macro_export]
macro_rules! zeros {
    ($dim: expr) => { Vec::from_iter(repeat(0.).take($dim)) };
    ($w: expr, $h: expr $(, $d: expr)*) => {
        Vec::from_iter(repeat(zeros!($h$(, $d)*)).take($w))
    };
}

pub enum Task {
    MaxGen,
    MinFit,
    MaxTime,
    SlowDown,
}

#[derive(Copy, Clone)]
pub struct Report {
    gen: u32,
    fitness: f64,
    time: f64,
}

/// The base of the objective function. For example:
/// ```
/// struct MyFunc(u32, Vec<f64>, Vec<f64>);
/// impl MyFunc {
///     fn new() { Self(0, vec![0., 0., 0.], vec![50., 50., 50.]) }
/// }
/// impl ObjFunc<f64> for MyFunc {
///     fn fitness(&self, v: &Vec<f64>) -> f64 {
///         v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
///     }
///     fn result(&self, v: &Vec<f64>) -> f64 { self.fitness(v) }
///     fn dim(&self) -> usize { 3 }
///     fn ub(&self, i: usize) -> f64 { self.2[i] }
///     fn lb(&self, i: usize) -> f64 { self.1[i] }
///     fn gen(&mut self) -> &mut u32 { &mut self.0 }
/// }
/// ```
pub trait ObjFunc {
    type Result;
    /// Return fitness, the smaller value represents good.
    fn fitness(&self, v: &Vec<f64>) -> f64;
    /// Return the final result of the problem.
    fn result(&self, v: &Vec<f64>) -> Self::Result;
    /// Get the number of variables.
    fn dim(&self) -> usize;
    /// Get upper bound.
    fn ub(&self, i: usize) -> f64;
    /// Get lower bound.
    fn lb(&self, i: usize) -> f64;
    /// Get generation.
    fn gen(&mut self) -> &mut u32;
}

pub struct Settings {
    task: Task,
    stop_at: f64,
    pop_num: usize,
    rpt: u32,
}

/// Base class of algorithms.
pub struct AlgorithmBase<F: ObjFunc> {
    pub pop_num: usize,
    pub dim: usize,
    rpt: u32,
    task: Task,
    stop_at: f64,
    best_f: f64,
    best: Vec<f64>,
    fitness: Vec<f64>,
    pool: Vec<Vec<f64>>,
    time_start: Instant,
    reports: Vec<Report>,
    pub func: F,
}

impl<F: ObjFunc> AlgorithmBase<F> {
    pub fn new(func: F, settings: Settings) -> Self {
        let dim = func.dim();
        Self {
            pop_num: settings.pop_num,
            dim,
            rpt: settings.rpt,
            task: settings.task,
            stop_at: settings.stop_at,
            best_f: f64::INFINITY,
            best: zeros!(dim),
            fitness: vec![],
            pool: vec![],
            time_start: Instant::now(),
            reports: vec![],
            func,
        }
    }
}

pub trait Algorithm<F: ObjFunc> {
    fn base(&self) -> &AlgorithmBase<F>;
    fn base_mut(&mut self) -> &mut AlgorithmBase<F>;
    fn init(&mut self);
    fn generation(&mut self);
    fn new_pop(&mut self) {
        let b = self.base_mut();
        b.fitness = zeros!(b.pop_num);
        b.pool = zeros!(b.pop_num, b.dim);
    }
    fn assign(&mut self, i: usize, j: usize) {
        let b = self.base_mut();
        b.fitness[i] = b.fitness[j];
        b.pool[i] = b.pool[j].clone();
    }
    fn assign_from(&mut self, i: usize, f: f64, v: &Vec<f64>) {
        let b = self.base_mut();
        b.fitness[i] = f;
        b.pool[i] = v.clone();
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
        let b = self.base_mut();
        for i in 0..b.pop_num {
            for s in 0..b.dim {
                b.pool[i][s] = rand_v!(b.func.lb(s), b.func.ub(s));
            }
            b.fitness[i] = b.func.fitness(&b.pool[i]);
        }
    }
    fn report(&mut self) {
        let b = self.base_mut();
        b.reports.push(Report {
            gen: *b.func.gen(),
            fitness: b.best_f,
            time: (Instant::now() - b.time_start).as_secs_f64(),
        });
    }
    fn history(&self) -> Vec<Report> { self.base().reports.clone() }
    fn result(&self) -> (Vec<f64>, f64) {
        let b = self.base();
        (b.best.clone(), b.best_f)
    }
    fn run(&mut self) -> F::Result {
        *self.base_mut().func.gen() = 0;
        self.base_mut().time_start = Instant::now();
        self.init();
        self.report();
        let mut last_diff = 0.;
        loop {
            let best_f = {
                let b = self.base_mut();
                *b.func.gen() += 1;
                b.best_f
            };
            self.generation();
            if {
                let b = self.base_mut();
                *b.func.gen() % b.rpt == 0
            } {
                self.report();
            }
            let b = self.base_mut();
            match b.task {
                Task::MaxGen => if *b.func.gen() >= b.stop_at as u32 {
                    break;
                }
                Task::MinFit => if b.best_f >= b.stop_at {
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
        {
            let b = self.base();
            b.func.result(&b.best)
        }
    }
}
