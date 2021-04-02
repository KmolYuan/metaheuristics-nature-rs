//! The definitions of objective function trait and algorithm base.
use std::{
    iter::{repeat, FromIterator},
    time::{Duration, Instant},
};
use rand::{thread_rng, Rng};

macro_rules! rand_v {
    ($v1: expr, $v2: expr) => {
        {
            let mut rng = rand::thread_rng();
            rng.gen_range($v1..$v2)
        }
    };
    () => { rand_v!(0., 1.) };
}

macro_rules! rand_i {
    ($v: expr) => { rand_v!(0, $v) };
}

macro_rules! zeros {
    ($dim: expr) => {
        Vec::from_iter(repeat(0.).take($dim))
    };
    ($w: expr, $h: expr) => {
        Vec::from_iter(repeat(zeros!($h)).take($w))
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

/// The base of the objective function.
/// For example:
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
pub struct Algorithm<F: ObjFunc> {
    pop_num: usize,
    dim: usize,
    rpt: u32,
    task: Task,
    stop_at: f64,
    best_f: f64,
    best: Vec<f64>,
    fitness: Vec<f64>,
    pool: Vec<Vec<f64>>,
    time_start: Instant,
    reports: Vec<Report>,
    func: F,
}

impl<F: ObjFunc> Algorithm<F> {
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
    pub fn new_pop(&mut self) {
        self.fitness = zeros!(self.pop_num);
        self.pool = zeros!(self.pop_num, self.dim);
    }
    pub fn make_tmp(&self) -> Vec<f64> { zeros!(self.dim) }
    pub fn assign(&mut self, i: usize, j: usize) {
        self.fitness[i] = self.fitness[j];
        self.pool[i] = self.pool[j].clone();
    }
    pub fn assign_from(&mut self, i: usize, f: f64, v: &Vec<f64>) {
        self.fitness[i] = f;
        self.pool[i] = v.clone();
    }
    pub fn set_best(&mut self, i: usize) {
        self.best_f = self.fitness[i];
        self.best = self.pool[i].clone();
    }
    pub fn find_best(&mut self) {
        let mut best = 0;
        for i in 0..self.pop_num {
            if self.fitness[i] < self.fitness[best] {
                best = i;
            }
        }
        if self.fitness[best] < self.best_f {
            self.set_best(best);
        }
    }
    pub fn init_pop(&mut self) {
        for i in 0..self.pop_num {
            for s in 0..self.dim {
                self.pool[i][s] = rand_v!(self.func.lb(s), self.func.ub(s));
            }
            self.fitness[i] = self.func.fitness(&self.pool[i]);
        }
    }
    pub fn report(&mut self) {
        self.reports.push(Report {
            gen: *self.func.gen(),
            fitness: self.best_f,
            time: (Instant::now() - self.time_start).as_secs_f64(),
        });
    }
    pub fn history(&self) -> Vec<Report> { self.reports.clone() }
    pub fn result(&self) -> (Vec<f64>, f64) {
        (self.best.clone(), self.best_f)
    }
}
