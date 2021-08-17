//! The utility API used to create a new algorithm.
//!
//! When building a new method, just import this module as prelude.
//!
//! ```
//! use metaheuristics_nature::{utility::*, *};
//! ```
//!
//! In other hand, if you went to fork the task manually by using parallel structure,
//! import [`thread_pool::ThreadPool`] is required.
pub use crate::random::*;
use crate::{thread_pool::ThreadPool, *};
use alloc::{sync::Arc, vec::Vec};
use ndarray::s;

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
    pub(crate) rpt: u32,
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
            reports: Vec::new(),
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

    pub(crate) fn init_pop(&mut self) {
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
    pub(crate) fn report(&mut self) {
        self.reports.push(self.report.clone());
    }
}

/// The methods of the metaheuristic algorithms.
///
/// This trait is extendable.
/// Create a structure and implement `Algorithm` member to implement it.
/// ```
/// use metaheuristics_nature::{setting_builder, utility::*};
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

/// Product two iterators together.
pub fn product<A, I1, I2>(iter1: I1, iter2: I2) -> impl Iterator<Item = (A, A)>
where
    A: Clone,
    I1: IntoIterator<Item = A>,
    I2: IntoIterator<Item = A> + Clone,
{
    iter1
        .into_iter()
        .flat_map(move |e: A| core::iter::repeat(e).zip(iter2.clone()))
}
