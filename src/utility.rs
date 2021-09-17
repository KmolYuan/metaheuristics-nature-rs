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
pub use ndarray::{s, Array1, Array2, AsArray};

/// Setting base.
///
/// Please see [setting!] for more usage.
pub struct BasicSetting {
    /// Termination condition.
    pub task: Task,
    /// Population number.
    pub pop_num: usize,
    /// The report frequency. (per generation)
    pub rpt: u32,
}

impl Default for BasicSetting {
    fn default() -> Self {
        Self {
            task: Task::MaxGen(200),
            pop_num: 200,
            rpt: 1,
        }
    }
}

/// A trait that provides a conversion to original setting.
///
/// The setting type is actually a builder of the [`Setting::Algorithm`] type.
///
/// Before the implementation,
/// the builder function of the setting type can be implemented by [`setting!`].
pub trait Setting {
    /// Associated algorithm.
    type Algorithm: Algorithm;
    /// Convert to original setting.
    fn base(&self) -> &BasicSetting;
    /// Create the algorithm.
    fn create(self) -> Self::Algorithm;
}

/// The base class of algorithms.
///
/// This type provides a shared dataset if you want to implement a new method.
///
/// Please see [`Algorithm`] for the implementation.
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
    pub(crate) fn new(func: F, setting: &BasicSetting) -> Self {
        let dim = func.lb().len();
        assert_eq!(
            dim,
            func.ub().len(),
            "different dimension of the variables!"
        );
        let pop_num = setting.pop_num;
        Self {
            pop_num,
            dim,
            rpt: setting.rpt,
            task: setting.task.clone(),
            best: Array1::zeros(dim),
            fitness: Array1::ones(pop_num) * f64::INFINITY,
            pool: Array2::zeros((pop_num, dim)),
            report: Report::default(),
            reports: Vec::new(),
            func: Arc::new(func),
        }
    }

    /// Get lower bound.
    #[inline(always)]
    pub fn lb(&self, i: usize) -> f64 {
        self.func.lb()[i]
    }

    /// Get upper bound.
    #[inline(always)]
    pub fn ub(&self, i: usize) -> f64 {
        self.func.ub()[i]
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

/// The methods of the meta-heuristic algorithms.
///
/// + First, use [`setting!`] macro to build a "setting" type.
/// + Second, implement [`Setting`] trait then indicate to a "method" type.
/// + Last, implement `Algorithm` trait on the "method" type.
///
/// Usually, the "method" type that implements this trait will not leak from the API.
/// All most common dataset is store in the [`Context`] type.
/// So the "method" type is used to store the additional data if any.
///
/// ```
/// use metaheuristics_nature::{setting, utility::*, ObjFunc};
///
/// setting! {
///     /// A setting with additional fields.
///     pub struct MySetting1 {
///         base,
///         my_option: u32 = 20,
///     }
/// }
///
/// /// The implementation of the structure with fields.
/// impl Setting for MySetting1 {
///     type Algorithm = Method;
///     fn base(&self) -> &BasicSetting {
///         &self.base
///     }
///     fn create(self) -> Self::Algorithm {
///         Method
///     }
/// }
///
/// setting! {
///     /// Tuple-like setting.
///     pub struct MySetting2(_);
/// }
///
/// /// The implementation of a tuple-like structure.
/// impl Setting for MySetting2 {
///     type Algorithm = Method;
///     fn base(&self) -> &BasicSetting {
///         &self.0
///     }
///     fn create(self) -> Self::Algorithm {
///         Method
///     }
/// }
///
/// pub struct Method;
///
/// impl Algorithm for Method {
///     fn generation<F: ObjFunc>(&mut self, ctx: &mut Context<F>) {
///         unimplemented!()
///     }
/// }
/// ```
///
/// Your algorithm will be implemented by [Solver] automatically.
/// All you have to do is implement the "initialization" method and
/// "generation" method, which are corresponded to the [`Algorithm::init`] and
/// [`Algorithm::generation`] respectively.
pub trait Algorithm {
    /// Initialization implementation.
    ///
    /// The information of the [`Context`] can be obtained or modified at this phase preliminarily.
    ///
    /// The default behavior is do nothing.
    #[inline(always)]
    #[allow(unused_variables)]
    fn init<F: ObjFunc>(&mut self, ctx: &mut Context<F>) {}

    /// Processing implementation of each generation.
    fn generation<F: ObjFunc>(&mut self, ctx: &mut Context<F>);
}

/// Product two iterators together.
///
/// For example, `[a, b, c]` and `[1, 2, 3]` will become `[a1, a2, a3, b1, b2, b3, c1, c2, c3]`.
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
