#[cfg(feature = "std")]
extern crate std;

use crate::{
    utility::{Algorithm, Context, Respond},
    ObjFunc, Report,
};
use alloc::vec::Vec;
use core::marker::PhantomData;
#[cfg(feature = "std")]
use std::time::Instant;

macro_rules! impl_basic_setting {
    ($(fn $name:ident($ty:ty))+) => {$(
        /// Set the base option.
        pub fn $name(mut self, $name: $ty) -> Self {
            self.basic.$name = $name;
            self
        }
    )+};
}

/// Setting base. This type store the basic configurations that provides to the algorithm framework.
///
/// This type should be included in the custom setting, which implements [`Setting`].
#[derive(Debug, PartialEq)]
pub struct BasicSetting {
    /// Termination condition.
    pub task: Task,
    /// Population number.
    pub pop_num: usize,
    /// Report frequency. (per generation)
    pub rpt: u64,
    /// Calculate the average of the fitness at [`Report`](crate::Report).
    /// Default to false.
    pub average: bool,
    /// Threshold of the adaptive factor. Default to disable this function.
    pub adaptive: Adaptive,
}

impl Default for BasicSetting {
    fn default() -> Self {
        Self {
            task: Task::MaxGen(200),
            pop_num: 200,
            rpt: 1,
            average: false,
            adaptive: Adaptive::Disable,
        }
    }
}

/// A trait that provides a conversion to original setting.
///
/// The setting type is actually a builder of the [`Setting::Algorithm`] type.
pub trait Setting {
    /// Associated algorithm.
    ///
    /// This type should implement [`Algorithm`](crate::utility::Algorithm) trait.
    type Algorithm;

    /// Create the algorithm.
    fn algorithm(self) -> Self::Algorithm;

    /// Default basic setting.
    fn default_basic() -> BasicSetting {
        Default::default()
    }
}

/// Terminal condition of the algorithm setting.
#[derive(Debug, PartialEq)]
pub enum Task {
    /// Max generation.
    MaxGen(u64),
    /// Minimum fitness.
    MinFit(f64),
    /// Max time.
    #[cfg(feature = "std")]
    #[cfg_attr(doc_cfg, doc(cfg(feature = "std")))]
    MaxTime(std::time::Duration),
    /// Minimum delta value.
    SlowDown(f64),
}

/// Adaptive factor option.
///
/// The adaptive function will provide a factor for "adaptive penalty factor".
///
/// The factor is calculated by dividing the "feasible individuals" by the "total individuals",
/// where the "feasible individuals" is decided by the threshold.
#[derive(Debug, PartialEq)]
pub enum Adaptive {
    /// Use constant threshold.
    Constant(f64),
    /// Use the average of the finite fitness as threshold.
    Average,
    /// Custom mark from objective function.
    ///
    /// The return type [`ObjFunc::Respond`](crate::ObjFunc::Respond) can be set to `(f64, bool)`.
    ///
    /// See [`Respond`](crate::utility::Respond) for more information.
    Custom,
    /// Disable this option.
    Disable,
}

/// A public API for using optimization methods.
///
/// Users can simply obtain their solution and see the result.
///
/// + The method is a type that implemented [`Algorithm`].
/// + The objective function is a type that implement [`ObjFunc`].
/// + A basic algorithm data is hold by [`Context`].
///
/// The builder of this type can infer the algorithm by [`Setting::Algorithm`].
///
/// ```
/// use metaheuristics_nature::{Rga, Solver, Task};
/// # use metaheuristics_nature::{ObjFunc, Report};
/// # struct MyFunc([f64; 3], [f64; 3]);
/// # impl MyFunc {
/// #     fn new() -> Self { Self([0.; 3], [50.; 3]) }
/// # }
/// # impl ObjFunc for MyFunc {
/// #     type Result = f64;
/// #     type Respond = f64;
/// #     fn fitness(&self, v: &[f64], _: &Report) -> Self::Respond {
/// #         v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
/// #     }
/// #     fn result(&self, v: &[f64]) -> Self::Result {
/// #         self.fitness(v, &Default::default())
/// #     }
/// #     fn ub(&self) -> &[f64] { &self.1 }
/// #     fn lb(&self) -> &[f64] { &self.0 }
/// # }
///
/// let s = Solver::build(Rga::default())
///     .task(Task::MinFit(1e-20))
///     .solve(MyFunc::new(), |_| true); // Run without callback
/// // Get the result from objective function
/// let ans = s.result();
/// // Get the optimized XY value of your function
/// let x = s.best_parameters();
/// let y = s.best_fitness();
/// // Get the history reports
/// let reports = s.reports();
/// ```
pub struct Solver<F: ObjFunc>(Context<F>);

/// Collect configuration and build the solver.
///
/// This type is created by [`Solver::build`] method.
pub struct SolverBuilder<S: Setting, F: ObjFunc> {
    basic: BasicSetting,
    setting: S,
    _phantom: PhantomData<F>,
}

impl<S: Setting, F: ObjFunc> SolverBuilder<S, F> {
    impl_basic_setting! {
        fn task(Task)
        fn pop_num(usize)
        fn rpt(u64)
        fn average(bool)
        fn adaptive(Adaptive)
    }

    /// Create the task and run the algorithm, which may takes a lot of time.
    ///
    /// Argument `callback` is a progress feedback function,
    /// returns true to keep algorithm running, same as the behavior of the while-loop.
    pub fn solve(self, func: F, mut callback: impl FnMut(&Report) -> bool) -> Solver<F>
    where
        S::Algorithm: Algorithm<F>,
    {
        let mut method = self.setting.algorithm();
        let mut ctx = Context::new(func, self.basic);
        #[cfg(feature = "std")]
        let time_start = Instant::now();
        ctx.init_pop();
        #[cfg(feature = "std")]
        let _ = { ctx.report.time = (Instant::now() - time_start).as_secs_f64() };
        method.init(&mut ctx);
        if !callback(&ctx.report) {
            return Solver(ctx);
        }
        ctx.report();
        loop {
            ctx.report.gen += 1;
            #[cfg(feature = "std")]
            let _ = { ctx.report.time = (Instant::now() - time_start).as_secs_f64() };
            let best_f = ctx.report.best_f;
            let diff = ctx.report.diff;
            method.generation(&mut ctx);
            ctx.report.diff = best_f - ctx.report.best_f;
            if ctx.average || ctx.adaptive == Adaptive::Average {
                let mut average = 0.;
                let mut count = 0;
                for v in ctx.fitness.iter().filter(|v| v.value().is_finite()) {
                    average += v.value();
                    count += 1;
                }
                ctx.report.average = average / count as f64;
            }
            if ctx.adaptive != Adaptive::Disable {
                let iter = ctx.fitness.iter();
                let feasible = match ctx.adaptive {
                    Adaptive::Constant(ada) => iter.filter(|f| f.value() > ada).count(),
                    Adaptive::Average => iter.filter(|f| f.value() > ctx.report.average).count(),
                    Adaptive::Custom => iter.filter(|f| f.feasible()).count(),
                    Adaptive::Disable => unreachable!(),
                };
                ctx.report.adaptive = feasible as f64 / ctx.pop_num() as f64;
            }
            if ctx.report.gen % ctx.rpt == 0 {
                if !callback(&ctx.report) {
                    break;
                }
                ctx.report();
            }
            match ctx.task {
                Task::MaxGen(v) => {
                    if ctx.report.gen >= v {
                        break;
                    }
                }
                Task::MinFit(v) => {
                    if ctx.report.best_f <= v {
                        break;
                    }
                }
                #[cfg(feature = "std")]
                Task::MaxTime(d) => {
                    if Instant::now() - time_start >= d {
                        break;
                    }
                }
                Task::SlowDown(v) => {
                    if ctx.report.diff / diff >= v {
                        break;
                    }
                }
            }
        }
        Solver(ctx)
    }
}

impl<F: ObjFunc> Solver<F> {
    /// Build the solver.
    pub fn build<S: Setting>(setting: S) -> SolverBuilder<S, F> {
        SolverBuilder {
            basic: S::default_basic(),
            setting,
            _phantom: PhantomData,
        }
    }

    /// Get the reference of the objective function.
    ///
    /// It's useful when you need to get the preprocessed data from the initialization process,
    /// which is stored in the objective function.
    #[inline(always)]
    pub fn func(&self) -> &F {
        &self.0.func
    }

    /// Get the history for plotting.
    #[inline(always)]
    pub fn reports(&self) -> Vec<Report> {
        self.0.reports.clone()
    }

    /// Get the best parameters.
    #[inline(always)]
    pub fn best_parameters(&self) -> &[f64] {
        self.0.best.as_slice().unwrap()
    }

    /// Get the best fitness.
    #[inline(always)]
    pub fn best_fitness(&self) -> f64 {
        self.0.report.best_f
    }

    /// Get the result of the objective function.
    #[inline(always)]
    pub fn result(&self) -> F::Result {
        self.0.func.result(self.0.best.as_slice().unwrap())
    }
}
