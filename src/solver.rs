use crate::{
    utility::{Algorithm, Context, Respond},
    ObjFunc,
};
use alloc::{boxed::Box, vec::Vec};
#[cfg(feature = "std")]
use std::time::Instant;

macro_rules! impl_basic_setting {
    ($($(#[$meta:meta])* fn $name:ident($ty:ty))+) => {$(
        $(#[$meta])*
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
    pub(crate) task: Task,
    pub(crate) pop_num: usize,
    pub(crate) rpt: u64,
    pub(crate) average: bool,
    pub(crate) adaptive: Adaptive,
    pub(crate) seed: Option<u128>,
}

impl Default for BasicSetting {
    fn default() -> Self {
        Self {
            task: Task::MaxGen(200),
            pop_num: 200,
            rpt: 1,
            average: false,
            adaptive: Adaptive::Disable,
            seed: None,
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
/// # use metaheuristics_nature::tests::TestObj as MyFunc;
///
/// // Build and run the solver
/// let s = Solver::build(Rga::default())
///     .task(Task::MaxGen(20))
///     .solve(MyFunc::new());
/// // Get the result from objective function
/// let ans = s.result();
/// // Get the optimized XY value of your function
/// let x = s.best_parameters();
/// let y = s.best_fitness();
/// // Get the history reports
/// let report = s.report();
/// ```
pub struct Solver<F: ObjFunc, R> {
    ctx: Context<F>,
    report: Vec<R>,
}

/// Collect configuration and build the solver.
///
/// This type is created by [`Solver::build`] method.
#[must_use = "solver builder do nothing unless call the \"solve\" method"]
pub struct SolverBuilder<'a, S: Setting, F: ObjFunc, R> {
    basic: BasicSetting,
    setting: S,
    record: Box<dyn Fn(&Context<F>) -> R + 'a>,
    callback: Box<dyn FnMut(&R) -> bool + 'a>,
}

impl<'a, S, F, R> SolverBuilder<'a, S, F, R>
where
    S: Setting,
    F: ObjFunc,
    S::Algorithm: Algorithm<F>,
{
    impl_basic_setting! {
        /// Termination condition.
        ///
        /// # Default
        ///
        /// By default, the algorithm will iterate 200 generation.
        fn task(Task)

        /// Population number.
        ///
        /// # Default
        ///
        /// If not changed by the algorithm setting, the default number is 200.
        fn pop_num(usize)

        /// Report frequency. (per generation)
        ///
        /// # Default
        ///
        /// By default, each generation will be reported.
        fn rpt(u64)

        /// Calculate the average of the fitness at [`Report`](crate::Report). Default to false.
        fn average(bool)

        /// Threshold of the adaptive factor.
        ///
        /// # Default
        ///
        /// By default, this function is disabled.
        fn adaptive(Adaptive)

        /// Set random seed.
        ///
        /// # Default
        ///
        /// By default, the random seed is `None`, which is decided by [`getrandom::getrandom`].
        fn seed(Option<u128>)
    }

    /// Set record function.
    ///
    /// The record function will be called at each generation and save the return value in the report.
    /// Due to memory allocation, this function should record as less information as possible.
    /// For example, return unit type `()` can totally disable this function.
    ///
    /// After calling [`solve`](Self::solve) function, you can take the report value with [`Solver::report`] method.
    ///
    /// # Default
    ///
    /// By default, the record function returns generation (`u64`) and best fitness (`f64`).
    pub fn record<'b, C>(self, record: C) -> SolverBuilder<'b, S, F, R>
    where
        'a: 'b,
        C: Fn(&Context<F>) -> R + 'b,
    {
        SolverBuilder {
            basic: self.basic,
            setting: self.setting,
            record: Box::new(record),
            callback: self.callback,
        }
    }

    /// Set callback function.
    ///
    /// The return value of the callback controls when to break the iteration.
    ///
    /// Return false to break, same as the while loop condition.
    ///
    /// In the example below, `app` is a mutable variable that changes every time.
    ///
    /// ```
    /// use metaheuristics_nature::{Rga, Solver, Task};
    /// # use metaheuristics_nature::tests::TestObj as MyFunc;
    /// # struct App;
    /// # impl App {
    /// #     fn show_generation(&mut self, _gen: u64) {}
    /// #     fn show_fitness(&mut self, _f: f64) {}
    /// #     fn is_stop(&self) -> bool { false }
    /// # }
    /// # let mut app = App;
    ///
    /// let s = Solver::build(Rga::default())
    ///     .task(Task::MaxGen(20))
    ///     .callback(|&(gen, fitness)| {
    ///         app.show_generation(gen);
    ///         app.show_fitness(fitness);
    ///         !app.is_stop()
    ///     })
    ///     .solve(MyFunc::new());
    /// ```
    ///
    /// # Default
    ///
    /// By default, the callback function will not break the iteration and does nothing.
    pub fn callback<'b, C>(self, callback: C) -> SolverBuilder<'b, S, F, R>
    where
        'a: 'b,
        C: FnMut(&R) -> bool + 'b,
    {
        SolverBuilder {
            basic: self.basic,
            setting: self.setting,
            record: self.record,
            callback: Box::new(callback),
        }
    }

    /// Create the task and run the algorithm, which may takes a lot of time.
    pub fn solve(self, func: F) -> Solver<F, R> {
        let mut method = self.setting.algorithm();
        let mut ctx = Context::new(func, self.basic);
        let record = self.record;
        let mut callback = self.callback;
        let mut report = Vec::new();
        #[cfg(feature = "std")]
        let time_start = Instant::now();
        ctx.init_pop();
        #[cfg(feature = "std")]
        let _ = { ctx.time = (Instant::now() - time_start).as_secs_f64() };
        method.init(&mut ctx);
        let r = record(&ctx);
        if !callback(&r) {
            return Solver { ctx, report };
        }
        report.push(r);
        loop {
            ctx.report.gen += 1;
            #[cfg(feature = "std")]
            let _ = { ctx.time = (Instant::now() - time_start).as_secs_f64() };
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
                let r = record(&ctx);
                if !callback(&r) {
                    break;
                }
                report.push(r);
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
        Solver { ctx, report }
    }
}

impl<F: ObjFunc> Solver<F, (u64, f64)> {
    /// Start to build a solver. Take a setting and setup the configurations.
    ///
    /// Please check [`SolverBuilder`] type, it will help you choose your configuration.
    ///
    /// If all things are well-setup, call [`SolverBuilder::solve`].
    ///
    /// # Defaults
    ///
    /// + The basic setting is generate by [`Setting::default_basic`].
    /// + `record` function returns generation (`u64`) and best fitness (`f64`).
    /// + `callback` function will not break the iteration and does nothing.
    pub fn build<S>(setting: S) -> SolverBuilder<'static, S, F, (u64, f64)>
    where
        S: Setting,
    {
        SolverBuilder {
            basic: S::default_basic(),
            setting,
            record: Box::new(|ctx| (ctx.report.gen, ctx.report.best_f)),
            callback: Box::new(|_| true),
        }
    }
}

impl<F: ObjFunc, R> Solver<F, R> {
    /// Get the reference of the objective function.
    ///
    /// It's useful when you need to get the preprocessed data from the initialization process,
    /// which is stored in the objective function.
    #[inline(always)]
    pub fn func(&self) -> &F {
        &self.ctx.func
    }

    /// Get the history for plotting.
    #[inline(always)]
    pub fn report(&self) -> &[R] {
        &self.report
    }

    /// Get the best parameters.
    #[inline(always)]
    pub fn best_parameters(&self) -> &[f64] {
        self.ctx.best.as_slice().unwrap()
    }

    /// Get the best fitness.
    #[inline(always)]
    pub fn best_fitness(&self) -> f64 {
        self.ctx.report.best_f
    }

    /// Get the result of the objective function.
    #[inline(always)]
    pub fn result(&self) -> F::Result {
        self.ctx.func.result(self.ctx.best.as_slice().unwrap())
    }
}
