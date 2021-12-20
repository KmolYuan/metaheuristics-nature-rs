use crate::utility::prelude::*;
use alloc::{boxed::Box, vec::Vec};

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
    pub(crate) pop_num: usize,
    pub(crate) rpt: u64,
    pub(crate) seed: Option<u128>,
}

impl Default for BasicSetting {
    fn default() -> Self {
        Self {
            pop_num: 200,
            rpt: 1,
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
/// use metaheuristics_nature::{Rga, Solver};
/// # use metaheuristics_nature::tests::TestObj as MyFunc;
///
/// // Build and run the solver
/// let s = Solver::build(Rga::default())
///     .task(|ctx| ctx.gen == 20)
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
    task: Box<dyn Fn(&Context<F>) -> bool + 'static>,
    record: Box<dyn Fn(&Context<F>) -> R + 'static>,
    adaptive: Box<dyn FnMut(&Context<F>) -> f64 + 'a>,
    callback: Box<dyn FnMut(&Context<F>) + 'a>,
}

impl<'a, S, F, R> SolverBuilder<'a, S, F, R>
where
    S: Setting,
    F: ObjFunc,
    S::Algorithm: Algorithm<F>,
{
    impl_basic_setting! {
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

        /// Set random seed.
        ///
        /// # Default
        ///
        /// By default, the random seed is `None`, which is decided by [`getrandom::getrandom`].
        fn seed(Option<u128>)
    }

    /// Termination condition.
    ///
    /// The task function will be check every generation, break if it returns true.
    ///
    /// # Default
    ///
    /// By default, the algorithm will iterate 200 generation.
    pub fn task<C>(mut self, task: C) -> Self
    where
        C: Fn(&Context<F>) -> bool + 'static,
    {
        self.task = Box::new(task);
        self
    }

    /// Set record function.
    ///
    /// The record function will be called at each generation and save the return value in the report.
    /// Due to memory allocation, this function should record as less information as possible.
    /// For example, return unit type `()` can totally disable this function.
    ///
    /// After calling [`solve`](Self::solve) function, you can take the report value with [`Solver::report`] method.
    /// The following example records generation and spent time for the report.
    ///
    /// ```
    /// use metaheuristics_nature::{Rga, Solver};
    /// # use metaheuristics_nature::tests::TestObj as MyFunc;
    ///
    /// let s = Solver::build(Rga::default())
    ///     .task(|ctx| ctx.gen == 20)
    ///     .record(|ctx| (ctx.gen, ctx.adaptive))
    ///     .solve(MyFunc::new());
    /// let report: &[(u64, f64)] = s.report();
    /// ```
    ///
    /// # Default
    ///
    /// By default, this function returns generation (`u64`) and best fitness (`f64`).
    pub fn record<C, NR>(self, record: C) -> SolverBuilder<'a, S, F, NR>
    where
        C: Fn(&Context<F>) -> NR + 'static,
    {
        SolverBuilder {
            basic: self.basic,
            setting: self.setting,
            task: self.task,
            record: Box::new(record),
            adaptive: self.adaptive,
            callback: self.callback,
        }
    }

    /// Set adaptive function.
    ///
    /// The adaptive value can be access from [`ObjFunc::fitness`].
    ///
    /// ```
    /// use metaheuristics_nature::{Rga, Solver};
    /// # use metaheuristics_nature::tests::TestObj as MyFunc;
    ///
    /// let s = Solver::build(Rga::default())
    /// #   .task(|ctx| ctx.gen == 20)
    ///     .adaptive(|ctx| ctx.gen as f64 / 20.)
    ///     .solve(MyFunc::new());
    /// ```
    ///
    /// The adaptive function is also allow to change the external variable.
    ///
    /// ```
    /// use metaheuristics_nature::{Rga, Solver};
    /// # use metaheuristics_nature::tests::TestObj as MyFunc;
    ///
    /// let mut diff = None;
    /// let s = Solver::build(Rga::default())
    /// #   .task(|ctx| ctx.gen == 20)
    ///     .adaptive(|ctx| {
    ///         if let Some(f) = diff {
    ///             let d = f - ctx.best_f;
    ///             diff = Some(d);
    ///             d
    ///         } else {
    ///             diff = Some(ctx.best_f);
    ///             ctx.best_f
    ///         }
    ///     })
    ///     .solve(MyFunc::new());
    /// ```
    ///
    /// # Default
    ///
    /// By default, this function returns zero.
    pub fn adaptive<'b, C>(self, adaptive: C) -> SolverBuilder<'b, S, F, R>
    where
        'a: 'b,
        C: FnMut(&Context<F>) -> f64 + 'b,
    {
        SolverBuilder {
            basic: self.basic,
            setting: self.setting,
            task: self.task,
            record: self.record,
            adaptive: Box::new(adaptive),
            callback: self.callback,
        }
    }

    /// Set callback function.
    ///
    /// In the example below, `app` is a mutable variable that changes every time.
    /// But we still need to use its method in [`task`](Self::task) condition,
    /// so a [`RwLock`](std::sync::RwLock) / [`Mutex`](std::sync::Mutex) lock within a reference counter is required.
    ///
    /// ```
    /// use metaheuristics_nature::{Rga, Solver};
    /// use std::{rc::Rc, sync::RwLock};
    /// # use metaheuristics_nature::tests::TestObj as MyFunc;
    /// # struct App;
    /// # impl App {
    /// #     fn show_generation(&mut self, _gen: u64) {}
    /// #     fn show_fitness(&mut self, _f: f64) {}
    /// #     fn is_stop(&self) -> bool { false }
    /// # }
    ///
    /// let app = Rc::new(RwLock::new(App));
    /// let app1 = app.clone();
    /// let app2 = app.clone();
    /// let s = Solver::build(Rga::default())
    ///     .task(move |ctx| ctx.gen == 20 || app1.read().unwrap().is_stop())
    ///     .callback(move |ctx| {
    ///         let mut app = app2.write().unwrap();
    ///         app.show_generation(ctx.gen);
    ///         app.show_fitness(ctx.best_f);
    ///     })
    ///     .solve(MyFunc::new());
    /// ```
    ///
    /// # Default
    ///
    /// By default, this function does nothing.
    pub fn callback<'b, C>(self, callback: C) -> SolverBuilder<'b, S, F, R>
    where
        'a: 'b,
        C: FnMut(&Context<F>) + 'b,
    {
        SolverBuilder {
            basic: self.basic,
            setting: self.setting,
            task: self.task,
            record: self.record,
            adaptive: self.adaptive,
            callback: Box::new(callback),
        }
    }

    /// Create the task and run the algorithm, which may takes a lot of time.
    #[must_use = "the result cannot access unless to store the solver"]
    pub fn solve(self, func: F) -> Solver<F, R> {
        let rpt = self.basic.rpt;
        assert!(rpt > 0, "report interval should not be zero");
        let mut method = self.setting.algorithm();
        let mut ctx = Context::new(func, self.basic);
        let task = self.task;
        let record = self.record;
        let mut adaptive = self.adaptive;
        let mut callback = self.callback;
        let mut report = Vec::new();
        loop {
            ctx.adaptive = adaptive(&ctx);
            if ctx.gen == 0 {
                ctx.init_pop();
                method.init(&mut ctx);
            } else {
                method.generation(&mut ctx);
            }
            if ctx.gen % rpt == 0 {
                report.push(record(&ctx));
            }
            callback(&ctx);
            if task(&ctx) {
                break;
            }
            ctx.gen += 1;
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
    /// + `adaptive` function returns zero.
    /// + `record` function returns generation (`u64`) and best fitness (`f64`).
    /// + `callback` function will not break the iteration and does nothing.
    pub fn build<S>(setting: S) -> SolverBuilder<'static, S, F, (u64, f64)>
    where
        S: Setting,
    {
        SolverBuilder {
            basic: S::default_basic(),
            setting,
            task: Box::new(|ctx| ctx.gen >= 200),
            record: Box::new(|ctx| (ctx.gen, ctx.best_f.value())),
            adaptive: Box::new(|_| 0.),
            callback: Box::new(|_| ()),
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
        self.ctx.best_f.value()
    }

    /// Get the result of the objective function.
    #[inline(always)]
    pub fn result(&self) -> F::Result {
        self.func().result(self.best_parameters())
    }
}
