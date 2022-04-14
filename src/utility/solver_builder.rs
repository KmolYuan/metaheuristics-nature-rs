use crate::utility::prelude::*;
use alloc::{boxed::Box, vec::Vec};

/// Collect configuration and build the solver.
///
/// This type is created by [`Solver::build`] method.
#[must_use = "solver builder do nothing unless call the \"solve\" method"]
pub struct SolverBuilder<'a, S: Setting, F: ObjFunc, R> {
    pop_num: usize,
    seed: Option<u128>,
    setting: S,
    task: Box<dyn Fn(&Context<F>) -> bool + 'a>,
    record: Box<dyn Fn(&Context<F>) -> R + 'a>,
    adaptive: Box<dyn FnMut(&Context<F>) -> f64 + 'a>,
    callback: Box<dyn FnMut(&Context<F>) + 'a>,
}

impl<'a, S, F, R> SolverBuilder<'a, S, F, R>
where
    S: Setting,
    F: ObjFunc,
{
    impl_builders! {
        /// Population number.
        ///
        /// # Default
        ///
        /// If not changed by the algorithm setting, the default number is 200.
        fn pop_num(usize)

        /// Set random seed.
        ///
        /// # Default
        ///
        /// By default, the random seed is `None`, which is decided by [`getrandom::getrandom`].
        fn seed(Option<u128>)
    }

    /// Termination condition.
    ///
    /// The task function will be check each iteration, breaks if the return is true.
    ///
    /// ```
    /// use metaheuristics_nature::{Rga, Solver};
    /// # use metaheuristics_nature::tests::TestObj as MyFunc;
    ///
    /// let s = Solver::build(Rga::default())
    ///     .task(|ctx| ctx.gen == 20)
    ///     .solve(MyFunc::new());
    /// ```
    ///
    /// # Default
    ///
    /// By default, the algorithm will iterate 200 generation.
    pub fn task<'b, C>(self, task: C) -> SolverBuilder<'b, S, F, R>
    where
        'a: 'b,
        C: Fn(&Context<F>) -> bool + 'b,
    {
        SolverBuilder {
            task: Box::new(task),
            ..self
        }
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
    /// #   .task(|ctx| ctx.gen == 20)
    ///     .record(|ctx| (ctx.gen, ctx.adaptive))
    ///     .solve(MyFunc::new());
    /// let report: &[(u64, f64)] = s.report();
    /// ```
    ///
    /// # Default
    ///
    /// By default, this function returns unit type `()`, which allocates nothing.
    pub fn record<'b, C, NR>(self, record: C) -> SolverBuilder<'b, S, F, NR>
    where
        'a: 'b,
        C: Fn(&Context<F>) -> NR + 'b,
    {
        SolverBuilder {
            pop_num: self.pop_num,
            seed: self.seed,
            setting: self.setting,
            task: self.task,
            record: Box::new(record),
            adaptive: self.adaptive,
            callback: self.callback,
        }
    }

    /// Set adaptive function.
    ///
    /// The adaptive value can be access from [`ObjFunc::fitness`],
    /// and can be used to enhance the fitness value.
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
    /// By default, this function returns one.
    pub fn adaptive<'b, C>(self, adaptive: C) -> SolverBuilder<'b, S, F, R>
    where
        'a: 'b,
        C: FnMut(&Context<F>) -> f64 + 'b,
    {
        SolverBuilder {
            adaptive: Box::new(adaptive),
            ..self
        }
    }

    /// Set callback function.
    ///
    /// Callback function allows to change an outer mutable variable in each iteration.
    ///
    /// ```
    /// use metaheuristics_nature::{Rga, Solver};
    /// # use metaheuristics_nature::tests::TestObj as MyFunc;
    ///
    /// let mut gen = 0;
    /// let s = Solver::build(Rga::default())
    /// #   .task(|ctx| ctx.gen == 20)
    ///     .callback(|ctx| gen = ctx.gen)
    ///     .solve(MyFunc::new());
    /// ```
    ///
    /// In the example below, the fields of the `app` are mutable variables that changes every time.
    /// But we still need to use its method in [`task`](Self::task) condition,
    /// so a [`RwLock`](std::sync::RwLock) / [`Mutex`](std::sync::Mutex) lock / [`std::sync::atomic`] is required.
    ///
    /// If you spawn the optimization process into another thread, adding a reference counter is also required.
    ///
    /// ```
    /// use metaheuristics_nature::{Rga, Solver};
    /// use std::sync::{
    ///     atomic::{AtomicBool, AtomicU64, Ordering},
    ///     Mutex,
    /// };
    /// # use metaheuristics_nature::tests::TestObj as MyFunc;
    ///
    /// #[derive(Default)]
    /// struct App {
    ///     is_start: AtomicBool,
    ///     gen: AtomicU64,
    ///     fitness: Mutex<f64>,
    /// }
    ///
    /// let app = App::default();
    /// // Spawn the solver here!
    /// let s = Solver::build(Rga::default())
    ///     .task(|ctx| ctx.gen == 20 || !app.is_start.load(Ordering::Relaxed))
    ///     .callback(|ctx| {
    ///         app.gen.store(ctx.gen, Ordering::Relaxed);
    ///         *app.fitness.lock().unwrap() = ctx.best_f;
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
            callback: Box::new(callback),
            ..self
        }
    }

    /// Create the task and run the algorithm, which may takes a lot of time.
    #[must_use = "the result cannot access unless to store the solver"]
    pub fn solve(self, func: F) -> Solver<F, R>
    where
        S::Algorithm: Algorithm<F>,
    {
        let Self {
            pop_num,
            seed,
            setting,
            task,
            record,
            mut adaptive,
            mut callback,
        } = self;
        let mut method = setting.algorithm();
        let mut ctx = Context::new(func, seed, pop_num);
        let mut report = Vec::new();
        loop {
            ctx.adaptive = adaptive(&ctx);
            if ctx.gen == 0 {
                ctx.init_pop();
                method.init(&mut ctx);
            } else {
                method.generation(&mut ctx);
            }
            report.push(record(&ctx));
            callback(&ctx);
            if task(&ctx) {
                break;
            }
            ctx.gen += 1;
        }
        Solver::new(ctx, report)
    }
}

impl<F: ObjFunc> Solver<F, ()> {
    /// Start to build a solver. Take a setting and setup the configurations.
    ///
    /// Please check [`SolverBuilder`] type, it will help you choose your configuration.
    ///
    /// If all things are well-setup, call [`SolverBuilder::solve`].
    ///
    /// The default value of each option can be found in their document.
    pub fn build<S>(setting: S) -> SolverBuilder<'static, S, F, ()>
    where
        S: Setting,
    {
        SolverBuilder {
            pop_num: S::default_pop(),
            seed: None,
            setting,
            task: Box::new(|ctx| ctx.gen >= 200),
            record: Box::new(|_| ()),
            adaptive: Box::new(|_| 1.),
            callback: Box::new(|_| ()),
        }
    }
}
