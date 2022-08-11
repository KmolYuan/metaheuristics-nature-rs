use crate::utility::prelude::*;
use alloc::{boxed::Box, vec::Vec};

fn assert_shape(b: bool) -> Result<(), ShapeError> {
    b.then_some(())
        .ok_or_else(|| ShapeError::from_kind(ErrorKind::IncompatibleShape))
}

enum Pool<'a, F: ObjFunc> {
    ReadyMade {
        pool: Array2<f64>,
        fitness: Vec<F::Fitness>,
    },
    Func(Box<dyn FnOnce(&Context<F>) -> Array2<f64> + 'a>),
}

/// Collect configuration and build the solver.
///
/// This type is created by [`Solver::build`] method.
#[must_use = "solver builder do nothing unless call the \"solve\" method"]
pub struct SolverBuilder<'a, S: Setting, F: ObjFunc, R> {
    pop_num: usize,
    seed: Option<u128>,
    setting: S,
    pool: Pool<'a, F>,
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

    /// Give a pool generating function.
    ///
    /// You can insert a ready-made pool from last states,
    /// or random values with another distribution.
    ///
    /// The array must be the shape of `ctx.pool_size()` and in the bounds of `[ctx.lb(), ctx.ub())`.
    ///
    /// ```
    /// use metaheuristics_nature::{utility::gaussian_pool, Rga, Solver};
    /// # use metaheuristics_nature::tests::TestObj as MyFunc;
    ///
    /// let s = Solver::build(Rga::default())
    /// #   .task(|ctx| ctx.gen == 1)
    ///     .pool(gaussian_pool(&[0.; 4], &[5.; 4]))
    ///     .solve(MyFunc::new())
    ///     .unwrap();
    /// ```
    ///
    /// # Default
    ///
    /// By default, the pool will generate with uniform distribution in the bounds.
    /// ([`uniform_pool`])
    ///
    /// # See Also
    ///
    /// [`Self::pool_and_fitness`], [`uniform_pool`], [`gaussian_pool`].
    pub fn pool<'b, C>(self, pool: C) -> SolverBuilder<'b, S, F, R>
    where
        'a: 'b,
        C: FnOnce(&Context<F>) -> Array2<f64> + 'b,
    {
        SolverBuilder { pool: Pool::Func(Box::new(pool)), ..self }
    }

    /// Give a ready-made pool and its fitness values directly.
    ///
    /// The `pool` must be the shape of `ctx.pool_size()` and in the bounds of `[ctx.lb(), ctx.ub())`,
    /// and the `fitness` must have the same length as `ctx.pop_num()`.
    ///
    /// # Default
    ///
    /// The default pool will generate with uniform distribution in the bounds.
    ///
    /// # See Also
    ///
    /// [`Self::pool`].
    pub fn pool_and_fitness(self, pool: Array2<f64>, fitness: Vec<F::Fitness>) -> Self {
        Self { pool: Pool::ReadyMade { pool, fitness }, ..self }
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
    ///     .solve(MyFunc::new())
    ///     .unwrap();
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
        SolverBuilder { task: Box::new(task), ..self }
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
    /// #   .task(|ctx| ctx.gen == 1)
    ///     .record(|ctx| (ctx.gen, ctx.adaptive))
    ///     .solve(MyFunc::new())
    ///     .unwrap();
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
        macro_rules! builder {
            ($field_new:ident, $($field:ident),+) => {
                SolverBuilder { $field_new: Box::new($field_new), $($field: self.$field),+ }
            };
        }
        builder!(record, pop_num, seed, setting, pool, task, adaptive, callback)
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
    /// #   .task(|ctx| ctx.gen == 1)
    ///     .adaptive(|ctx| ctx.gen as f64 / 20.)
    ///     .solve(MyFunc::new())
    ///     .unwrap();
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
    /// #   .task(|ctx| ctx.gen == 1)
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
    ///     .solve(MyFunc::new())
    ///     .unwrap();
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
        SolverBuilder { adaptive: Box::new(adaptive), ..self }
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
    /// #   .task(|ctx| ctx.gen == 1)
    ///     .callback(|ctx| gen = ctx.gen)
    ///     .solve(MyFunc::new())
    ///     .unwrap();
    /// ```
    ///
    /// In the example below, the fields of the `app` are mutable variables that changes every time.
    /// But we still need to use its method in [`task`](Self::task) condition,
    /// so a [`RwLock`](std::sync::RwLock) / [`Mutex`](std::sync::Mutex) lock / [`std::sync::atomic`] is required.
    ///
    /// If you spawn the optimization process into another thread,
    /// adding a reference counter ([`Arc`](std::sync::Arc)) is also required.
    ///
    /// ```
    /// use metaheuristics_nature::{Rga, Solver};
    /// use std::sync::{
    ///     atomic::{AtomicBool, AtomicU64, Ordering},
    ///     Arc, Mutex,
    /// };
    /// # use metaheuristics_nature::tests::TestObj as MyFunc;
    ///
    /// #[derive(Default)]
    /// struct App {
    ///     is_start: Arc<AtomicBool>,
    ///     gen: Arc<AtomicU64>,
    ///     fitness: Arc<Mutex<f64>>,
    /// }
    ///
    /// let app = App::default();
    /// // Create references of Arc,
    /// // they will be moved into a static closure
    /// let is_start = app.is_start.clone();
    /// let gen = app.gen.clone();
    /// let fitness = app.fitness.clone();
    /// // Spawn the solver here!
    /// let handle = std::thread::spawn(move || {
    ///     Solver::build(Rga::default())
    ///         .task(|ctx| ctx.gen == 20 || !is_start.load(Ordering::Relaxed))
    ///         .callback(|ctx| {
    ///             gen.store(ctx.gen, Ordering::Relaxed);
    ///             *fitness.lock().unwrap() = ctx.best_f;
    ///         })
    ///         .solve(MyFunc::new())
    ///         .unwrap()
    /// });
    /// /* do other things such as GUI */
    /// let s = handle.join();
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
        SolverBuilder { callback: Box::new(callback), ..self }
    }

    /// Create the task and run the algorithm, which may takes a lot of time.
    ///
    /// Generation `ctx.gen` is start from 1, initialized at 0.
    ///
    /// # Error
    ///
    /// This method returns a `Result` object.
    /// It will be `Ok` and returns result when the `ctx.pool` and `ctx.fitness`
    /// initialized successfully;
    /// `Err` when the boundary check fails.
    pub fn solve(self, func: F) -> Result<Solver<F, R>, ShapeError>
    where
        S::Algorithm: Algorithm<F>,
    {
        let Self {
            pop_num,
            seed,
            setting,
            pool,
            task,
            record,
            mut adaptive,
            mut callback,
        } = self;
        assert_shape(func.bound().iter().all(|[lb, ub]| lb < ub))?;
        let mut method = setting.algorithm();
        let mut ctx = Context::new(func, seed, pop_num);
        let mut report = Vec::new();
        match pool {
            Pool::ReadyMade { pool, fitness } => {
                ctx.pool = pool;
                ctx.fitness = fitness;
                assert_shape(ctx.pool.shape() == ctx.pool_size())?;
            }
            Pool::Func(f) => {
                let pool = f(&ctx);
                assert_shape(pool.shape() == ctx.pool_size())?;
                ctx.init_pop(pool);
            }
        }
        loop {
            ctx.adaptive = adaptive(&ctx);
            if ctx.gen == 0 {
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
        Ok(Solver::new(ctx, report))
    }
}

impl<F: ObjFunc + 'static> Solver<F, ()> {
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
            pool: Pool::Func(Box::new(uniform_pool)),
            task: Box::new(|ctx| ctx.gen >= 200),
            record: Box::new(|_| ()),
            adaptive: Box::new(|_| 1.),
            callback: Box::new(|_| ()),
        }
    }
}

/// A function generates a uniform pool.
///
/// Please see [`SolverBuilder::pool`] for more information.
pub fn uniform_pool<F: ObjFunc>(ctx: &Context<F>) -> Array2<f64> {
    Array2::from_shape_fn(ctx.pool_size(), |(_, s)| {
        ctx.rng.float(ctx.lb(s)..ctx.ub(s))
    })
}

/// A function generates a Gaussian pool.
///
/// Where `mean` is the mean value, `std` is the standard deviation.
///
/// Please see [`SolverBuilder::pool`] for more information.
///
/// # Panics
///
/// Panic when the lengths of `mean` and `std` are not the same.
pub fn gaussian_pool<'a, F: ObjFunc>(
    mean: &'a [f64],
    std: &'a [f64],
) -> impl Fn(&Context<F>) -> Array2<f64> + 'a {
    assert_eq!(mean.len(), std.len());
    move |ctx| {
        Array2::from_shape_fn(ctx.pool_size(), |(_, s)| {
            ctx.rng
                .rand_norm(mean[s], std[s])
                .clamp(ctx.lb(s), ctx.ub(s))
        })
    }
}

/// A function generates a Gaussian pool, including the mean (center).
///
/// Where `mean` is the mean value, `std` is the standard deviation.
///
/// Please see [`SolverBuilder::pool`] for more information.
///
/// # Panics
///
/// Panic when the lengths of `mean` and `std` are not the same.
pub fn gaussian_pool_inclusive<'a, F: ObjFunc>(
    mean: &'a [f64],
    std: &'a [f64],
) -> impl Fn(&Context<F>) -> Array2<f64> + 'a {
    assert_eq!(mean.len(), std.len());
    move |ctx| {
        Array2::from_shape_fn(ctx.pool_size(), |(i, s)| {
            if i == 0 {
                mean[s]
            } else {
                ctx.rng
                    .rand_norm(mean[s], std[s])
                    .clamp(ctx.lb(s), ctx.ub(s))
            }
        })
    }
}
