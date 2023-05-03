use crate::utility::prelude::*;
use alloc::{boxed::Box, vec::Vec};

type PoolFunc<'a, F> = Box<dyn FnOnce(&Ctx<F>, &Rng) -> Array2<f64> + Send + 'a>;
type TaskFunc<'a, F> = Box<dyn Fn(&Ctx<F>) -> bool + Send + 'a>;
type CallbackFunc<'a, F> = Box<dyn FnMut(&mut Ctx<F>) + Send + 'a>;

fn assert_shape(b: bool) -> Result<(), ndarray::ShapeError> {
    b.then_some(())
        .ok_or_else(|| ndarray::ShapeError::from_kind(ndarray::ErrorKind::IncompatibleShape))
}

enum Pool<'a, F: ObjFunc> {
    ReadyMade {
        pool: Array2<f64>,
        fitness: Vec<F::Fitness>,
    },
    Sorted {
        pool: Array2<f64>,
        fitness: Vec<F::Fitness>,
    },
    Func(PoolFunc<'a, F>),
}

/// Collect configuration and build the solver.
///
/// This type is created by [`Solver::build()`] method.
#[must_use = "solver builder do nothing unless call the \"solve\" method"]
pub struct SolverBuilder<'a, S: Setting, F: ObjFunc> {
    func: F,
    pop_num: usize,
    seed: SeedOption,
    regen: bool,
    setting: S,
    pool: Pool<'a, F>,
    task: TaskFunc<'a, F>,
    callback: CallbackFunc<'a, F>,
}

impl<'a, S, F> SolverBuilder<'a, S, F>
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

        /// Regenerate the invalid individuals per generation. May spent more time.
        ///
        /// # Default
        ///
        /// By default, this function is disabled.
        fn regen(bool)
    }

    /// Set the random seed to get a determined result.
    ///
    /// # Default
    ///
    /// By default, the random seed is auto-decided.
    pub fn seed(self, seed: impl Into<SeedOption>) -> Self {
        Self { seed: seed.into(), ..self }
    }

    /// Give a pool generating function.
    ///
    /// You can insert a ready-made pool from last states,
    /// or random values with another distribution.
    ///
    /// The array must be the shape of `ctx.pool_size()` and in the bounds of
    /// `[ctx.lb(), ctx.ub())`.
    ///
    /// ```
    /// use metaheuristics_nature::{utility::gaussian_pool, Rga, Solver};
    /// # use metaheuristics_nature::tests::TestObj as MyFunc;
    ///
    /// let s = Solver::build(Rga::default(), MyFunc::new())
    /// #   .task(|ctx| ctx.gen == 1)
    ///     .pool(gaussian_pool(&[0.; 4], &[5.; 4]))
    ///     .solve()
    ///     .unwrap();
    /// ```
    ///
    /// # Default
    ///
    /// By default, the pool will generate with uniform distribution in the
    /// bounds. ([`uniform_pool()`])
    ///
    /// # See Also
    ///
    /// [`Self::pool_and_fitness()`], [`uniform_pool()`], [`gaussian_pool()`]
    pub fn pool<'b, C>(self, pool: C) -> SolverBuilder<'b, S, F>
    where
        'a: 'b,
        C: FnOnce(&Ctx<F>, &Rng) -> Array2<f64> + Send + 'b,
    {
        SolverBuilder { pool: Pool::Func(Box::new(pool)), ..self }
    }

    /// Give a ready-made pool and its fitness values directly.
    ///
    /// The `pool` must be the shape of `ctx.pool_size()` and in the bounds of
    /// `[ctx.lb(), ctx.ub())`, and the `fitness` must have the same length
    /// as `ctx.pop_num()`.
    ///
    /// # See Also
    ///
    /// [`Self::pool()`], [`Self::pool_and_fitness()`]
    pub fn pool_and_fitness(self, pool: Array2<f64>, fitness: Vec<F::Fitness>) -> Self {
        Self { pool: Pool::ReadyMade { pool, fitness }, ..self }
    }

    /// Give a sorted pool and its fitness values directly.
    ///
    /// # See Also
    ///
    /// [`Self::pool()`], [`Self::pool_and_fitness()`]
    pub fn pool_and_fitness_sorted(self, pool: Array2<f64>, fitness: Vec<F::Fitness>) -> Self {
        Self { pool: Pool::Sorted { pool, fitness }, ..self }
    }

    /// Termination condition.
    ///
    /// The task function will be check each iteration, breaks if the return is
    /// true.
    ///
    /// ```
    /// use metaheuristics_nature::{Rga, Solver};
    /// # use metaheuristics_nature::tests::TestObj as MyFunc;
    ///
    /// let s = Solver::build(Rga::default(), MyFunc::new())
    ///     .task(|ctx| ctx.gen == 20)
    ///     .solve()
    ///     .unwrap();
    /// ```
    ///
    /// # Default
    ///
    /// By default, the algorithm will iterate 200 generation.
    pub fn task<'b, C>(self, task: C) -> SolverBuilder<'b, S, F>
    where
        'a: 'b,
        C: Fn(&Ctx<F>) -> bool + Send + 'b,
    {
        SolverBuilder { task: Box::new(task), ..self }
    }

    /// Set callback function.
    ///
    /// Callback function allows to change an outer mutable variable in each
    /// iteration.
    ///
    /// ```
    /// use metaheuristics_nature::{Rga, Solver};
    /// # use metaheuristics_nature::tests::TestObj as MyFunc;
    ///
    /// let mut gen = 0;
    /// let s = Solver::build(Rga::default(), MyFunc::new())
    /// #   .task(|ctx| ctx.gen == 1)
    ///     .callback(|ctx| gen = ctx.gen)
    ///     .solve()
    ///     .unwrap();
    /// ```
    ///
    /// # Default
    ///
    /// By default, this function does nothing.
    pub fn callback<'b, C>(self, callback: C) -> SolverBuilder<'b, S, F>
    where
        'a: 'b,
        C: FnMut(&mut Ctx<F>) + Send + 'b,
    {
        SolverBuilder { callback: Box::new(callback), ..self }
    }

    /// Run the algorithm with a single thread condition.
    ///
    /// This function can control the thread number to one.
    ///
    /// # See Also
    ///
    /// [`crate::rayon::single_thread()`]
    #[cfg(feature = "rayon")]
    pub fn solve_single_thread(self, when: bool) -> Result<Solver<F>, ndarray::ShapeError>
    where
        S: Send,
    {
        crate::rayon::single_thread(when, || self.solve())
    }

    /// Create the task and run the algorithm, which may takes a lot of time.
    ///
    /// Generation `ctx.gen` is start from 1, initialized at 0.
    ///
    /// # Error
    ///
    /// This function will be `Ok` and returns result when the `ctx.pool` and
    /// `ctx.fitness` initialized successfully; `Err` when the boundary check
    /// fails.
    pub fn solve(self) -> Result<Solver<F>, ndarray::ShapeError> {
        let Self {
            func,
            pop_num,
            seed,
            regen,
            setting,
            pool,
            task,
            mut callback,
        } = self;
        assert_shape(func.bound().iter().all(|[lb, ub]| lb <= ub))?;
        let mut method = setting.algorithm();
        let mut ctx = Ctx::new(func, pop_num);
        let rng = Rng::new(seed);
        match pool {
            Pool::ReadyMade { pool, fitness } => {
                assert_shape(pool.shape() == ctx.pool_size())?;
                ctx.pool = pool;
                ctx.pool_f = fitness;
                ctx.find_best_force();
            }
            Pool::Sorted { pool, fitness } => {
                assert_shape(pool.shape() == ctx.pool_size())?;
                ctx.pool = pool;
                ctx.pool_f = fitness;
                ctx.best = ctx.pool.slice(s![0, ..]).to_owned();
                ctx.best_f = ctx.pool_f[0].clone();
            }
            Pool::Func(f) => {
                let pool = f(&ctx, &rng);
                assert_shape(pool.shape() == ctx.pool_size())?;
                ctx.init_pop(pool);
            }
        }
        method.init(&mut ctx, &rng);
        loop {
            callback(&mut ctx);
            if task(&ctx) {
                break;
            }
            ctx.gen += 1;
            method.generation(&mut ctx, &rng);
            if regen {
                ctx.pool
                    .axis_iter_mut(Axis(0))
                    .zip(ctx.pool_f.iter_mut())
                    .filter(|(_, f)| f.partial_cmp(f).is_none())
                    .for_each(|(mut xs, f)| {
                        xs.iter_mut()
                            .enumerate()
                            .for_each(|(s, x)| *x = rng.range(ctx.func.bound_range(s)));
                        *f = ctx.func.fitness(xs.as_slice().unwrap());
                    });
            }
        }
        Ok(Solver::new(ctx, rng.seed()))
    }
}

impl<F: ObjFunc> Solver<F> {
    /// Start to build a solver. Take a setting and setup the configurations.
    ///
    /// Please check [`SolverBuilder`] type, it will help you choose your
    /// configuration.
    ///
    /// If all things are well-setup, call [`SolverBuilder::solve()`].
    ///
    /// The default value of each option can be found in their document.
    pub fn build<S>(setting: S, func: F) -> SolverBuilder<'static, S, F>
    where
        S: Setting,
    {
        SolverBuilder {
            func,
            pop_num: S::default_pop(),
            seed: SeedOption::None,
            regen: false,
            setting,
            pool: Pool::Func(Box::new(|ctx, rng| uniform_pool(ctx, rng))), // dynamic lifetime
            task: Box::new(|ctx| ctx.gen >= 200),
            callback: Box::new(|_| ()),
        }
    }
}

/// A function generates a uniform pool.
///
/// Please see [`SolverBuilder::pool()`] for more information.
pub fn uniform_pool<F: ObjFunc>(ctx: &Ctx<F>, rng: &Rng) -> Array2<f64> {
    Array2::from_shape_fn(ctx.pool_size(), |(_, s)| rng.range(ctx.func.bound_range(s)))
}

/// A function generates a Gaussian pool.
///
/// Where `mean` is the mean value, `std` is the standard deviation.
///
/// Please see [`SolverBuilder::pool()`] for more information.
///
/// # Panics
///
/// Panic when the lengths of `mean` and `std` are not the same.
pub fn gaussian_pool<'a, F: ObjFunc>(
    mean: &'a [f64],
    std: &'a [f64],
) -> impl Fn(&Ctx<F>, &Rng) -> Array2<f64> + 'a {
    assert_eq!(mean.len(), std.len());
    move |ctx, rng| {
        Array2::from_shape_fn(ctx.pool_size(), |(_, s)| {
            ctx.clamp(s, rng.normal(mean[s], std[s]))
        })
    }
}

/// A function generates a Gaussian pool, including the mean (center).
///
/// Where `mean` is the mean value, `std` is the standard deviation.
///
/// Please see [`SolverBuilder::pool()`] for more information.
///
/// # Panics
///
/// Panic when the lengths of `mean` and `std` are not the same.
pub fn gaussian_pool_inclusive<'a, F: ObjFunc>(
    mean: &'a [f64],
    std: &'a [f64],
) -> impl Fn(&Ctx<F>, &Rng) -> Array2<f64> + 'a {
    assert_eq!(mean.len(), std.len());
    move |ctx, rng| {
        Array2::from_shape_fn(ctx.pool_size(), |(i, s)| {
            if i == 0 {
                mean[s]
            } else {
                ctx.clamp(s, rng.normal(mean[s], std[s]))
            }
        })
    }
}
