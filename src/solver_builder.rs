use crate::prelude::*;
use alloc::{boxed::Box, vec::Vec};

/// A [`SolverBuilder`] that use a boxed algorithm.
///
/// Generated by [`Solver::build_boxed()`] method.
pub type SolverBox<'a, F> = SolverBuilder<'a, Box<dyn Algorithm<F> + Send>, F>;

type PoolFunc<'a> = Box<dyn Fn(usize, core::ops::RangeInclusive<f64>, &mut Rng) -> f64 + Send + 'a>;

/// Initial pool generating options.
///
/// Use [`SolverBuilder::init_pool()`] to set this option.
pub enum Pool<'a, F: ObjFunc> {
    /// A ready-made pool and its fitness values.
    Ready {
        /// Pool
        pool: Vec<Vec<f64>>,
        /// Fitness values
        pool_y: Vec<F::Ys>,
    },
    /// Generate the pool uniformly with a filter function to check the
    /// validity.
    ///
    /// This filter function returns true if the design variables are valid.
    #[allow(clippy::type_complexity)]
    UniformBy(Box<dyn Fn(&[f64]) -> bool + Send + 'a>),
    /// Generate the pool with a specific function.
    ///
    /// The function signature is `fn(s, min..max, &rng) -> value`
    /// + `s` is the index of the variable
    /// + `min..max` is the range of the variable
    /// + `rng` is the random number generator
    ///
    /// Two examples are [`uniform_pool()`] and [`gaussian_pool()`].
    ///
    /// ```
    /// use metaheuristics_nature::{gaussian_pool, Pool, Rga, Solver};
    /// # use metaheuristics_nature::tests::TestObj as MyFunc;
    ///
    /// let pool = Pool::Func(Box::new(gaussian_pool(&[0.; 4], &[1.; 4])));
    /// let s = Solver::build(Rga::default(), MyFunc::new())
    ///     .seed(0)
    ///     .task(|ctx| ctx.gen == 20)
    ///     .init_pool(pool)
    ///     .solve();
    /// ```
    Func(PoolFunc<'a>),
}

/// Collect configuration and build the solver.
///
/// This type is created by [`Solver::build()`] method.
///
/// + First, setting a fixed seed with [`SolverBuilder::seed()`] method to get a
///   determined result is highly recommended.
/// + Next is [`SolverBuilder::task()`] method with a termination condition.
/// + Finally, call [`SolverBuilder::solve()`] method to start the algorithm.
#[allow(clippy::type_complexity)]
#[must_use = "solver builder do nothing unless call the \"solve\" method"]
pub struct SolverBuilder<'a, A: Algorithm<F>, F: ObjFunc> {
    func: F,
    algorithm: A,
    pop_num: usize,
    pareto_limit: usize,
    seed: SeedOpt,
    pool: Pool<'a, F>,
    task: Box<dyn FnMut(&Ctx<F>) -> bool + Send + 'a>,
    callback: Box<dyn FnMut(&Ctx<F>) + Send + 'a>,
}

impl<'a, A: Algorithm<F>, F: ObjFunc> SolverBuilder<'a, A, F> {
    impl_builders! {
        /// Population number.
        ///
        /// # Default
        ///
        /// If not changed by the algorithm setting, the default number is 200.
        fn pop_num(usize)
    }

    /// Pareto front limit.
    ///
    /// It is not working for single-objective optimization.
    ///
    /// ```
    /// use metaheuristics_nature::{Rga, Solver};
    /// # use metaheuristics_nature::tests::TestMO as MyFunc;
    ///
    /// let s = Solver::build(Rga::default(), MyFunc::new())
    ///     .seed(0)
    ///     .task(|ctx| ctx.gen == 20)
    ///     .pareto_limit(10)
    ///     .solve();
    /// ```
    ///
    /// # Default
    ///
    /// By default, there is no limit. The limit is set to `usize::MAX`.
    pub fn pareto_limit(self, pareto_limit: usize) -> Self
    where
        F::Ys: Fitness<Best<F::Ys> = Pareto<F::Ys>>,
    {
        Self { pareto_limit, ..self }
    }

    /// Set a fixed random seed to get a determined result.
    ///
    /// # Default
    ///
    /// By default, the random seed is auto-decided so you cannot reproduce the
    /// result. Please print the seed via [`Solver::seed()`] method to get the
    /// seed that used in the algorithm.
    pub fn seed(self, seed: impl Into<SeedOpt>) -> Self {
        Self { seed: seed.into(), ..self }
    }

    /// Initialize the pool with the pool option.
    ///
    /// # Default
    ///
    /// By default, the pool is generated by the uniform distribution
    /// [`uniform_pool()`].
    pub fn init_pool(self, pool: Pool<'a, F>) -> Self {
        Self { pool, ..self }
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
    ///     .seed(0)
    ///     .task(|ctx| ctx.gen == 20)
    ///     .solve();
    /// ```
    ///
    /// # Default
    ///
    /// By default, the algorithm will iterate 200 generation.
    pub fn task<'b, C>(self, task: C) -> SolverBuilder<'b, A, F>
    where
        'a: 'b,
        C: FnMut(&Ctx<F>) -> bool + Send + 'b,
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
    /// let mut report = Vec::with_capacity(20);
    /// let s = Solver::build(Rga::default(), MyFunc::new())
    ///     .seed(0)
    ///     .task(|ctx| ctx.gen == 20)
    ///     .callback(|ctx| report.push(ctx.best.get_eval()))
    ///     .solve();
    /// ```
    ///
    /// # Default
    ///
    /// By default, this function does nothing.
    pub fn callback<'b, C>(self, callback: C) -> SolverBuilder<'b, A, F>
    where
        'a: 'b,
        C: FnMut(&Ctx<F>) + Send + 'b,
    {
        SolverBuilder { callback: Box::new(callback), ..self }
    }

    /// Create the task and run the algorithm, which may takes a lot of time.
    ///
    /// Generation `ctx.gen` is start from 1, initialized at 0.
    ///
    /// # Panics
    ///
    /// Panics before starting the algorithm if the following conditions are
    /// met:
    /// + The dimension size is zero.
    /// + The lower bound is greater than the upper bound.
    /// + Using the [`Pool::Ready`] option and the pool size or dimension size
    ///   is not consistent.
    pub fn solve(self) -> Solver<F> {
        let Self {
            func,
            mut algorithm,
            pop_num,
            pareto_limit,
            seed,
            pool,
            mut task,
            mut callback,
        } = self;
        assert!(func.dim() != 0, "Dimension should be greater than 0");
        assert!(
            func.bound().iter().all(|[lb, ub]| lb <= ub),
            "Lower bound should be less than upper bound"
        );
        let mut rng = Rng::new(seed);
        let mut ctx = match pool {
            Pool::Ready { pool, pool_y } => {
                assert_eq!(pool.len(), pool_y.len(), "Pool size mismatched");
                let dim = func.dim();
                pool.iter()
                    .for_each(|xs| assert!(xs.len() == dim, "Pool dimension mismatched"));
                Ctx::from_parts(func, pareto_limit, pool, pool_y)
            }
            Pool::UniformBy(filter) => {
                let dim = func.dim();
                let mut pool = Vec::with_capacity(pop_num);
                let rand_f = uniform_pool();
                while pool.len() < pop_num {
                    let xs = (0..dim)
                        .map(|s| rand_f(s, func.bound_range(s), &mut rng))
                        .collect::<Vec<_>>();
                    if filter(&xs) {
                        pool.push(xs);
                    }
                }
                Ctx::from_pool(func, pareto_limit, pool)
            }
            Pool::Func(f) => {
                let dim = func.dim();
                let pool = (0..pop_num)
                    .map(|_| {
                        (0..dim)
                            .map(|s| f(s, func.bound_range(s), &mut rng))
                            .collect()
                    })
                    .collect();
                Ctx::from_pool(func, pareto_limit, pool)
            }
        };
        algorithm.init(&mut ctx, &mut rng);
        loop {
            callback(&ctx);
            if task(&ctx) {
                break;
            }
            ctx.gen += 1;
            algorithm.generation(&mut ctx, &mut rng);
        }
        Solver::new(ctx, rng.seed())
    }
}

impl<F: ObjFunc> Solver<F> {
    /// Start to build a solver. Take a setting and setup the configurations.
    ///
    /// The signature is something like `Solver::build(Rga::default(),
    /// MyFunc::new())`. Please check the [`SolverBuilder`] type, it will help
    /// you choose your configuration.
    ///
    /// If all things are well-setup, call [`SolverBuilder::solve()`].
    ///
    /// The default value of each option can be found in their document.
    ///
    /// Use [`Solver::build_boxed()`] for dynamic dispatching.
    pub fn build<A: AlgCfg>(cfg: A, func: F) -> SolverBuilder<'static, A::Algorithm<F>, F> {
        Self::build_default(cfg.algorithm(), A::pop_num(), func)
    }

    /// Start to build a solver with a boxed algorithm, the dynamic dispatching.
    ///
    /// This method allows you to choose the algorithm at runtime and mix them
    /// with the same type.
    ///
    /// ```
    /// use metaheuristics_nature as mh;
    /// # use metaheuristics_nature::tests::TestObj as MyFunc;
    ///
    /// # let use_ga = true;
    /// let s = if use_ga {
    ///     mh::Solver::build_boxed(mh::Rga::default(), MyFunc::new())
    /// } else {
    ///     mh::Solver::build_boxed(mh::De::default(), MyFunc::new())
    /// };
    /// ```
    ///
    /// Use [`Solver::build()`] for optimized memory allocation and access.
    pub fn build_boxed<A>(cfg: A, func: F) -> SolverBox<'static, F>
    where
        A: AlgCfg,
        A::Algorithm<F>: Send,
    {
        Self::build_default(Box::new(cfg.algorithm()), A::pop_num(), func)
    }

    fn build_default<A: Algorithm<F>>(
        algorithm: A,
        pop_num: usize,
        func: F,
    ) -> SolverBuilder<'static, A, F> {
        SolverBuilder {
            func,
            algorithm,
            pop_num,
            pareto_limit: usize::MAX,
            seed: SeedOpt::Entropy,
            pool: Pool::Func(Box::new(uniform_pool())),
            task: Box::new(|ctx| ctx.gen == 200),
            callback: Box::new(|_| ()),
        }
    }
}

/// A function generates a uniform pool.
///
/// See also [`gaussian_pool()`], [`Pool::Func`], and
/// [`SolverBuilder::init_pool()`].
pub fn uniform_pool() -> PoolFunc<'static> {
    Box::new(move |_, range, rng| rng.range(range))
}

/// A function generates a Gaussian pool.
///
/// Where `mean` is the mean value, `std` is the standard deviation.
///
/// See also [`uniform_pool()`], [`Pool::Func`], and
/// [`SolverBuilder::init_pool()`].
///
/// # Panics
///
/// Panic when the lengths of `mean` and `std` are not the same.
pub fn gaussian_pool<'a>(mean: &'a [f64], std: &'a [f64]) -> PoolFunc<'a> {
    assert_eq!(mean.len(), std.len());
    Box::new(move |s, _, rng| rng.normal(mean[s], std[s]))
}
