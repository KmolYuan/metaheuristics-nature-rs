//! # Differential Evolution
//!
//! <https://en.wikipedia.org/wiki/Differential_evolution>
use self::Strategy::*;
use crate::prelude::*;
use alloc::{boxed::Box, vec::Vec};

/// Differential Evolution type.
pub type Method = De;
type Func<F> = Box<dyn Fn(&Ctx<F>, &[f64], usize) -> f64>;

const DEF: De = De { strategy: S1, f: 0.6, cross: 0.9 };

/// The Differential Evolution strategy.
///
/// Each strategy has different formulas on the recombination.
///
/// # Variable formula
///
/// This formula decide how to generate new variable *n*.
/// Where *vi* is the random indicator on the individuals,
/// they are different from each other.
///
/// + *f1*: best{n} + F * (v0{n} - v1{n})
/// + *f2*: v0{n} + F * (v1{n} - v2{n})
/// + *f3*: self{n} + F * (best{n} - self{n} + v0{n} - v1{n})
/// + *f4*: best{n} + F * (v0{n} + v1{n} - v2{n} - v3{n})
/// + *f5*: v4{n} + F * (v0{n} + v1{n} - v2{n} - v3{n})
///
/// # Crossover formula
///
/// + *c1*: Continue crossover in order until end with probability.
/// + *c2*: Each variable has independent probability.
#[derive(Default, Copy, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Strategy {
    /// *f1* + *c1*
    #[default]
    S1,
    /// *f2* + *c1*
    S2,
    /// *f3* + *c1*
    S3,
    /// *f4* + *c1*
    S4,
    /// *f5* + *c1*
    S5,
    /// *f1* + *c2*
    S6,
    /// *f2* + *c2*
    S7,
    /// *f3* + *c2*
    S8,
    /// *f4* + *c2*
    S9,
    /// *f5* + *c2*
    S10,
}

/// Differential Evolution settings.
#[derive(Clone, PartialEq)]
#[cfg_attr(feature = "clap", derive(clap::Args))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct De {
    /// Strategy
    #[cfg_attr(feature = "clap", clap(long, value_enum, default_value_t = DEF.strategy))]
    pub strategy: Strategy,
    /// F factor in the formula
    #[cfg_attr(feature = "clap", clap(long, default_value_t = DEF.f))]
    pub f: f64,
    /// Crossover rate
    #[cfg_attr(feature = "clap", clap(long, default_value_t = DEF.cross))]
    pub cross: f64,
}

impl De {
    /// Constant default value.
    pub const fn new() -> Self {
        DEF
    }

    impl_builders! {
        /// Strategy of the formula.
        fn strategy(Strategy)
        /// F factor.
        fn f(f64)
        /// Crossing probability.
        fn cross(f64)
    }
}

impl Default for De {
    fn default() -> Self {
        DEF
    }
}

impl Setting for De {
    type Algorithm<F: ObjFunc> = Method;

    fn algorithm<F: ObjFunc>(self) -> Self::Algorithm<F> {
        self
    }

    fn default_pop() -> usize {
        400
    }
}

impl Method {
    fn formula<F: ObjFunc>(&self, ctx: &Ctx<F>, rng: &mut Rng) -> Func<F> {
        let f = self.f;
        match self.strategy {
            S1 | S6 => {
                let [v0, v1] = rng.array(0..ctx.pop_num());
                let best = ctx.best.sample_xs(rng).to_vec();
                Box::new(move |ctx, _, s| best[s] + f * (ctx.pool[v0][s] - ctx.pool[v1][s]))
            }
            S2 | S7 => Box::new({
                let [v0, v1, v2] = rng.array(0..ctx.pop_num());
                move |ctx, _, s| ctx.pool[v0][s] + f * (ctx.pool[v1][s] - ctx.pool[v2][s])
            }),
            S3 | S8 => Box::new({
                let [v0, v1] = rng.array(0..ctx.pop_num());
                let best = ctx.best.sample_xs(rng).to_vec();
                move |ctx, xs, s| xs[s] + f * (best[s] - xs[s] + ctx.pool[v0][s] - ctx.pool[v1][s])
            }),
            S4 | S9 => Box::new({
                let [v0, v1, v2, v3] = rng.array(0..ctx.pop_num());
                let best = ctx.best.sample_xs(rng).to_vec();
                move |ctx, _, s| {
                    best[s]
                        + f * (ctx.pool[v0][s] + ctx.pool[v1][s]
                            - ctx.pool[v2][s]
                            - ctx.pool[v3][s])
                }
            }),
            S5 | S10 => Box::new({
                let [v0, v1, v2, v3, v4] = rng.array(0..ctx.pop_num());
                move |ctx, _, s| {
                    ctx.pool[v4][s]
                        + f * (ctx.pool[v0][s] + ctx.pool[v1][s]
                            - ctx.pool[v2][s]
                            - ctx.pool[v3][s])
                }
            }),
        }
    }

    fn c1<F>(&self, ctx: &Ctx<F>, rng: &mut Rng, xs: &mut [f64], formula: Func<F>)
    where
        F: ObjFunc,
    {
        let dim = ctx.dim();
        for s in (0..dim).cycle().skip(rng.ub(dim)).take(dim) {
            if !rng.maybe(self.cross) {
                break;
            }
            xs[s] = rng.clamp(formula(ctx, xs, s), ctx.bound_range(s));
        }
    }

    fn c2<F>(&self, ctx: &Ctx<F>, rng: &mut Rng, xs: &mut [f64], formula: Func<F>)
    where
        F: ObjFunc,
    {
        for s in 0..ctx.dim() {
            if rng.maybe(self.cross) {
                xs[s] = rng.clamp(formula(ctx, xs, s), ctx.bound_range(s));
            }
        }
    }
}

impl<F: ObjFunc> Algorithm<F> for Method {
    fn generation(&mut self, ctx: &mut Ctx<F>, rng: &mut Rng) {
        let mut pool = ctx.pool.clone();
        let mut pool_y = ctx.pool_y.clone();
        let rng = rng.stream(ctx.pop_num());
        #[cfg(not(feature = "rayon"))]
        let iter = rng.into_iter();
        #[cfg(feature = "rayon")]
        let iter = rng.into_par_iter();
        let (xs, ys): (Vec<_>, Vec<_>) = iter
            .zip(&mut pool)
            .zip(&mut pool_y)
            .map(|((mut rng, xs), ys)| {
                // Generate Vector
                let formula = self.formula(ctx, &mut rng);
                // Recombination
                let mut xs_try = xs.clone();
                match self.strategy {
                    S1 | S2 | S3 | S4 | S5 => self.c1(ctx, &mut rng, &mut xs_try, formula),
                    S6 | S7 | S8 | S9 | S10 => self.c2(ctx, &mut rng, &mut xs_try, formula),
                }
                let ys_try = ctx.fitness(&xs_try);
                if ys_try.is_dominated(ys) {
                    *xs = xs_try;
                    *ys = ys_try;
                }
                // Search with the trial individuals
                (&*xs, &*ys)
            })
            .unzip();
        ctx.best.update_all(xs, ys);
        ctx.pool = pool;
        ctx.pool_y = pool_y;
    }
}
