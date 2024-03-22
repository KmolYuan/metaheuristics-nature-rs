//! # Differential Evolution
//!
//! <https://en.wikipedia.org/wiki/Differential_evolution>
use self::Strategy::*;
use crate::prelude::*;
use alloc::{boxed::Box, vec::Vec};

/// Algorithm of the Differential Evolution.
pub type Method = De;
type Func<F> = Box<dyn Fn(&Ctx<F>, &[f64], usize) -> f64>;

const DEF: De = De { strategy: C1F1, f: 0.6, cross: 0.9 };

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
    C1F1,
    /// *f2* + *c1*
    C1F2,
    /// *f3* + *c1*
    C1F3,
    /// *f4* + *c1*
    C1F4,
    /// *f5* + *c1*
    C1F5,
    /// *f1* + *c2*
    C2F1,
    /// *f2* + *c2*
    C2F2,
    /// *f3* + *c2*
    C2F3,
    /// *f4* + *c2*
    C2F4,
    /// *f5* + *c2*
    C2F5,
}

impl Strategy {
    /// A list of all strategies.
    pub const LIST: [Self; 10] = [C1F1, C1F2, C1F3, C1F4, C1F5, C2F1, C2F2, C2F3, C2F4, C2F5];
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

impl AlgCfg for De {
    type Algorithm<F: ObjFunc> = Method;
    fn algorithm<F: ObjFunc>(self) -> Self::Algorithm<F> {
        self
    }
    fn pop_num() -> usize {
        400
    }
}

impl Method {
    fn formula<F: ObjFunc>(&self, ctx: &Ctx<F>, rng: &mut Rng) -> Func<F> {
        let f = self.f;
        match self.strategy {
            C1F1 | C2F1 => {
                let [v0, v1] = rng.array(0..ctx.pop_num());
                let best = ctx.best.sample_xs(rng).to_vec();
                Box::new(move |ctx, _, s| best[s] + f * (ctx.pool[v0][s] - ctx.pool[v1][s]))
            }
            C1F2 | C2F2 => Box::new({
                let [v0, v1, v2] = rng.array(0..ctx.pop_num());
                move |ctx, _, s| ctx.pool[v0][s] + f * (ctx.pool[v1][s] - ctx.pool[v2][s])
            }),
            C1F3 | C2F3 => Box::new({
                let [v0, v1] = rng.array(0..ctx.pop_num());
                let best = ctx.best.sample_xs(rng).to_vec();
                move |ctx, xs, s| xs[s] + f * (best[s] - xs[s] + ctx.pool[v0][s] - ctx.pool[v1][s])
            }),
            C1F4 | C2F4 => Box::new({
                let [v0, v1, v2, v3] = rng.array(0..ctx.pop_num());
                let best = ctx.best.sample_xs(rng).to_vec();
                move |ctx, _, s| {
                    best[s]
                        + f * (ctx.pool[v0][s] + ctx.pool[v1][s]
                            - ctx.pool[v2][s]
                            - ctx.pool[v3][s])
                }
            }),
            C1F5 | C2F5 => Box::new({
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
        for (i, s) in (0..dim).cycle().skip(rng.ub(dim)).take(dim).enumerate() {
            // At last two variables are modified
            if i > 1 && !rng.maybe(self.cross) {
                break;
            }
            xs[s] = rng.clamp(formula(ctx, xs, s), ctx.bound_range(s));
        }
    }

    fn c2<F>(&self, ctx: &Ctx<F>, rng: &mut Rng, xs: &mut [f64], formula: Func<F>)
    where
        F: ObjFunc,
    {
        // At least one variable is modified
        let sss = rng.ub(ctx.dim());
        for s in 0..ctx.dim() {
            if sss == s || rng.maybe(self.cross) {
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
            .filter_map(|((mut rng, xs), ys)| {
                // Generate Vector
                let formula = self.formula(ctx, &mut rng);
                // Recombination
                let mut xs_trial = xs.clone();
                match self.strategy {
                    C1F1 | C1F2 | C1F3 | C1F4 | C1F5 => {
                        self.c1(ctx, &mut rng, &mut xs_trial, formula)
                    }
                    C2F1 | C2F2 | C2F3 | C2F4 | C2F5 => {
                        self.c2(ctx, &mut rng, &mut xs_trial, formula)
                    }
                }
                let ys_trial = ctx.fitness(&xs_trial);
                if ys_trial.is_dominated(ys) {
                    *xs = xs_trial;
                    *ys = ys_trial;
                    Some((&*xs, &*ys))
                } else {
                    None
                }
            })
            .unzip();
        ctx.best.update_all(xs, ys);
        ctx.pool = pool;
        ctx.pool_y = pool_y;
    }
}
