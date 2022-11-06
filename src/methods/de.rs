//! Differential Evolution.
//!
//! <https://en.wikipedia.org/wiki/Differential_evolution>
use self::Strategy::*;
use crate::utility::prelude::*;
use alloc::boxed::Box;

type Func<F> = Box<dyn Fn(&Ctx<F>, &Array1<f64>, usize) -> f64>;

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
#[derive(Clone, Default)]
#[cfg_attr(feature = "clap", derive(clap::ValueEnum))]
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
#[cfg_attr(feature = "clap", derive(clap::Args))]
pub struct De {
    #[cfg_attr(feature = "clap", clap(long, value_enum, default_value_t = DEF.strategy))]
    strategy: Strategy,
    #[cfg_attr(feature = "clap", clap(long, default_value_t = DEF.f))]
    f: f64,
    #[cfg_attr(feature = "clap", clap(long, default_value_t = DEF.cross))]
    cross: f64,
}

impl De {
    impl_builders! {
        default,
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
        let Self { strategy, f, cross } = self;
        Method { f, cross, strategy }
    }

    fn default_pop() -> usize {
        400
    }
}

/// Differential Evolution type.
pub struct Method {
    f: f64,
    cross: f64,
    strategy: Strategy,
}

impl Method {
    fn formula<F: ObjFunc>(&self, ctx: &Ctx<F>) -> Func<F> {
        let f = self.f;
        match self.strategy {
            S1 | S6 => {
                let [v0, v1] = ctx.rng.array(0..ctx.pop_num());
                Box::new(move |ctx, _, s| ctx.best[s] + f * (ctx.pool[[v0, s]] - ctx.pool[[v1, s]]))
            }
            S2 | S7 => Box::new({
                let [v0, v1, v2] = ctx.rng.array(0..ctx.pop_num());
                move |ctx, _, s| ctx.pool[[v0, s]] + f * (ctx.pool[[v1, s]] - ctx.pool[[v2, s]])
            }),
            S3 | S8 => Box::new({
                let [v0, v1] = ctx.rng.array(0..ctx.pop_num());
                move |ctx, tmp, s| {
                    tmp[s] + f * (ctx.best[s] - tmp[s] + ctx.pool[[v0, s]] - ctx.pool[[v1, s]])
                }
            }),
            S4 | S9 => Box::new({
                let [v0, v1, v2, v3] = ctx.rng.array(0..ctx.pop_num());
                move |ctx, _, s| {
                    ctx.best[s]
                        + f * (ctx.pool[[v0, s]] + ctx.pool[[v1, s]]
                            - ctx.pool[[v2, s]]
                            - ctx.pool[[v3, s]])
                }
            }),
            S5 | S10 => Box::new({
                let [v0, v1, v2, v3, v4] = ctx.rng.array(0..ctx.pop_num());
                move |ctx, _, s| {
                    ctx.pool[[v4, s]]
                        + f * (ctx.pool[[v0, s]] + ctx.pool[[v1, s]]
                            - ctx.pool[[v2, s]]
                            - ctx.pool[[v3, s]])
                }
            }),
        }
    }

    fn c1<F: ObjFunc>(&mut self, ctx: &Ctx<F>, tmp: &mut Array1<f64>, formula: Func<F>) {
        (0..ctx.dim())
            .cycle()
            .skip(ctx.rng.ub(ctx.dim()))
            .take(ctx.dim())
            .take_while(|_| ctx.rng.maybe(self.cross))
            .for_each(|s| tmp[s] = formula(ctx, tmp, s))
    }

    fn c2<F: ObjFunc>(&mut self, ctx: &Ctx<F>, tmp: &mut Array1<f64>, formula: Func<F>) {
        (0..ctx.dim())
            .filter(|_| ctx.rng.maybe(self.cross))
            .for_each(|s| tmp[s] = formula(ctx, tmp, s))
    }
}

impl<F: ObjFunc> Algorithm<F> for Method {
    fn generation(&mut self, ctx: &mut Ctx<F>) {
        for i in 0..ctx.pop_num() {
            // Generate Vector
            let formula = self.formula(ctx);
            // Recombination
            let mut tmp = ctx.pool.slice(s![i, ..]).to_owned();
            match self.strategy {
                S1 | S2 | S3 | S4 | S5 => self.c1(ctx, &mut tmp, formula),
                S6 | S7 | S8 | S9 | S10 => self.c2(ctx, &mut tmp, formula),
            }
            if (0..ctx.dim()).any(|s| !ctx.bound_range(s).contains(&tmp[s])) {
                continue;
            }
            let tmp_f = ctx.func.fitness(tmp.as_slice().unwrap());
            if tmp_f < ctx.pool_f[i] {
                ctx.assign_from(i, tmp_f, &tmp);
            }
        }
        ctx.find_best();
    }
}
