//! Differential Evolution.
//!
//! <https://en.wikipedia.org/wiki/Differential_evolution>
use self::Strategy::*;
use crate::utility::prelude::*;
use alloc::{vec, vec::Vec};

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
/// # Crossing formula
///
/// + *c1*: Continue crossing with the variables order until failure.
/// + *c2*: Each variable has independent probability.
#[derive(Clone)]
pub enum Strategy {
    /// *f1* + *c1*
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
pub struct De {
    strategy: Strategy,
    f: f64,
    cross: f64,
}

impl De {
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
        Self {
            strategy: S1,
            f: 0.6,
            cross: 0.9,
        }
    }
}

impl Setting for De {
    type Algorithm = Method;

    fn algorithm(self) -> Self::Algorithm {
        let num = match self.strategy {
            S1 | S3 | S6 | S8 => 2,
            S2 | S7 => 3,
            S4 | S9 => 4,
            S5 | S10 => 5,
        };
        Method {
            f: self.f,
            cross: self.cross,
            num,
            strategy: self.strategy,
        }
    }

    fn default_basic() -> BasicSetting {
        BasicSetting {
            pop_num: 400,
            ..Default::default()
        }
    }
}

/// Differential Evolution type.
pub struct Method {
    f: f64,
    cross: f64,
    num: usize,
    strategy: Strategy,
}

impl Method {
    fn f45<F: ObjFunc>(&self, ctx: &Context<F>, v: &[usize], n: usize) -> f64 {
        (ctx.pool[[v[0], n]] + ctx.pool[[v[1], n]] - ctx.pool[[v[2], n]] - ctx.pool[[v[3], n]])
            * self.f
    }

    fn formula<F: ObjFunc>(
        &self,
        ctx: &Context<F>,
        tmp: &Array1<f64>,
        v: &[usize],
        n: usize,
    ) -> f64 {
        match self.strategy {
            S1 | S6 => ctx.best[n] + self.f * (ctx.pool[[v[0], n]] - ctx.pool[[v[1], n]]),
            S2 | S7 => ctx.pool[[v[0], n]] + self.f * (ctx.pool[[v[1], n]] - ctx.pool[[v[2], n]]),
            S3 | S8 => {
                tmp[n] + self.f * (ctx.best[n] - tmp[n] + ctx.pool[[v[0], n]] - ctx.pool[[v[1], n]])
            }
            S4 | S9 => ctx.best[n] + self.f45(ctx, v, n),
            S5 | S10 => ctx.pool[[v[4], n]] + self.f45(ctx, v, n),
        }
    }

    fn c1<F: ObjFunc>(
        &mut self,
        ctx: &Context<F>,
        tmp: &mut Array1<f64>,
        v: Vec<usize>,
        mut n: usize,
    ) {
        for _ in 0..ctx.dim() {
            tmp[n] = self.formula(ctx, tmp, &v, n);
            n = (n + 1) % ctx.dim();
            if !ctx.rng.maybe(self.cross) {
                break;
            }
        }
    }

    fn c2<F: ObjFunc>(
        &mut self,
        ctx: &Context<F>,
        tmp: &mut Array1<f64>,
        v: Vec<usize>,
        mut n: usize,
    ) {
        for lv in 0..ctx.dim() {
            if !ctx.rng.maybe(self.cross) || lv == ctx.dim() - 1 {
                tmp[n] = self.formula(ctx, tmp, &v, n);
            }
            n = (n + 1) % ctx.dim();
        }
    }
}

impl<F: ObjFunc> Algorithm<F> for Method {
    fn generation(&mut self, ctx: &mut Context<F>) {
        'a: for i in 0..ctx.pop_num() {
            // Generate Vector
            let mut v = vec![0; self.num];
            ctx.rng.vector(&mut v, 0, 0..ctx.pop_num());
            // Recombination
            let mut tmp = ctx.pool.slice(s![i, ..]).to_owned();
            let n = ctx.rng.int(0..ctx.dim());
            match self.strategy {
                S1 | S2 | S3 | S4 | S5 => self.c1(ctx, &mut tmp, v, n),
                S6 | S7 | S8 | S9 | S10 => self.c2(ctx, &mut tmp, v, n),
            }
            for s in 0..ctx.dim() {
                if tmp[s] > ctx.ub(s) || tmp[s] < ctx.lb(s) {
                    continue 'a;
                }
            }
            let tmp_f = ctx.func.fitness(tmp.as_slice().unwrap(), ctx.adaptive);
            if tmp_f.value() < ctx.fitness[i].value() {
                ctx.assign_from(i, tmp_f, &tmp);
            }
        }
        ctx.find_best();
    }
}
