//! Differential Evolution.
//!
//! <https://en.wikipedia.org/wiki/Differential_evolution>
use self::Strategy::*;
use crate::utility::prelude::*;

type Func<F> = Box<dyn Fn(&Ctx<F>, &Array1<f64>, usize) -> f64>;

macro_rules! impl_f45 {
    ($ctx:ident, $v:ident, $n:ident, $f:ident) => {
        $f * ($ctx.pool[[$v[0], $n]] + $ctx.pool[[$v[1], $n]]
            - $ctx.pool[[$v[2], $n]]
            - $ctx.pool[[$v[3], $n]])
    };
}

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
        Self { strategy: S1, f: 0.6, cross: 0.9 }
    }
}

impl Setting for De {
    type Algorithm = Method;

    fn algorithm(self) -> Self::Algorithm {
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
                let v = ctx.rng.vector([0; 2], 0, 0..ctx.pop_num());
                Box::new(move |ctx, _, n| {
                    ctx.best[n] + f * (ctx.pool[[v[0], n]] - ctx.pool[[v[1], n]])
                })
            }
            S2 | S7 => {
                let v = ctx.rng.vector([0; 3], 0, 0..ctx.pop_num());
                Box::new(move |ctx, _, n| {
                    ctx.pool[[v[0], n]] + f * (ctx.pool[[v[1], n]] - ctx.pool[[v[2], n]])
                })
            }
            S3 | S8 => {
                let v = ctx.rng.vector([0; 2], 0, 0..ctx.pop_num());
                Box::new(move |ctx, tmp, n| {
                    tmp[n] + f * (ctx.best[n] - tmp[n] + ctx.pool[[v[0], n]] - ctx.pool[[v[1], n]])
                })
            }
            S4 | S9 => {
                let v = ctx.rng.vector([0; 4], 0, 0..ctx.pop_num());
                Box::new(move |ctx, _, n| ctx.best[n] + impl_f45!(ctx, v, n, f))
            }
            S5 | S10 => {
                let v = ctx.rng.vector([0; 5], 0, 0..ctx.pop_num());
                Box::new(move |ctx, _, n| ctx.pool[[v[4], n]] + impl_f45!(ctx, v, n, f))
            }
        }
    }

    fn c1<F: ObjFunc>(&mut self, ctx: &Ctx<F>, tmp: &mut Array1<f64>, formula: Func<F>) {
        let mut n = ctx.rng.int(0..ctx.dim());
        for _ in 0..ctx.dim() {
            tmp[n] = formula(ctx, tmp, n);
            n += 1;
            if n >= ctx.dim() {
                n = 0;
            }
            if !ctx.rng.maybe(self.cross) {
                break;
            }
        }
    }

    fn c2<F: ObjFunc>(&mut self, ctx: &Ctx<F>, tmp: &mut Array1<f64>, formula: Func<F>) {
        let mut n = ctx.rng.int(0..ctx.dim());
        for lv in 0..ctx.dim() {
            if !ctx.rng.maybe(self.cross) || lv == ctx.dim() - 1 {
                tmp[n] = formula(ctx, tmp, n);
            }
            if n >= ctx.dim() {
                n = 0;
            }
        }
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
            if !(0..ctx.dim()).all(|s| ctx.bound_range(s).contains(&tmp[s])) {
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
