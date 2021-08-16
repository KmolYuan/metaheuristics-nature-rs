use self::Strategy::*;
use crate::{random::*, *};
use ndarray::{s, Array1};

/// The Differential Evolution strategy.
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

setting_builder! {
    /// Differential Evolution settings.
    pub struct DESetting for DE {
        @base,
        @pop_num = 400,
        /// Strategy of the formula.
        strategy: Strategy = S1,
        /// F factor.
        f: f64 = 0.6,
        /// Crossing probability.
        cross: f64 = 0.9,
    }
}

/// Differential Evolution type.
pub struct DE {
    f: f64,
    cross: f64,
    num: usize,
    strategy: Strategy,
}

impl DE {
    fn formula<F: ObjFunc>(
        &self,
        ctx: &Context<F>,
        tmp: &Array1<f64>,
        v: &Array1<usize>,
        n: usize,
    ) -> f64 {
        match self.strategy {
            S1 | S6 => ctx.best[n] + self.f * (ctx.pool[[v[0], n]] - ctx.pool[[v[1], n]]),
            S2 | S7 => ctx.pool[[v[0], n]] + self.f * (ctx.pool[[v[1], n]] - ctx.pool[[v[2], n]]),
            S3 | S8 => {
                tmp[n] + self.f * (ctx.best[n] - tmp[n] + ctx.pool[[v[0], n]] - ctx.pool[[v[1], n]])
            }
            S4 | S9 => {
                ctx.best[n]
                    + (ctx.pool[[v[0], n]] + ctx.pool[[v[1], n]]
                        - ctx.pool[[v[2], n]]
                        - ctx.pool[[v[3], n]])
                        * self.f
            }
            S5 | S10 => {
                ctx.pool[[v[4], n]]
                    + (ctx.pool[[v[0], n]] + ctx.pool[[v[1], n]]
                        - ctx.pool[[v[2], n]]
                        - ctx.pool[[v[3], n]])
                        * self.f
            }
        }
    }

    fn c1<F: ObjFunc>(
        &mut self,
        ctx: &Context<F>,
        tmp: &mut Array1<f64>,
        v: Array1<usize>,
        mut n: usize,
    ) {
        for _ in 0..ctx.dim {
            tmp[n] = self.formula(ctx, tmp, &v, n);
            n = (n + 1) % ctx.dim;
            if !maybe(self.cross) {
                break;
            }
        }
    }

    fn c2<F: ObjFunc>(
        &mut self,
        ctx: &Context<F>,
        tmp: &mut Array1<f64>,
        v: Array1<usize>,
        mut n: usize,
    ) {
        for lv in 0..ctx.dim {
            if !maybe(self.cross) || lv == ctx.dim - 1 {
                tmp[n] = self.formula(ctx, tmp, &v, n);
            }
            n = (n + 1) % ctx.dim;
        }
    }
}

impl Algorithm for DE {
    type Setting = DESetting;

    fn create(settings: &Self::Setting) -> Self {
        let num = match settings.strategy {
            S1 | S3 | S6 | S8 => 2,
            S2 | S7 => 3,
            S4 | S9 => 4,
            S5 | S10 => 5,
        };
        Self {
            f: settings.f,
            cross: settings.cross,
            num,
            strategy: settings.strategy.clone(),
        }
    }

    fn generation<F: ObjFunc>(&mut self, ctx: &mut Context<F>) {
        'a: for i in 0..ctx.pop_num {
            // Generate Vector
            let mut v = Array1::zeros(self.num);
            rand_vector(v.as_slice_mut().unwrap(), 0, 0, ctx.pop_num);
            // Recombination
            let mut tmp = ctx.pool.slice(s![i, ..]).to_owned();
            let n = rand_int(0, ctx.dim);
            match self.strategy {
                S1 | S2 | S3 | S4 | S5 => self.c1(ctx, &mut tmp, v, n),
                S6 | S7 | S8 | S9 | S10 => self.c2(ctx, &mut tmp, v, n),
            }
            for s in 0..ctx.dim {
                if tmp[s] > ctx.ub(s) || tmp[s] < ctx.lb(s) {
                    continue 'a;
                }
            }
            let tmp_f = ctx.func.fitness(&tmp, &ctx.report);
            if tmp_f < ctx.fitness[i] {
                ctx.assign_from(i, tmp_f, &tmp);
            }
        }
        ctx.find_best();
    }
}
