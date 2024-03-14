//! # Teaching Learning Based Optimization
//!
//! <https://doi.org/10.1016/j.cad.2010.12.015>
//!
//! This method require round function.
use crate::utility::prelude::*;
use core::iter::zip;

/// Teaching Learning Based Optimization type.
pub type Method = Tlbo;

/// Teaching Learning Based Optimization settings.
#[derive(Default, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "clap", derive(clap::Args))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Tlbo;

impl Tlbo {
    /// Constant default value.
    pub const fn new() -> Self {
        Self
    }
}

impl Setting for Tlbo {
    type Algorithm<F: ObjFunc> = Method;

    fn algorithm<F: ObjFunc>(self) -> Self::Algorithm<F> {
        self
    }
}

impl Method {
    fn register<F: ObjFunc>(ctx: &mut Ctx<F>, i: usize, student: &[f64]) {
        let f_new = ctx.func.fitness(student);
        if f_new < ctx.pool_f[i] {
            ctx.pool[i] = student.to_vec();
            ctx.pool_f[i] = f_new.clone();
            if f_new < ctx.best_f {
                ctx.best_f = f_new;
                ctx.best = student.to_vec();
            }
        }
    }

    fn teaching<F: ObjFunc>(&mut self, ctx: &mut Ctx<F>, rng: &Rng, i: usize, student: &mut [f64]) {
        let tf = rng.range(1f64..2.).round();
        for (s, (&[min, max], (student, (base, best)))) in zip(
            ctx.bound(),
            zip(student.iter_mut(), zip(&ctx.pool[i], &ctx.best)),
        )
        .enumerate()
        {
            let mut mean = 0.;
            for j in 0..ctx.pop_num() {
                mean += ctx.pool[j][s];
            }
            mean /= ctx.dim() as f64;
            let v = base + rng.range(1.0..ctx.dim() as f64) * (best - tf * mean);
            *student = v.clamp(min, max);
        }
        Self::register(ctx, i, student);
    }

    fn learning<F: ObjFunc>(&mut self, ctx: &mut Ctx<F>, rng: &Rng, i: usize, student: &mut [f64]) {
        let j = {
            let j = rng.ub(ctx.pop_num() - 1);
            if j >= i {
                j + 1
            } else {
                j
            }
        };
        for (&[min, max], (student, (a, b))) in zip(
            ctx.bound(),
            zip(student.iter_mut(), zip(&ctx.pool[i], &ctx.pool[j])),
        ) {
            let diff = if ctx.pool_f[j] < ctx.pool_f[i] {
                a - b
            } else {
                b - a
            };
            let v = a + rng.range(1.0..ctx.dim() as f64) * diff;
            *student = v.clamp(min, max);
        }
        Self::register(ctx, i, student);
    }
}

impl<F: ObjFunc> Algorithm<F> for Method {
    fn generation(&mut self, ctx: &mut Ctx<F>, rng: &Rng) {
        for i in 0..ctx.pop_num() {
            let mut student = vec![0.; ctx.dim()];
            self.teaching(ctx, rng, i, &mut student);
            self.learning(ctx, rng, i, &mut student);
        }
    }
}
