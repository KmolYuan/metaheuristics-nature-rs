//! # Teaching Learning Based Optimization
//!
//! <https://doi.org/10.1016/j.cad.2010.12.015>
//!
//! This method require round function.
use crate::prelude::*;
use alloc::vec::Vec;
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
    fn register<F: ObjFunc>(ctx: &mut Ctx<F>, i: usize, student: Vec<f64>) {
        let f_new = ctx.fitness(&student);
        if f_new.is_dominated(&ctx.pool_f[i]) {
            ctx.set_from(i, student, f_new);
            ctx.best.update(&ctx.pool[i], &ctx.pool_f[i]);
        }
    }

    fn teaching<F: ObjFunc>(&mut self, ctx: &mut Ctx<F>, rng: &Rng, i: usize) {
        let tf = rng.range(1f64..2.).round();
        let best = ctx.best.sample_xs(rng);
        let student = zip(ctx.bound(), zip(&ctx.pool[i], best))
            .enumerate()
            .map(|(s, (&[min, max], (base, best)))| {
                let mut mean = 0.;
                for other in &ctx.pool {
                    mean += other[s];
                }
                let dim = ctx.dim() as f64;
                mean /= dim;
                (base + rng.range(1.0..dim) * (best - tf * mean)).clamp(min, max)
            })
            .collect();
        Self::register(ctx, i, student);
    }

    fn learning<F: ObjFunc>(&mut self, ctx: &mut Ctx<F>, rng: &Rng, i: usize) {
        let j = {
            let j = rng.ub(ctx.pop_num() - 1);
            if j >= i {
                j + 1
            } else {
                j
            }
        };
        let student = zip(ctx.bound(), zip(&ctx.pool[i], &ctx.pool[j]))
            .map(|(&[min, max], (a, b))| {
                let diff = if ctx.pool_f[j].is_dominated(&ctx.pool_f[i]) {
                    a - b
                } else {
                    b - a
                };
                (a + rng.range(1.0..ctx.dim() as f64) * diff).clamp(min, max)
            })
            .collect();
        Self::register(ctx, i, student);
    }
}

impl<F: ObjFunc> Algorithm<F> for Method {
    fn generation(&mut self, ctx: &mut Ctx<F>, rng: &Rng) {
        for i in 0..ctx.pop_num() {
            self.teaching(ctx, rng, i);
            self.learning(ctx, rng, i);
        }
    }
}
