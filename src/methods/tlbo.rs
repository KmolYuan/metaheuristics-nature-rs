//! Teaching Learning Based Optimization.
//!
//! <https://doi.org/10.1016/j.cad.2010.12.015>
//!
//! This method require round function.
use crate::utility::prelude::*;

/// Teaching Learning Based Optimization settings.
#[derive(Default)]
pub struct Tlbo;

impl Setting for Tlbo {
    type Algorithm = Method;

    fn algorithm(self) -> Self::Algorithm {
        Method
    }
}

/// Teaching Learning Based Optimization type.
pub struct Method;

impl Method {
    fn register<F: ObjFunc>(ctx: &mut Ctx<F>, i: usize, student: &Array1<f64>) {
        let f_new = ctx.func.fitness(student.as_slice().unwrap(), ctx.adaptive);
        if f_new < ctx.pool_f[i] {
            ctx.pool.slice_mut(s![i, ..]).assign(student);
            ctx.pool_f[i] = f_new.clone();
        }
        if f_new < ctx.best_f {
            ctx.set_best(i);
        }
    }

    fn teaching<F: ObjFunc>(&mut self, ctx: &mut Ctx<F>, i: usize, student: &mut Array1<f64>) {
        let tf = (ctx.rng.rand() + 1.).round();
        for s in 0..ctx.dim() {
            let mut mean = 0.;
            for j in 0..ctx.pop_num() {
                mean += ctx.pool[[j, s]];
            }
            mean /= ctx.dim() as f64;
            let v =
                ctx.pool[[i, s]] + ctx.rng.float(1.0..ctx.dim() as f64) * (ctx.best[s] - tf * mean);
            student[s] = ctx.clamp(s, v);
        }
        Self::register(ctx, i, student);
    }

    fn learning<F: ObjFunc>(&mut self, ctx: &mut Ctx<F>, i: usize, student: &mut Array1<f64>) {
        let j = {
            let j = ctx.rng.int(0..ctx.pop_num() - 1);
            if j >= i {
                j + 1
            } else {
                j
            }
        };
        for s in 0..ctx.dim() {
            let diff = if ctx.pool_f[j] < ctx.pool_f[i] {
                ctx.pool[[i, s]] - ctx.pool[[j, s]]
            } else {
                ctx.pool[[j, s]] - ctx.pool[[i, s]]
            };
            let v = ctx.pool[[i, s]] + ctx.rng.float(1.0..ctx.dim() as f64) * diff;
            student[s] = ctx.clamp(s, v);
        }
        Self::register(ctx, i, student);
    }
}

impl<F: ObjFunc> Algorithm<F> for Method {
    #[inline(always)]
    fn generation(&mut self, ctx: &mut Ctx<F>) {
        for i in 0..ctx.pop_num() {
            let mut student = Array1::zeros(ctx.dim());
            self.teaching(ctx, i, &mut student);
            self.learning(ctx, i, &mut student);
        }
    }
}
