//! Teaching Learning Based Optimization.
//!
//! <https://doi.org/10.1016/j.cad.2010.12.015>
use crate::{utility::*, *};
use ndarray::s;

setting! {
    /// Teaching Learning Based Optimization settings.
    pub struct Tlbo(@base);
}

impl Setting for Tlbo {
    type Algorithm = Method;

    fn base(&self) -> &BasicSetting {
        &self.0
    }

    fn create(self) -> Self::Algorithm {
        Method
    }
}

/// Teaching Learning Based Optimization type.
pub struct Method;

impl Method {
    fn register<F: ObjFunc>(ctx: &mut Context<F>, i: usize, student: &Array1<f64>) {
        let f_new = ctx.func.fitness(student, &ctx.report);
        if f_new < ctx.fitness[i] {
            ctx.pool.slice_mut(s![i, ..]).assign(student);
            ctx.fitness[i] = f_new;
        }
        if f_new < ctx.report.best_f {
            ctx.set_best(i);
        }
    }

    fn teaching<F: ObjFunc>(&mut self, ctx: &mut Context<F>, i: usize, student: &mut Array1<f64>) {
        let tf = f64::round(rand() + 1.);
        for s in 0..ctx.dim {
            let mut mean = 0.;
            for j in 0..ctx.pop_num {
                mean += ctx.pool[[j, s]];
            }
            mean /= ctx.dim as f64;
            let v = ctx.pool[[i, s]] + rand_float(1., ctx.dim as f64) * (ctx.best[s] - tf * mean);
            student[s] = ctx.check(s, v);
        }
        Self::register(ctx, i, student);
    }

    fn learning<F: ObjFunc>(&mut self, ctx: &mut Context<F>, i: usize, student: &mut Array1<f64>) {
        let j = {
            let j = rand_int(0, ctx.pop_num - 1);
            if j >= i {
                j + 1
            } else {
                j
            }
        };
        for s in 0..ctx.dim {
            let diff = if ctx.fitness[j] < ctx.fitness[i] {
                ctx.pool[[i, s]] - ctx.pool[[j, s]]
            } else {
                ctx.pool[[j, s]] - ctx.pool[[i, s]]
            };
            let v = ctx.pool[[i, s]] + rand_float(1., ctx.dim as f64) * diff;
            student[s] = ctx.check(s, v);
        }
        Self::register(ctx, i, student);
    }
}

impl Algorithm for Method {
    #[inline(always)]
    fn generation<F: ObjFunc>(&mut self, ctx: &mut Context<F>) {
        for i in 0..ctx.pop_num {
            let mut student = Array1::zeros(ctx.dim);
            self.teaching(ctx, i, &mut student);
            self.learning(ctx, i, &mut student);
        }
    }
}
