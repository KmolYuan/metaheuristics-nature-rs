//! Real-coded Genetic Algorithm.
//!
//! (Real-valued Genetic Algorithm)
//!
//! <https://en.wikipedia.org/wiki/Genetic_algorithm>
use crate::{thread_pool::ThreadPool, utility::*, *};
use ndarray::s;

setting! {
    /// Real-coded Genetic Algorithm settings.
    pub struct Rga {
        @base,
        @pop_num = 500,
        /// Crossing probability.
        cross: f64 = 0.95,
        /// Mutation probability.
        mutate: f64 = 0.05,
        /// Winning probability.
        win: f64 = 0.95,
        /// Delta factor.
        delta: f64 = 5.,
    }
}

impl Setting for Rga {
    type Algorithm = Method;

    fn base(&self) -> &BasicSetting {
        &self.base
    }

    fn create(self) -> Self::Algorithm {
        Method {
            cross: self.cross,
            mutate: self.mutate,
            win: self.win,
            delta: self.delta,
            pool_new: Array2::zeros((1, 1)),
            fitness_new: Array1::ones(1) * f64::INFINITY,
        }
    }
}

#[inline(always)]
fn check<F: ObjFunc>(ctx: &Context<F>, s: usize, v: f64) -> f64 {
    if ctx.ub(s) < v || v < ctx.lb(s) {
        rand_float(ctx.lb(s), ctx.ub(s))
    } else {
        v
    }
}

/// Real-coded Genetic Algorithm type.
pub struct Method {
    cross: f64,
    mutate: f64,
    win: f64,
    delta: f64,
    fitness_new: Array1<f64>,
    pool_new: Array2<f64>,
}

impl Method {
    fn crossover<F: ObjFunc>(&mut self, ctx: &mut Context<F>) {
        for i in (0..(ctx.pop_num - 1)).step_by(2) {
            if !maybe(self.cross) {
                continue;
            }
            let mut tmp = Array2::zeros((3, ctx.dim));
            let mut f_tmp = Array1::zeros(3);
            for s in 0..ctx.dim {
                tmp[[0, s]] = 0.5 * ctx.pool[[i, s]] + 0.5 * ctx.pool[[i + 1, s]];
                let v = 1.5 * ctx.pool[[i, s]] - 0.5 * ctx.pool[[i + 1, s]];
                tmp[[1, s]] = check(ctx, s, v);
                let v = -0.5 * ctx.pool[[i, s]] + 1.5 * ctx.pool[[i + 1, s]];
                tmp[[2, s]] = check(ctx, s, v);
            }
            let mut tasks = ThreadPool::new();
            for j in 0..3 {
                tasks.insert(
                    j,
                    ctx.func.clone(),
                    ctx.report.clone(),
                    tmp.slice(s![j, ..]),
                );
            }
            for (j, f) in tasks {
                f_tmp[j] = f;
            }
            if f_tmp[0] > f_tmp[1] {
                f_tmp.swap(0, 1);
                for j in 0..2 {
                    tmp.swap([0, j], [1, j]);
                }
            }
            if f_tmp[0] > f_tmp[2] {
                f_tmp.swap(0, 2);
                for j in 0..2 {
                    tmp.swap([0, j], [2, j]);
                }
            }
            if f_tmp[1] > f_tmp[2] {
                f_tmp.swap(1, 2);
                for j in 0..2 {
                    tmp.swap([1, j], [2, j]);
                }
            }
            ctx.assign_from(i, f_tmp[0], tmp.slice(s![0, ..]));
            ctx.assign_from(i + 1, f_tmp[1], tmp.slice(s![1, ..]));
        }
    }

    fn get_delta<F: ObjFunc>(&self, ctx: &Context<F>, y: f64) -> f64 {
        let r = match ctx.task {
            Task::MaxGen(v) if v > 0 => ctx.report.gen as f64 / v as f64,
            _ => 1.,
        };
        y * rand() * (1. - r).powf(self.delta)
    }

    fn mutate<F: ObjFunc>(&mut self, ctx: &mut Context<F>) {
        for i in 0..ctx.pop_num {
            if !maybe(self.mutate) {
                continue;
            }
            let s = rand_int(0, ctx.dim);
            if maybe(0.5) {
                ctx.pool[[i, s]] += self.get_delta(ctx, ctx.ub(s) - ctx.pool[[i, s]]);
            } else {
                ctx.pool[[i, s]] -= self.get_delta(ctx, ctx.pool[[i, s]] - ctx.lb(s));
            }
            ctx.fitness(i);
        }
        ctx.find_best();
    }

    fn select<F: ObjFunc>(&mut self, ctx: &mut Context<F>) {
        for i in 0..ctx.pop_num {
            let (j, k) = {
                let mut v = [i, 0, 0];
                rand_vector(&mut v, 1, 0, ctx.pop_num);
                (v[1], v[2])
            };
            if ctx.fitness[j] > ctx.fitness[k] && maybe(self.win) {
                self.fitness_new[i] = ctx.fitness[k];
                self.pool_new
                    .slice_mut(s![i, ..])
                    .assign(&ctx.pool.slice(s![k, ..]));
            } else {
                self.fitness_new[i] = ctx.fitness[j];
                self.pool_new
                    .slice_mut(s![i, ..])
                    .assign(&ctx.pool.slice(s![j, ..]));
            }
            ctx.fitness.assign(&self.fitness_new);
            ctx.pool.assign(&self.pool_new);
            ctx.assign_from_best(rand_int(0, ctx.pop_num));
        }
    }
}

impl Algorithm for Method {
    #[inline(always)]
    fn init<F: ObjFunc>(&mut self, ctx: &mut Context<F>) {
        self.pool_new = Array2::zeros(ctx.pool.raw_dim());
        self.fitness_new = Array1::ones(ctx.fitness.raw_dim()) * f64::INFINITY;
    }

    #[inline(always)]
    fn generation<F: ObjFunc>(&mut self, ctx: &mut Context<F>) {
        self.select(ctx);
        self.crossover(ctx);
        self.mutate(ctx);
    }
}
