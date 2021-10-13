//! Real-coded Genetic Algorithm.
//!
//! (Real-valued Genetic Algorithm)
//!
//! <https://en.wikipedia.org/wiki/Genetic_algorithm>
//!
//! This method require floating point power function.
use crate::{thread_pool::ThreadPool, utility::*, *};
use alloc::{vec, vec::Vec};
use core::marker::PhantomData;

/// Real-coded Genetic Algorithm settings.
pub struct Rga<R: Respond> {
    /// Base setting.
    pub base: BasicSetting,
    /// Crossing probability.
    pub cross: f64,
    /// Mutation probability.
    pub mutate: f64,
    /// Winning probability.
    pub win: f64,
    /// Delta factor.
    pub delta: f64,
    _marker: PhantomData<R>,
}

impl<R: Respond> Default for Rga<R> {
    fn default() -> Self {
        Self {
            base: BasicSetting {
                pop_num: 500,
                ..Default::default()
            },
            cross: 0.95,
            mutate: 0.05,
            win: 0.95,
            delta: 5.,
            _marker: PhantomData,
        }
    }
}

impl<R: Respond> Setting for Rga<R> {
    type Algorithm = Method<R>;

    fn base(&self) -> &BasicSetting {
        &self.base
    }

    fn base_mut(&mut self) -> &mut BasicSetting {
        &mut self.base
    }

    fn create(self) -> Self::Algorithm {
        Method {
            cross: self.cross,
            mutate: self.mutate,
            win: self.win,
            delta: self.delta,
            fitness_new: Vec::new(),
            pool_new: Array2::zeros((1, 1)),
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
pub struct Method<R: Respond> {
    cross: f64,
    mutate: f64,
    win: f64,
    delta: f64,
    fitness_new: Vec<R>,
    pool_new: Array2<f64>,
}

impl<R: Respond> Method<R> {
    fn crossover<F>(&mut self, ctx: &mut Context<F>)
    where
        F: ObjFunc<Respond = R>,
    {
        for i in (0..(ctx.pop_num() - 1)).step_by(2) {
            if !maybe(self.cross) {
                continue;
            }
            let mut tmp = Array2::zeros((3, ctx.dim()));
            for s in 0..ctx.dim() {
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
            let mut f_tmp = Vec::with_capacity(3);
            f_tmp.extend(tasks.join());
            if f_tmp[0].value() > f_tmp[1].value() {
                f_tmp.swap(0, 1);
                for j in 0..2 {
                    tmp.swap([0, j], [1, j]);
                }
            }
            if f_tmp[0].value() > f_tmp[2].value() {
                f_tmp.swap(0, 2);
                for j in 0..2 {
                    tmp.swap([0, j], [2, j]);
                }
            }
            if f_tmp[1].value() > f_tmp[2].value() {
                f_tmp.swap(1, 2);
                for j in 0..2 {
                    tmp.swap([1, j], [2, j]);
                }
            }
            ctx.assign_from(i, f_tmp[0].clone(), tmp.slice(s![0, ..]));
            ctx.assign_from(i + 1, f_tmp[1].clone(), tmp.slice(s![1, ..]));
        }
    }

    fn get_delta<F>(&self, ctx: &Context<F>, y: f64) -> f64
    where
        F: ObjFunc<Respond = R>,
    {
        let r = match ctx.task {
            Task::MaxGen(v) if v > 0 => ctx.report.gen as f64 / v as f64,
            _ => 1.,
        };
        #[cfg(feature = "std")]
        #[allow(unused)]
        let pow_f = (1. - r).powf(self.delta);
        #[cfg(feature = "libm")]
        let pow_f = libm::pow(1. - r, self.delta);
        y * rand() * pow_f
    }

    fn mutate<F>(&mut self, ctx: &mut Context<F>)
    where
        F: ObjFunc<Respond = R>,
    {
        for i in 0..ctx.pop_num() {
            if !maybe(self.mutate) {
                continue;
            }
            let s = rand_int(0, ctx.dim());
            if maybe(0.5) {
                ctx.pool[[i, s]] += self.get_delta(ctx, ctx.ub(s) - ctx.pool[[i, s]]);
            } else {
                ctx.pool[[i, s]] -= self.get_delta(ctx, ctx.pool[[i, s]] - ctx.lb(s));
            }
            ctx.fitness(i);
        }
        ctx.find_best();
    }

    fn select<F>(&mut self, ctx: &mut Context<F>)
    where
        F: ObjFunc<Respond = R>,
    {
        for i in 0..ctx.pop_num() {
            let (j, k) = {
                let mut v = [i, 0, 0];
                rand_vector(&mut v, 1, 0, ctx.pop_num());
                (v[1], v[2])
            };
            if ctx.fitness[j].value() > ctx.fitness[k].value() && maybe(self.win) {
                self.fitness_new[i] = ctx.fitness[k].clone();
                self.pool_new
                    .slice_mut(s![i, ..])
                    .assign(&ctx.pool.slice(s![k, ..]));
            } else {
                self.fitness_new[i] = ctx.fitness[j].clone();
                self.pool_new
                    .slice_mut(s![i, ..])
                    .assign(&ctx.pool.slice(s![j, ..]));
            }
            ctx.fitness = self.fitness_new.clone();
            ctx.pool.assign(&self.pool_new);
            ctx.assign_from_best(rand_int(0, ctx.pop_num()));
        }
    }
}

impl<F, R> Algorithm<F> for Method<R>
where
    F: ObjFunc<Respond = R>,
    R: Respond,
{
    #[inline(always)]
    fn init(&mut self, ctx: &mut Context<F>) {
        self.pool_new = Array2::zeros(ctx.pool.raw_dim());
        self.fitness_new = vec![R::INFINITY; ctx.fitness.len()];
    }

    #[inline(always)]
    fn generation(&mut self, ctx: &mut Context<F>) {
        self.select(ctx);
        self.crossover(ctx);
        self.mutate(ctx);
    }
}
