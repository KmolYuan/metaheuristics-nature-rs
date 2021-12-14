//! Real-coded Genetic Algorithm.
//!
//! (Real-valued Genetic Algorithm)
//!
//! <https://en.wikipedia.org/wiki/Genetic_algorithm>
//!
//! This method require floating point power function.
use crate::utility::prelude::*;
use alloc::{vec, vec::Vec};
use core::marker::PhantomData;

/// Real-coded Genetic Algorithm settings.
pub struct Rga<R: Respond> {
    cross: f64,
    mutate: f64,
    win: f64,
    delta: f64,
    _marker: PhantomData<R>,
}

impl<R: Respond> Rga<R> {
    impl_builders! {
        /// Crossing probability.
        fn cross(f64)
        /// Mutation probability.
        fn mutate(f64)
        /// Winning probability.
        fn win(f64)
        /// Delta factor.
        fn delta(f64)
    }
}

impl<R: Respond> Default for Rga<R> {
    fn default() -> Self {
        Self {
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

    fn algorithm(self) -> Self::Algorithm {
        Method {
            cross: self.cross,
            mutate: self.mutate,
            win: self.win,
            delta: self.delta,
            fitness_new: Vec::new(),
            pool_new: Array2::zeros((1, 1)),
        }
    }

    fn default_basic() -> BasicSetting {
        BasicSetting {
            pop_num: 500,
            ..Default::default()
        }
    }
}

#[inline(always)]
fn check<F: ObjFunc>(ctx: &mut Context<F>, s: usize, v: f64) -> f64 {
    if ctx.ub(s) < v || v < ctx.lb(s) {
        ctx.rng.rand_float(ctx.lb(s), ctx.ub(s))
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
            if !ctx.rng.maybe(self.cross) {
                continue;
            }
            use TmpId::*;
            enum TmpId {
                I1,
                I2,
                I3,
            }
            #[cfg(feature = "parallel")]
            let iter = [I1, I2, I3].into_par_iter();
            #[cfg(not(feature = "parallel"))]
            let iter = IntoIterator::into_iter([I1, I2, I3]);
            let mut v = iter
                .map(|id| {
                    let mut v = Array1::zeros(ctx.dim());
                    for s in 0..ctx.dim() {
                        let variable = match id {
                            I1 => 0.5 * ctx.pool[[i, s]] + 0.5 * ctx.pool[[i + 1, s]],
                            I2 => 1.5 * ctx.pool[[i, s]] - 0.5 * ctx.pool[[i + 1, s]],
                            I3 => -0.5 * ctx.pool[[i, s]] + 1.5 * ctx.pool[[i + 1, s]],
                        };
                        v[s] = check(ctx, s, variable);
                    }
                    let f = ctx.func.fitness(v.as_slice().unwrap(), &ctx.report);
                    (f, v)
                })
                .collect::<Vec<_>>();
            v.sort_unstable_by(|(a, _), (b, _)| a.value().partial_cmp(&b.value()).unwrap());
            ctx.assign_from(i, v[0].0.clone(), &v[0].1);
            ctx.assign_from(i + 1, v[1].0.clone(), &v[1].1);
        }
    }

    fn get_delta<F>(&self, ctx: &mut Context<F>, y: f64) -> f64
    where
        F: ObjFunc<Respond = R>,
    {
        let r = match ctx.task {
            Task::MaxGen(v) if v > 0 => ctx.report.gen as f64 / v as f64,
            _ => 1.,
        };
        #[cfg(all(feature = "std", not(feature = "libm")))]
        let pow_f = (1. - r).powf(self.delta);
        #[cfg(feature = "libm")]
        let pow_f = libm::pow(1. - r, self.delta);
        y * ctx.rng.rand() * pow_f
    }

    fn mutate<F>(&mut self, ctx: &mut Context<F>)
    where
        F: ObjFunc<Respond = R>,
    {
        for i in 0..ctx.pop_num() {
            if !ctx.rng.maybe(self.mutate) {
                continue;
            }
            let s = ctx.rng.rand_int(0, ctx.dim());
            if ctx.rng.maybe(0.5) {
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
                ctx.rng.rand_vector(&mut v, 1, 0, ctx.pop_num());
                (v[1], v[2])
            };
            if ctx.fitness[j].value() > ctx.fitness[k].value() && ctx.rng.maybe(self.win) {
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
            let i = ctx.rng.rand_int(0, ctx.pop_num());
            ctx.assign_from_best(i);
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
