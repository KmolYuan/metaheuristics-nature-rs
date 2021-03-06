//! Real-coded Genetic Algorithm.
//!
//! (Real-valued Genetic Algorithm)
//!
//! <https://en.wikipedia.org/wiki/Genetic_algorithm>
//!
//! This method require floating point power function.
use crate::utility::prelude::*;
use alloc::vec::Vec;
use core::marker::PhantomData;

/// Real-coded Genetic Algorithm settings.
pub struct Rga<F: Fitness> {
    cross: f64,
    mutate: f64,
    win: f64,
    delta: f64,
    _marker: PhantomData<F>,
}

impl<F: Fitness> Rga<F> {
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

impl<F: Fitness> Default for Rga<F> {
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

impl<F: Fitness> Setting for Rga<F> {
    type Algorithm = Method<F>;

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

    fn default_pop() -> usize {
        500
    }
}

/// Real-coded Genetic Algorithm type.
pub struct Method<F: Fitness> {
    cross: f64,
    mutate: f64,
    win: f64,
    delta: f64,
    fitness_new: Vec<F>,
    pool_new: Array2<f64>,
}

impl<Ft: Fitness> Method<Ft> {
    fn get_delta<F>(&self, ctx: &Context<F>, y: f64) -> f64
    where
        F: ObjFunc<Fitness = Ft>,
    {
        let r = if ctx.gen < 100 {
            ctx.gen as f64 / 100.
        } else {
            1.
        };
        y * ctx.rng.rand() * (1. - r).powf(self.delta)
    }
}

impl<F: ObjFunc> Algorithm<F> for Method<F::Fitness> {
    #[inline(always)]
    fn init(&mut self, ctx: &mut Context<F>) {
        self.pool_new = Array2::zeros(ctx.pool.raw_dim());
        self.fitness_new = ctx.fitness.clone();
    }

    #[inline(always)]
    fn generation(&mut self, ctx: &mut Context<F>) {
        // Select
        for i in 0..ctx.pop_num() {
            let (j, k) = {
                let mut v = [i, 0, 0];
                ctx.rng.vector(&mut v, 1, 0..ctx.pop_num());
                (v[1], v[2])
            };
            if ctx.fitness[j] > ctx.fitness[k] && ctx.rng.maybe(self.win) {
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
            let i = ctx.rng.int(0..ctx.pop_num());
            ctx.assign_from_best(i);
        }
        // Crossover
        for i in (0..ctx.pop_num() - 1).step_by(2) {
            if !ctx.rng.maybe(self.cross) {
                continue;
            }
            enum Id {
                I1,
                I2,
                I3,
            }
            #[cfg(feature = "rayon")]
            let iter = [Id::I1, Id::I2, Id::I3].into_par_iter();
            #[cfg(not(feature = "rayon"))]
            let iter = [Id::I1, Id::I2, Id::I3].into_iter();
            let mut v = iter
                .map(|id| {
                    let mut v = Array1::zeros(ctx.dim());
                    for s in 0..ctx.dim() {
                        let var = match id {
                            Id::I1 => 0.5 * ctx.pool[[i, s]] + 0.5 * ctx.pool[[i + 1, s]],
                            Id::I2 => 1.5 * ctx.pool[[i, s]] - 0.5 * ctx.pool[[i + 1, s]],
                            Id::I3 => -0.5 * ctx.pool[[i, s]] + 1.5 * ctx.pool[[i + 1, s]],
                        };
                        v[s] = if ctx.ub(s) < var || var < ctx.lb(s) {
                            ctx.rng.float(ctx.lb(s)..ctx.ub(s))
                        } else {
                            var
                        };
                    }
                    let f = ctx.func.fitness(v.as_slice().unwrap(), ctx.adaptive);
                    (f, v)
                })
                .collect::<Vec<_>>();
            v.sort_unstable_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap());
            ctx.assign_from(i, v[0].0.clone(), &v[0].1);
            ctx.assign_from(i + 1, v[1].0.clone(), &v[1].1);
        }
        // Mutate
        for i in 0..ctx.pop_num() {
            if !ctx.rng.maybe(self.mutate) {
                continue;
            }
            let s = ctx.rng.int(0..ctx.dim());
            if ctx.rng.maybe(0.5) {
                ctx.pool[[i, s]] += self.get_delta(ctx, ctx.ub(s) - ctx.pool[[i, s]]);
            } else {
                ctx.pool[[i, s]] -= self.get_delta(ctx, ctx.pool[[i, s]] - ctx.lb(s));
            }
            ctx.fitness(i);
        }
        ctx.find_best();
    }
}
