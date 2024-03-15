//! # Real-coded Genetic Algorithm
//!
//! Aka Real-valued Genetic Algorithm.
//!
//! <https://en.wikipedia.org/wiki/Genetic_algorithm>
//!
//! This method require floating point power function.
use crate::prelude::*;
use alloc::vec::Vec;
use core::iter::zip;

/// Real-coded Genetic Algorithm type.
pub type Method = Rga;

const DEF: Rga = Rga { cross: 0.95, mutate: 0.05, win: 0.95, delta: 5. };

/// Real-coded Genetic Algorithm settings.
#[derive(Clone, PartialEq)]
#[cfg_attr(feature = "clap", derive(clap::Args))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct Rga {
    /// Crossover rate
    #[cfg_attr(feature = "clap", clap(long, default_value_t = DEF.cross))]
    pub cross: f64,
    /// Mutation rate
    #[cfg_attr(feature = "clap", clap(long, default_value_t = DEF.mutate))]
    pub mutate: f64,
    /// Win rate
    #[cfg_attr(feature = "clap", clap(long, default_value_t = DEF.win))]
    pub win: f64,
    /// Delta
    #[cfg_attr(feature = "clap", clap(long, default_value_t = DEF.delta))]
    pub delta: f64,
}

impl Rga {
    /// Constant default value.
    pub const fn new() -> Self {
        DEF
    }

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

impl Default for Rga {
    fn default() -> Self {
        DEF
    }
}

impl Setting for Rga {
    type Algorithm<F: ObjFunc> = Method;

    fn algorithm<F: ObjFunc>(self) -> Self::Algorithm<F> {
        self
    }

    fn default_pop() -> usize {
        500
    }
}

impl Method {
    fn get_delta(&self, gen: u64, rng: &Rng, y: f64) -> f64 {
        let r = if gen < 100 { gen as f64 / 100. } else { 1. };
        rng.rand() * y * (1. - r).powf(self.delta)
    }
}

impl<F: ObjFunc> Algorithm<F> for Method {
    fn generation(&mut self, ctx: &mut Ctx<F>, rng: &Rng) {
        // Select
        let mut pool = ctx.pool.clone();
        let mut pool_f = ctx.pool_f.clone();
        for (xs, f) in zip(&mut pool, &mut pool_f) {
            let [a, b] = rng.array(0..ctx.pop_num());
            let i = if ctx.pool_f[a] < ctx.pool_f[b] { a } else { b };
            if rng.maybe(self.win) {
                *xs = ctx.pool[i].clone();
                *f = ctx.pool_f[i].clone();
            }
        }
        ctx.pool = pool;
        ctx.pool_f = pool_f;
        ctx.set_from_best(rng.ub(ctx.pop_num()));
        // Crossover
        for i in (0..ctx.pop_num() - 1).step_by(2) {
            if !rng.maybe(self.cross) {
                continue;
            }
            #[cfg(not(feature = "rayon"))]
            let iter = 0..3;
            #[cfg(feature = "rayon")]
            let iter = (0..3).into_par_iter();
            let mut ret = iter
                .zip(rng.stream(3))
                .map(|(id, rng)| {
                    let xs = zip(ctx.func.bound(), zip(&ctx.pool[i], &ctx.pool[i + 1]))
                        .map(|(&[min, max], (a, b))| {
                            let v = match id {
                                0 => 0.5 * (a + b),
                                1 => 1.5 * a - 0.5 * b,
                                _ => -0.5 * a + 1.5 * b,
                            };
                            rng.clamp(v, min..=max)
                        })
                        .collect::<Vec<_>>();
                    let f = ctx.func.fitness(&xs);
                    (f, xs)
                })
                .collect::<Vec<_>>();
            ret.sort_unstable_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap());
            let [(t1_f, t1_x), (t2_f, t2_x)] = [ret.remove(0), ret.remove(0)];
            ctx.set_from(i, t1_x, t1_f);
            ctx.set_from(i + 1, t2_x, t2_f);
        }
        // Mutate
        let dim = ctx.dim();
        for (xs, f) in zip(&mut ctx.pool, &mut ctx.pool_f) {
            if !rng.maybe(self.mutate) {
                continue;
            }
            let s = rng.ub(dim);
            if rng.maybe(0.5) {
                xs[s] += self.get_delta(ctx.gen, rng, ctx.func.ub(s) - xs[s]);
            } else {
                xs[s] -= self.get_delta(ctx.gen, rng, xs[s] - ctx.func.lb(s));
            }
            *f = ctx.func.fitness(xs);
        }
        ctx.find_best();
    }
}
