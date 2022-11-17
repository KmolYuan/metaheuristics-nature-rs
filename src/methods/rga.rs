//! # Real-coded Genetic Algorithm
//!
//! Aka Real-valued Genetic Algorithm.
//!
//! <https://en.wikipedia.org/wiki/Genetic_algorithm>
//!
//! This method require floating point power function.
use crate::utility::prelude::*;
use alloc::vec::Vec;

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
    impl_builders! {
        default,
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
    fn get_delta<F>(&self, ctx: &Ctx<F>, y: f64) -> f64
    where
        F: ObjFunc,
    {
        let r = if ctx.gen < 100 {
            ctx.gen as f64 / 100.
        } else {
            1.
        };
        ctx.rng.rand() * y * (1. - r).powf(self.delta)
    }
}

impl<F: ObjFunc> Algorithm<F> for Method {
    fn generation(&mut self, ctx: &mut Ctx<F>) {
        // Select
        let mut pool = ctx.pool.clone();
        let mut pool_f = ctx.pool_f.clone();
        pool.axis_iter_mut(Axis(0))
            .zip(pool_f.iter_mut())
            .for_each(|(mut selected, f)| {
                let [a, b] = ctx.rng.array(0..ctx.pop_num());
                let i = if ctx.pool_f[a] < ctx.pool_f[b] { a } else { b };
                if ctx.rng.maybe(self.win) {
                    *f = ctx.pool_f[i].clone();
                    selected.assign(&ctx.pool.slice(s![i, ..]));
                }
            });
        ctx.pool = pool;
        ctx.pool_f = pool_f;
        ctx.assign_from_best(ctx.rng.ub(ctx.pop_num()));
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
                            Id::I1 => 0.5 * (ctx.pool[[i, s]] + ctx.pool[[i + 1, s]]),
                            Id::I2 => 1.5 * ctx.pool[[i, s]] - 0.5 * ctx.pool[[i + 1, s]],
                            Id::I3 => -0.5 * ctx.pool[[i, s]] + 1.5 * ctx.pool[[i + 1, s]],
                        };
                        v[s] = ctx.clamp_rand(s, var);
                    }
                    let f = ctx.func.fitness(v.as_slice().unwrap());
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
            let s = ctx.rng.ub(ctx.dim());
            if ctx.rng.maybe(0.5) {
                ctx.pool[[i, s]] += self.get_delta(ctx, ctx.func.ub(s) - ctx.pool[[i, s]]);
            } else {
                ctx.pool[[i, s]] -= self.get_delta(ctx, ctx.pool[[i, s]] - ctx.func.lb(s));
            }
            ctx.fitness(i);
        }
        ctx.find_best();
    }
}
