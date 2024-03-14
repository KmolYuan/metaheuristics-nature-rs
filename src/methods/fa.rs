//! # Firefly Algorithm
//!
//! <https://en.wikipedia.org/wiki/Firefly_algorithm>
//!
//! This method require exponential function.
use crate::utility::prelude::*;
use alloc::vec::Vec;
use core::iter::zip;

/// Firefly Algorithm type.
pub type Method = Fa;

const DEF: Fa = Fa { alpha: 1., beta_min: 1., gamma: 0.01 };

/// Firefly Algorithm settings.
#[derive(Clone, PartialEq)]
#[cfg_attr(feature = "clap", derive(clap::Args))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct Fa {
    /// Alpha factor
    #[cfg_attr(feature = "clap", clap(long, default_value_t = DEF.alpha))]
    pub alpha: f64,
    /// Min beta value
    #[cfg_attr(feature = "clap", clap(long, default_value_t = DEF.beta_min))]
    pub beta_min: f64,
    /// Gamma factor
    #[cfg_attr(feature = "clap", clap(long, default_value_t = DEF.gamma))]
    pub gamma: f64,
}

impl Fa {
    /// Constant default value.
    pub const fn new() -> Self {
        DEF
    }

    impl_builders! {
        /// Alpha factor.
        fn alpha(f64)
        /// Minimum beta factor.
        fn beta_min(f64)
        /// Gamma factor.
        fn gamma(f64)
    }
}

impl Default for Fa {
    fn default() -> Self {
        DEF
    }
}

impl Setting for Fa {
    type Algorithm<F: ObjFunc> = Method;

    fn algorithm<F: ObjFunc>(self) -> Self::Algorithm<F> {
        self
    }

    fn default_pop() -> usize {
        80
    }
}

impl Method {
    fn move_firefly<F: ObjFunc>(
        &self,
        ctx: &Ctx<F>,
        rng: &Rng,
        i: usize,
        j: usize,
    ) -> (Vec<f64>, F::Fitness) {
        let (i, j) = if ctx.pool_f[i] > ctx.pool_f[j] {
            (i, j)
        } else {
            (j, i)
        };
        let r = zip(&ctx.pool[i], &ctx.pool[j])
            .map(|(a, b)| a - b)
            .map(|v| v * v)
            .sum::<f64>();
        let beta = self.beta_min * (-self.gamma * r).exp();
        let xs = zip(ctx.func.bound(), zip(&ctx.pool[i], &ctx.pool[j]))
            .map(|(&[min, max], (a, b))| {
                let step = self.alpha * (max - min) * rng.range(-0.5..0.5);
                let surround = a + beta * (b - a);
                (surround + step).clamp(min, max)
            })
            .collect::<Vec<_>>();
        let f = ctx.func.fitness(&xs);
        (xs, f)
    }
}

impl<F: ObjFunc> Algorithm<F> for Method {
    fn generation(&mut self, ctx: &mut Ctx<F>, rng: &Rng) {
        // Move fireflies
        let mut fitness = ctx.pool_f.clone();
        let mut pool = ctx.pool.clone();
        #[cfg(not(feature = "rayon"))]
        let iter = fitness.iter_mut();
        #[cfg(feature = "rayon")]
        let iter = fitness.par_iter_mut();
        iter.zip(&mut pool)
            .zip(rng.stream(ctx.pop_num()))
            .enumerate()
            .for_each(|(i, ((fitness, pool), rng))| {
                for j in i + 1..ctx.pop_num() {
                    let (v, f) = self.move_firefly(ctx, &rng, i, j);
                    if f < *fitness {
                        *fitness = f;
                        *pool = v;
                    }
                }
            });
        ctx.pool_f = fitness;
        ctx.pool = pool;
        self.alpha *= 0.95;
        ctx.find_best();
    }
}
