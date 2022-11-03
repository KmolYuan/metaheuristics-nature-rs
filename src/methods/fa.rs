//! Firefly Algorithm.
//!
//! <https://en.wikipedia.org/wiki/Firefly_algorithm>
//!
//! This method require exponential function.
use crate::utility::prelude::*;

/// Firefly Algorithm settings.
pub struct Fa {
    alpha: f64,
    beta_min: f64,
    gamma: f64,
}

impl Fa {
    impl_builders! {
        default,
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
        Self { alpha: 1., beta_min: 1., gamma: 0.01 }
    }
}

impl Setting for Fa {
    type Algorithm = Method;

    fn algorithm(self) -> Self::Algorithm {
        let Self { alpha, beta_min, gamma } = self;
        Method { alpha, beta_min, gamma }
    }

    fn default_pop() -> usize {
        80
    }
}

/// Firefly Algorithm type.
pub struct Method {
    alpha: f64,
    beta_min: f64,
    gamma: f64,
}

impl Method {
    fn move_firefly<F: ObjFunc>(
        &self,
        ctx: &Ctx<F>,
        i: usize,
        j: usize,
    ) -> (Array1<f64>, F::Fitness) {
        let (i, j) = if ctx.pool_f[i] > ctx.pool_f[j] {
            (i, j)
        } else {
            (j, i)
        };
        let mut v = Array1::zeros(ctx.dim());
        let r = (&ctx.pool.slice(s![i, ..]) - &ctx.pool.slice(s![j, ..]))
            .mapv(|v| v * v)
            .sum();
        let beta = self.beta_min * (-self.gamma * r).exp();
        for s in 0..ctx.dim() {
            let step = self.alpha * ctx.bound_width(s) * ctx.rng.range(-0.5..0.5);
            let surround = ctx.pool[[i, s]] + beta * (ctx.pool[[j, s]] - ctx.pool[[i, s]]);
            v[s] = ctx.clamp(s, surround + step);
        }
        let f = ctx.func.fitness(v.as_slice().unwrap());
        (v, f)
    }

    fn move_fireflies<F: ObjFunc>(&mut self, ctx: &mut Ctx<F>) {
        let mut fitness = ctx.pool_f.clone();
        let mut pool = ctx.pool.clone();
        #[cfg(feature = "rayon")]
        let iter = fitness.par_iter_mut();
        #[cfg(not(feature = "rayon"))]
        let iter = fitness.iter_mut();
        iter.zip(pool.axis_iter_mut(Axis(0)))
            .enumerate()
            .for_each(|(i, (fitness, mut pool))| {
                for j in i + 1..ctx.pop_num() {
                    let (v, f) = self.move_firefly(ctx, i, j);
                    if f < *fitness {
                        *fitness = f;
                        pool.assign(&v);
                    }
                }
            });
        ctx.pool_f = fitness;
        ctx.pool = pool;
    }
}

impl<F: ObjFunc> Algorithm<F> for Method {
    #[inline(always)]
    fn generation(&mut self, ctx: &mut Ctx<F>) {
        self.move_fireflies(ctx);
        self.alpha *= 0.95;
        ctx.find_best();
    }
}
