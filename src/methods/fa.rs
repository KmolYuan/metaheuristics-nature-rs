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
        Self {
            alpha: 1.,
            beta_min: 1.,
            gamma: 0.01,
        }
    }
}

impl Setting for Fa {
    type Algorithm = Method;

    fn algorithm(self) -> Self::Algorithm {
        Method {
            alpha: self.alpha,
            beta_min: self.beta_min,
            gamma: self.gamma,
        }
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
        ctx: &Context<F>,
        i: usize,
        j: usize,
    ) -> (Array1<f64>, F::Fitness) {
        let mut tmp = Array1::zeros(ctx.dim());
        let pool_j = ctx.pool.slice(s![j, ..]);
        let r = {
            let mut dist = 0.;
            for s in 0..ctx.dim() {
                let diff = ctx.pool[[i, s]] - pool_j[s];
                dist += diff * diff;
            }
            dist
        };
        #[cfg(all(feature = "std", not(feature = "libm")))]
        let gamma_r = (-self.gamma * r).exp();
        #[cfg(feature = "libm")]
        let gamma_r = libm::exp(-self.gamma * r);
        let beta = self.beta_min * gamma_r;
        for s in 0..ctx.dim() {
            let step = self.alpha * (ctx.ub(s) - ctx.lb(s)) * ctx.rng.float(-0.5..0.5);
            let v = ctx.pool[[i, s]] + beta * (pool_j[s] - ctx.pool[[i, s]]) + step;
            tmp[s] = ctx.check(s, v);
        }
        let f = ctx.func.fitness(tmp.as_slice().unwrap(), ctx.adaptive);
        (tmp, f)
    }

    #[cfg(not(feature = "parallel"))]
    fn move_fireflies<F: ObjFunc>(&mut self, ctx: &mut Context<F>) {
        for i in 0..ctx.pop_num() - 1 {
            for j in i + 1..ctx.pop_num() {
                let (i, j) = if ctx.fitness[i] > ctx.fitness[j] {
                    (i, j)
                } else {
                    (j, i)
                };
                let (tmp, f) = self.move_firefly(ctx, i, j);
                if f < ctx.fitness[i] {
                    ctx.assign_from(i, f, &tmp);
                }
            }
        }
    }

    #[cfg(feature = "parallel")]
    fn move_fireflies<F: ObjFunc>(&mut self, ctx: &mut Context<F>) {
        use std::sync::Mutex;
        let fitness = Mutex::new(ctx.fitness.clone());
        let pool = Mutex::new(ctx.pool.clone());
        (0..ctx.pop_num() - 1).into_par_iter().for_each(|i| {
            (i + 1..ctx.pop_num()).into_par_iter().for_each(|j| {
                let (i, j) = if ctx.fitness[i] > ctx.fitness[j] {
                    (i, j)
                } else {
                    (j, i)
                };
                let (v, f) = self.move_firefly(ctx, i, j);
                if f < ctx.fitness[i] {
                    let mut fitness = fitness.lock().unwrap();
                    let mut pool = pool.lock().unwrap();
                    fitness[i] = f;
                    pool.slice_mut(s![i, ..]).assign(&v);
                }
            });
        });
        ctx.fitness = fitness.into_inner().unwrap();
        ctx.pool = pool.into_inner().unwrap();
    }
}

impl<F: ObjFunc> Algorithm<F> for Method {
    #[inline(always)]
    fn generation(&mut self, ctx: &mut Context<F>) {
        self.move_fireflies(ctx);
        self.alpha *= 0.95;
        ctx.find_best();
    }
}
