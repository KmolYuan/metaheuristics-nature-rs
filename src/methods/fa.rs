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
            alpha: 0.05,
            beta_min: 0.2,
            gamma: 1.,
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
            beta0: 1.,
        }
    }

    fn default_basic() -> BasicSetting {
        BasicSetting {
            pop_num: 80,
            ..Default::default()
        }
    }
}

/// Firefly Algorithm type.
pub struct Method {
    alpha: f64,
    beta_min: f64,
    gamma: f64,
    beta0: f64,
}

impl Method {
    fn move_fireflies<F: ObjFunc>(&mut self, ctx: &mut Context<F>) {
        for (i, j) in product(0..ctx.pop_num(), 0..ctx.pop_num()) {
            if ctx.fitness[i] <= ctx.fitness[j] {
                continue;
            }
            let mut tmp = Array1::zeros(ctx.dim());
            let pool_j = if i == j {
                ctx.best.view()
            } else {
                ctx.pool.slice(s![j, ..])
            };
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
            let beta = (self.beta0 - self.beta_min) * gamma_r + self.beta_min;
            for s in 0..ctx.dim() {
                let v = ctx.pool[[i, s]]
                    + beta * (pool_j[s] - ctx.pool[[i, s]])
                    + self.alpha * (ctx.ub(s) - ctx.lb(s)) * ctx.rng.float(-0.5..0.5);
                tmp[s] = ctx.check(s, v);
            }
            let tmp_f = ctx.func.fitness(tmp.as_slice().unwrap(), ctx.adaptive);
            if tmp_f < ctx.fitness[i] {
                ctx.assign_from(i, tmp_f, &tmp);
            }
        }
    }
}

impl<F: ObjFunc> Algorithm<F> for Method {
    #[inline(always)]
    fn generation(&mut self, ctx: &mut Context<F>) {
        self.move_fireflies(ctx);
        ctx.find_best();
    }
}
