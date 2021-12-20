//! Particle Swarm Optimization.
//!
//! <https://en.wikipedia.org/wiki/Particle_swarm_optimization>
use crate::utility::prelude::*;

/// Particle Swarm Optimization settings.
pub struct Pso {
    cognition: f64,
    social: f64,
    velocity: f64,
}

impl Pso {
    impl_builders! {
        /// Cognition factor.
        fn cognition(f64)
        /// Social factor.
        fn social(f64)
        /// Moving velocity.
        fn velocity(f64)
    }
}

impl Default for Pso {
    fn default() -> Self {
        Self {
            cognition: 2.05,
            social: 2.05,
            velocity: 1.3,
        }
    }
}

impl Setting for Pso {
    type Algorithm = Method;

    fn algorithm(self) -> Self::Algorithm {
        Method {
            cognition: self.cognition,
            social: self.social,
            velocity: self.velocity,
            best_past: Array2::zeros((1, 1)),
            best_past_f: Array1::ones(1) * f64::INFINITY,
        }
    }

    fn default_basic() -> BasicSetting {
        BasicSetting {
            pop_num: 200,
            ..Default::default()
        }
    }
}

/// Particle Swarm Optimization type.
pub struct Method {
    cognition: f64,
    social: f64,
    velocity: f64,
    best_past: Array2<f64>,
    best_past_f: Array1<f64>,
}

impl<F: ObjFunc> Algorithm<F> for Method {
    #[inline(always)]
    fn init(&mut self, ctx: &mut Context<F>) {
        self.best_past = ctx.pool.clone();
        self.best_past_f = Array1::from_iter(ctx.fitness.iter().map(|r| r.value()));
    }

    fn generation(&mut self, ctx: &mut Context<F>) {
        let mut fitness = ctx.fitness.clone();
        let mut pool = ctx.pool.clone();
        let mut best_past = self.best_past.clone();
        let mut best_past_f = self.best_past_f.clone();
        let zip = Zip::from(&mut fitness)
            .and(pool.axis_iter_mut(Axis(0)))
            .and(best_past.axis_iter_mut(Axis(0)))
            .and(&mut best_past_f);
        #[cfg(not(feature = "parallel"))]
        {
            zip.for_each(|f, mut v, mut past, f_past| {
                let alpha = ctx.rng.float(0.0..self.cognition);
                let beta = ctx.rng.float(0.0..self.social);
                for s in 0..ctx.dim() {
                    let variable = self.velocity * v[s]
                        + alpha * (past[s] - v[s])
                        + beta * (ctx.best[s] - v[s]);
                    v[s] = ctx.check(s, variable);
                }
                *f = ctx.func.fitness(v.as_slice().unwrap(), ctx.adaptive);
                if f.value() < *f_past {
                    *f_past = f.value();
                    past.assign(&v);
                }
            });
            ctx.find_best();
        }
        #[cfg(feature = "parallel")]
        {
            let (f, v) = zip
                .into_par_iter()
                .map(|(f, mut v, mut past, past_f)| {
                    let alpha = ctx.rng.float(0.0..self.cognition);
                    let beta = ctx.rng.float(0.0..self.social);
                    for s in 0..ctx.dim() {
                        let variable = self.velocity * v[s]
                            + alpha * (past[s] - v[s])
                            + beta * (ctx.best[s] - v[s]);
                        v[s] = ctx.check(s, variable);
                    }
                    *f = ctx.func.fitness(v.as_slice().unwrap(), ctx.adaptive);
                    if f.value() < *past_f {
                        *past_f = f.value();
                        past.assign(&v);
                    }
                    (f, v)
                })
                .min_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap())
                .unwrap();
            ctx.set_best_from(f.clone(), &v);
        }
        ctx.fitness = fitness;
        ctx.pool = pool;
        self.best_past = best_past;
        self.best_past_f = best_past_f;
    }
}
