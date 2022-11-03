//! Particle Swarm Optimization.
//!
//! <https://en.wikipedia.org/wiki/Particle_swarm_optimization>
use crate::utility::prelude::*;
use alloc::vec::Vec;
use core::marker::PhantomData;

/// Particle Swarm Optimization settings.
pub struct Pso<F: Fitness> {
    cognition: f64,
    social: f64,
    velocity: f64,
    _marker: PhantomData<F>,
}

impl<F: Fitness> Pso<F> {
    impl_builders! {
        default,
        /// Cognition factor.
        fn cognition(f64)
        /// Social factor.
        fn social(f64)
        /// Moving velocity.
        fn velocity(f64)
    }
}

impl<F: Fitness> Default for Pso<F> {
    fn default() -> Self {
        Self {
            cognition: 2.05,
            social: 2.05,
            velocity: 1.3,
            _marker: PhantomData,
        }
    }
}

impl<F: Fitness> Setting for Pso<F> {
    type Algorithm = Method<F>;

    fn algorithm(self) -> Self::Algorithm {
        let Self { cognition, social, velocity, _marker } = self;
        Method {
            cognition,
            social,
            velocity,
            best_past: Array2::zeros((1, 1)),
            best_past_f: Vec::new(),
        }
    }
}

/// Particle Swarm Optimization type.
pub struct Method<F: Fitness> {
    cognition: f64,
    social: f64,
    velocity: f64,
    best_past: Array2<f64>,
    best_past_f: Vec<F>,
}

impl<F: ObjFunc> Algorithm<F> for Method<F::Fitness> {
    #[inline(always)]
    fn init(&mut self, ctx: &mut Ctx<F>) {
        self.best_past = ctx.pool.clone();
        self.best_past_f = ctx.pool_f.clone();
    }

    fn generation(&mut self, ctx: &mut Ctx<F>) {
        let mut fitness = ctx.pool_f.clone();
        let mut pool = ctx.pool.clone();
        let mut best_past = self.best_past.clone();
        let mut best_past_f = self.best_past_f.clone();
        #[cfg(feature = "rayon")]
        let iter = fitness.par_iter_mut();
        #[cfg(not(feature = "rayon"))]
        let iter = fitness.iter_mut();
        let (f, v) = iter
            .zip(pool.axis_iter_mut(Axis(0)))
            .zip(best_past.axis_iter_mut(Axis(0)))
            .zip(&mut best_past_f)
            .map(|(((f, mut v), mut past), past_f)| {
                let alpha = ctx.rng.float(0.0..self.cognition);
                let beta = ctx.rng.float(0.0..self.social);
                for s in 0..ctx.dim() {
                    let variable = self.velocity * v[s]
                        + alpha * (past[s] - v[s])
                        + beta * (ctx.best[s] - v[s]);
                    v[s] = ctx.clamp(s, variable);
                }
                *f = ctx.func.fitness(v.as_slice().unwrap());
                if *f < *past_f {
                    *past_f = f.clone();
                    past.assign(&v);
                }
                (f, v)
            })
            .min_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap())
            .unwrap();
        ctx.set_best_from(f.clone(), &v);
        ctx.pool_f = fitness;
        ctx.pool = pool;
        self.best_past = best_past;
        self.best_past_f = best_past_f;
    }
}
