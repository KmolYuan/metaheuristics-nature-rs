//! # Particle Swarm Optimization
//!
//! <https://en.wikipedia.org/wiki/Particle_swarm_optimization>
use crate::utility::prelude::*;
use alloc::vec::Vec;

const DEF: Pso = Pso { cognition: 2.05, social: 2.05, velocity: 1.3 };

/// Particle Swarm Optimization settings.
#[derive(Clone)]
#[cfg_attr(feature = "clap", derive(clap::Args))]
pub struct Pso {
    /// Cognition factor
    #[cfg_attr(feature = "clap", clap(long, default_value_t = DEF.cognition))]
    pub cognition: f64,
    /// Social factor
    #[cfg_attr(feature = "clap", clap(long, default_value_t = DEF.social))]
    pub social: f64,
    /// Velocity factor
    #[cfg_attr(feature = "clap", clap(long, default_value_t = DEF.velocity))]
    pub velocity: f64,
}

impl Pso {
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

impl Default for Pso {
    fn default() -> Self {
        DEF
    }
}

impl Setting for Pso {
    type Algorithm<F: ObjFunc> = Method<F::Fitness>;

    fn algorithm<F: ObjFunc>(self) -> Self::Algorithm<F> {
        Method {
            pso: self,
            best_past: Array2::zeros((1, 1)),
            best_past_f: Vec::new(),
        }
    }
}

/// Particle Swarm Optimization type.
pub struct Method<F: Fitness> {
    pso: Pso,
    best_past: Array2<f64>,
    best_past_f: Vec<F>,
}

impl<F: Fitness> core::ops::Deref for Method<F> {
    type Target = Pso;

    fn deref(&self) -> &Self::Target {
        &self.pso
    }
}

impl<F: ObjFunc> Algorithm<F> for Method<F::Fitness> {
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
                let alpha = ctx.rng.ub(self.cognition);
                let beta = ctx.rng.ub(self.social);
                for s in 0..ctx.dim() {
                    let var = self.velocity * v[s]
                        + alpha * (past[s] - v[s])
                        + beta * (ctx.best[s] - v[s]);
                    v[s] = ctx.clamp(s, var);
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
