//! # Particle Swarm Optimization
//!
//! <https://en.wikipedia.org/wiki/Particle_swarm_optimization>
use crate::prelude::*;
use alloc::vec::Vec;

const DEF: Pso = Pso { cognition: 2.05, social: 2.05, velocity: 1.3 };

/// Particle Swarm Optimization settings.
#[derive(Clone, PartialEq)]
#[cfg_attr(feature = "clap", derive(clap::Args))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
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
    /// Constant default value.
    pub const fn new() -> Self {
        DEF
    }

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
        DEF
    }
}

impl Setting for Pso {
    type Algorithm<F: ObjFunc> = Method<F::Fitness>;

    fn algorithm<F: ObjFunc>(self) -> Self::Algorithm<F> {
        Method {
            pso: self,
            best_past: Vec::new(),
            best_past_f: Vec::new(),
        }
    }
}

/// Particle Swarm Optimization type.
pub struct Method<F: Fitness> {
    pso: Pso,
    best_past: Vec<Vec<f64>>,
    best_past_f: Vec<F>,
}

impl<F: Fitness> core::ops::Deref for Method<F> {
    type Target = Pso;

    fn deref(&self) -> &Self::Target {
        &self.pso
    }
}

impl<F: ObjFunc> Algorithm<F> for Method<F::Fitness> {
    fn init(&mut self, ctx: &mut Ctx<F>, _: &Rng) {
        self.best_past = ctx.pool.clone();
        self.best_past_f = ctx.pool_f.clone();
    }

    fn generation(&mut self, ctx: &mut Ctx<F>, rng: &Rng) {
        let rng = rng.stream(ctx.pop_num());
        let cognition = self.cognition;
        let social = self.social;
        let velocity = self.velocity;
        #[cfg(not(feature = "rayon"))]
        let iter = rng.into_iter();
        #[cfg(feature = "rayon")]
        let iter = rng.into_par_iter();
        let iter = iter
            .zip(&mut ctx.pool)
            .zip(&mut ctx.pool_f)
            .zip(&mut self.best_past)
            .zip(&mut self.best_past_f)
            .filter_map(|((((rng, xs), f), past), past_f)| {
                let alpha = rng.ub(cognition);
                let beta = rng.ub(social);
                let best = ctx.best.sample_xs(&rng);
                for s in 0..ctx.func.dim() {
                    let var =
                        velocity * xs[s] + alpha * (past[s] - xs[s]) + beta * (best[s] - xs[s]);
                    xs[s] = ctx.func.clamp(s, var);
                }
                *f = ctx.func.fitness(xs);
                if f.is_dominated(&*past_f) {
                    *past_f = f.clone();
                    past_f.mark_not_best();
                    *past = xs.clone();
                    Some((xs, f))
                } else {
                    None
                }
            });
        #[cfg(not(feature = "rayon"))]
        let (xs, f) = iter
            .reduce(|a, b| if a.1.is_dominated(&*b.1) { a } else { b })
            .unwrap();
        #[cfg(feature = "rayon")]
        let (xs, f) = iter
            .reduce_with(|a, b| if a.1.is_dominated(&*b.1) { a } else { b })
            .unwrap();
        ctx.best.update(xs, f);
        ctx.prune_fitness();
    }
}
