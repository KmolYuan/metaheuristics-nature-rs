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
        let mut fitness = ctx.pool_f.clone();
        let mut pool = ctx.pool.clone();
        let mut best_past = self.best_past.clone();
        let mut best_past_f = self.best_past_f.clone();
        #[cfg(feature = "rayon")]
        let iter = fitness.par_iter_mut();
        #[cfg(not(feature = "rayon"))]
        let iter = fitness.iter_mut();
        if let Some((f, xs)) = iter
            .zip(&mut pool)
            .zip(&mut best_past)
            .zip(&mut best_past_f)
            .zip(rng.stream(ctx.pop_num()))
            .filter_map(|((((f, xs), past), past_f), rng)| {
                let alpha = rng.ub(self.cognition);
                let beta = rng.ub(self.social);
                for s in 0..ctx.dim() {
                    let var = self.velocity * xs[s]
                        + alpha * (past[s] - xs[s])
                        + beta * (ctx.best[s] - xs[s]);
                    xs[s] = ctx.clamp(s, var);
                }
                *f = ctx.func.fitness(xs);
                if *f < *past_f {
                    *past_f = f.clone();
                    past_f.mark_not_best();
                    *past = xs.clone();
                }
                (*f < ctx.best_f).then_some((f, xs))
            })
            .min_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap())
        {
            ctx.best = xs.clone();
            ctx.best_f = f.clone();
        }
        ctx.pool = pool;
        ctx.pool_f = fitness;
        ctx.prune_fitness();
        self.best_past = best_past;
        self.best_past_f = best_past_f;
    }
}
