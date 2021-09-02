//! Particle Swarm Optimization.
//!
//! <https://en.wikipedia.org/wiki/Particle_swarm_optimization>
use crate::{thread_pool::ThreadPool, utility::*, *};

setting! {
    /// Particle Swarm Optimization settings.
    pub struct Pso {
        @base,
        @pop_num = 200,
        /// Cognition factor.
        cognition: f64 = 2.05,
        /// Social factor.
        social: f64 = 2.05,
        /// Moving velocity.
        velocity: f64 = 1.3,
    }
}

impl Setting for Pso {
    type Algorithm = Method;

    fn base(&self) -> &BasicSetting {
        &self.base
    }

    fn create(self) -> Self::Algorithm {
        Method {
            cognition: self.cognition,
            social: self.social,
            velocity: self.velocity,
            best_past: Array2::zeros((1, 1)),
            best_f_past: Array1::ones(1) * f64::INFINITY,
        }
    }
}

/// Particle Swarm Optimization type.
pub struct Method {
    cognition: f64,
    social: f64,
    velocity: f64,
    best_past: Array2<f64>,
    best_f_past: Array1<f64>,
}

impl Method {
    fn set_past<F: ObjFunc>(&mut self, ctx: &mut Context<F>, i: usize) {
        self.best_past
            .slice_mut(s![i, ..])
            .assign(&ctx.pool.slice(s![i, ..]));
        self.best_f_past[i] = ctx.fitness[i];
    }
}

impl Algorithm for Method {
    #[inline(always)]
    fn init<F: ObjFunc>(&mut self, ctx: &mut Context<F>) {
        self.best_past = ctx.pool.clone();
        self.best_f_past = ctx.fitness.clone();
    }

    fn generation<F: ObjFunc>(&mut self, ctx: &mut Context<F>) {
        let mut tasks = ThreadPool::new();
        for i in 0..ctx.pop_num {
            let alpha = rand_float(0., self.cognition);
            let beta = rand_float(0., self.social);
            for s in 0..ctx.dim {
                let v = self.velocity * ctx.pool[[i, s]]
                    + alpha * (self.best_past[[i, s]] - ctx.pool[[i, s]])
                    + beta * (ctx.best[s] - ctx.pool[[i, s]]);
                ctx.pool[[i, s]] = ctx.check(s, v);
            }
            tasks.insert(
                i,
                ctx.func.clone(),
                ctx.report.clone(),
                ctx.pool.slice(s![i, ..]),
            );
        }
        for (i, f) in tasks {
            ctx.fitness[i] = f;
            if ctx.fitness[i] < self.best_f_past[i] {
                self.set_past(ctx, i);
            }
            if ctx.fitness[i] < ctx.report.best_f {
                ctx.set_best(i);
            }
        }
    }
}
