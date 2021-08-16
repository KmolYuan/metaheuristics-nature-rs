use crate::{random::*, thread_pool::ThreadPool, *};
use ndarray::s;

setting_builder! {
    /// Particle Swarm Optimization settings.
    pub struct PsoSetting for Pso {
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

/// Particle Swarm Optimization type.
pub struct Pso {
    cognition: f64,
    social: f64,
    velocity: f64,
    best_past: Array2<f64>,
    best_f_past: Array1<f64>,
}

impl Pso {
    fn set_past<F: ObjFunc>(&mut self, ctx: &mut Context<F>, i: usize) {
        self.best_past
            .slice_mut(s![i, ..])
            .assign(&ctx.pool.slice(s![i, ..]));
        self.best_f_past[i] = ctx.fitness[i];
    }
}

impl Algorithm for Pso {
    type Setting = PsoSetting;

    fn create(settings: &Self::Setting) -> Self {
        Self {
            cognition: settings.cognition,
            social: settings.social,
            velocity: settings.velocity,
            best_past: Array2::zeros((1, 1)),
            best_f_past: Array1::ones(1) * f64::INFINITY,
        }
    }

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
