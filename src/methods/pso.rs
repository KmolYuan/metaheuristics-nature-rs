use crate::{random::*, thread_pool::ThreadPool, *};
use core::ops::{Deref, DerefMut};
use ndarray::{s, Array1, Array2};

setting_builder! {
    /// Particle Swarm Optimization settings.
    pub struct PSOSetting {
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
pub struct PSO<F: ObjFunc> {
    cognition: f64,
    social: f64,
    velocity: f64,
    best_past: Array2<f64>,
    best_f_past: Array1<f64>,
    base: AlgorithmBase<F>,
}

impl<F: ObjFunc> PSO<F> {
    fn set_past(&mut self, i: usize) {
        self.best_past
            .slice_mut(s![i, ..])
            .assign(&self.base.pool.slice(s![i, ..]));
        self.best_f_past[i] = self.fitness[i];
    }
}

impl<F: ObjFunc> DerefMut for PSO<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.base
    }
}

impl<F: ObjFunc> Deref for PSO<F> {
    type Target = AlgorithmBase<F>;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl<F: ObjFunc> Algorithm<F> for PSO<F> {
    type Setting = PSOSetting;

    fn create(func: F, settings: Self::Setting) -> Self {
        let base = AlgorithmBase::new(func, settings.base);
        Self {
            cognition: settings.cognition,
            social: settings.social,
            velocity: settings.velocity,
            best_past: Array2::zeros((base.pop_num, base.dim)),
            best_f_past: Array1::ones(base.pop_num) * f64::INFINITY,
            base,
        }
    }

    #[inline(always)]
    fn init(&mut self) {
        self.best_past = self.pool.to_owned();
        self.best_f_past = self.fitness.to_owned();
    }

    fn generation(&mut self) {
        let mut tasks = ThreadPool::new();
        for i in 0..self.pop_num {
            let alpha = rand_float(0., self.cognition);
            let beta = rand_float(0., self.social);
            for s in 0..self.dim {
                let v = self.velocity * self.pool[[i, s]]
                    + alpha * (self.best_past[[i, s]] - self.pool[[i, s]])
                    + beta * (self.best[s] - self.pool[[i, s]]);
                self.pool[[i, s]] = self.check(s, v);
            }
            tasks.insert(
                i,
                self.func.clone(),
                self.report.clone(),
                self.pool.slice(s![i, ..]),
            );
        }
        for (i, f) in tasks {
            self.fitness[i] = f;
            if self.fitness[i] < self.best_f_past[i] {
                self.set_past(i);
            }
            if self.fitness[i] < self.report.best_f {
                self.set_best(i);
            }
        }
    }
}
