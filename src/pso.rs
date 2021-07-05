use crate::*;
use ndarray::{s, Array1, Array2};

setting_builder! {
    /// Particle Swarm Optimization settings.
    pub struct PSOSetting {
        @base,
        @pop_num = 200,
        cognition: f64 = 2.05,
        social: f64 = 2.05,
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

impl<F> Algorithm<F> for PSO<F>
where
    F: ObjFunc,
{
    type Setting = PSOSetting;
    fn new(func: F, settings: Self::Setting) -> Self {
        let base = AlgorithmBase::new(func, settings.base);
        Self {
            cognition: settings.cognition,
            social: settings.social,
            velocity: settings.velocity,
            best_past: Array2::zeros((base.pop_num, base.dim)),
            best_f_past: Array1::zeros(base.pop_num),
            base,
        }
    }
    fn base(&self) -> &AlgorithmBase<F> {
        &self.base
    }
    fn base_mut(&mut self) -> &mut AlgorithmBase<F> {
        &mut self.base
    }
    fn init(&mut self) {
        self.best_past = self.base.pool.clone();
        self.best_f_past = self.base.fitness.clone();
    }
    fn generation(&mut self) {
        for i in 0..self.base.pop_num {
            let alpha = rand!(0., self.cognition);
            let beta = rand!(0., self.social);
            for s in 0..self.base.dim {
                self.base.pool[[i, s]] = self.check(
                    s,
                    self.velocity * self.base.pool[[i, s]]
                        + alpha * (self.best_past[[i, s]] - self.base.pool[[i, s]])
                        + beta * (self.base.best[s] - self.base.pool[[i, s]]),
                );
            }
            self.base.fitness(i);
            if self.base.fitness[i] < self.best_f_past[i] {
                self.best_past
                    .slice_mut(s![i, ..])
                    .assign(&self.base.pool.slice(s![i, ..]));
                self.best_f_past[i] = self.base.fitness[i].clone();
            }
            if self.base.fitness[i] < self.base.best_f {
                self.set_best(i);
            }
        }
    }
}
