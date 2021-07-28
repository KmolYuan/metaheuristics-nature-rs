use crate::{random::*, *};
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

impl<F> PSO<F>
where
    F: ObjFunc,
{
    fn set_past(&mut self, i: usize) {
        self.best_past
            .slice_mut(s![i, ..])
            .assign(&self.base.pool.slice(s![i, ..]));
        self.best_f_past[i] = self.base.fitness[i].clone();
    }
}

impl<F> Algorithm<F> for PSO<F>
where
    F: ObjFunc,
{
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
    fn base(&self) -> &AlgorithmBase<F> {
        &self.base
    }

    #[inline(always)]
    fn base_mut(&mut self) -> &mut AlgorithmBase<F> {
        &mut self.base
    }

    #[inline(always)]
    fn init(&mut self) {
        self.best_past = self.base.pool.clone();
        self.best_f_past = self.base.fitness.clone();
    }

    fn generation(&mut self) {
        #[cfg(feature = "parallel")]
        let mut tasks = crate::thread_pool::ThreadPool::new();
        for i in 0..self.base.pop_num {
            let alpha = rand_rng(0., self.cognition);
            let beta = rand_rng(0., self.social);
            for s in 0..self.base.dim {
                let v = self.velocity * self.base.pool[[i, s]]
                    + alpha * (self.best_past[[i, s]] - self.base.pool[[i, s]])
                    + beta * (self.base.best[s] - self.base.pool[[i, s]]);
                self.base.pool[[i, s]] = self.check(s, v);
            }
            #[cfg(feature = "parallel")]
            tasks.insert(
                i,
                self.base.func.clone(),
                self.base.report.clone(),
                self.base.pool.slice(s![i, ..]),
            );
            #[cfg(not(feature = "parallel"))]
            {
                self.base.fitness(i);
                if self.base.fitness[i] < self.best_f_past[i] {
                    self.set_past(i);
                }
                if self.base.fitness[i] < self.base.report.best_f {
                    self.base.set_best(i);
                }
            }
        }
        #[cfg(feature = "parallel")]
        for (i, f) in tasks {
            self.base.fitness[i] = f;
            if self.base.fitness[i] < self.best_f_past[i] {
                self.set_past(i);
            }
            if self.base.fitness[i] < self.base.report.best_f {
                self.base.set_best(i);
            }
        }
    }
}
