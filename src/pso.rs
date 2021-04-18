use crate::{Algorithm, AlgorithmBase, ObjFunc, Setting};

/// Particle Swarm Optimization settings.
pub struct PSOSetting {
    pub base: Setting,
    pub cognition: f64,
    pub social: f64,
    pub velocity: f64,
}

impl Default for PSOSetting {
    fn default() -> Self {
        Self {
            base: Setting {
                pop_num: 200,
                ..Default::default()
            },
            cognition: 2.05,
            social: 2.05,
            velocity: 1.3,
        }
    }
}

/// Particle Swarm Optimization type.
pub struct PSO<F: ObjFunc> {
    cognition: f64,
    social: f64,
    velocity: f64,
    best_past: Vec<Vec<f64>>,
    best_f_past: Vec<f64>,
    base: AlgorithmBase<F>,
}

impl<F: ObjFunc> PSO<F> {
    pub fn new(func: F, settings: PSOSetting) -> Self {
        let base = AlgorithmBase::new(func, settings.base);
        Self {
            cognition: settings.cognition,
            social: settings.social,
            velocity: settings.velocity,
            best_past: vec![],
            best_f_past: vec![],
            base,
        }
    }
}

impl<F: ObjFunc> Algorithm<F> for PSO<F> {
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
                self.base.pool[i][s] = self.check(
                    s,
                    self.velocity * self.base.pool[i][s]
                        + alpha * (self.best_past[i][s] - self.base.pool[i][s])
                        + beta * (self.base.best[s] - self.base.pool[i][s]),
                );
            }
            self.base.fitness(i);
            if self.base.fitness[i] < self.best_f_past[i] {
                self.best_past[i] = self.base.pool[i].clone();
                self.best_f_past[i] = self.base.fitness[i].clone();
            }
            if self.base.fitness[i] < self.base.best_f {
                self.set_best(i);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        tests::{test, TestObj},
        {PSOSetting, Setting, Task, PSO},
    };

    #[test]
    fn pso() {
        test(PSO::new(
            TestObj::new(),
            PSOSetting {
                base: Setting {
                    task: Task::MinFit(1e-20),
                    ..Default::default()
                },
                ..Default::default()
            },
        ));
    }
}
