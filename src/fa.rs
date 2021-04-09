use crate::{Algorithm, AlgorithmBase, ObjFunc, Setting};

/// Firefly Algorithm settings.
pub struct FASetting {
    pub base: Setting,
    pub alpha: f64,
    pub beta_min: f64,
    pub gamma: f64,
    pub beta0: f64,
}

impl Default for FASetting {
    fn default() -> Self {
        Self {
            base: Setting {
                pop_num: 80,
                ..Default::default()
            },
            alpha: 0.01,
            beta_min: 0.2,
            gamma: 1.,
            beta0: 1.,
        }
    }
}

fn distance(me: &Vec<f64>, she: &Vec<f64>) -> f64 {
    let mut dist = 0.;
    for s in 0..me.len() {
        let diff = me[s] - she[s];
        dist += diff * diff;
    }
    dist
}

/// Firefly Algorithm type.
pub struct FA<F: ObjFunc> {
    alpha: f64,
    beta_min: f64,
    gamma: f64,
    beta0: f64,
    base: AlgorithmBase<F>,
}

impl<F: ObjFunc> FA<F> {
    pub fn new(func: F, settings: FASetting) -> Self {
        let base = AlgorithmBase::new(func, settings.base);
        Self {
            alpha: settings.alpha,
            beta_min: settings.beta_min,
            gamma: settings.gamma,
            beta0: settings.beta0,
            base,
        }
    }
    fn move_firefly(&mut self, me: usize, she: usize) {
        let r = distance(&self.base.pool[me], &self.base.pool[she]);
        self.beta0 -= self.beta_min;
        let beta = self.beta0 * (-self.gamma * r).exp() + self.beta_min;
        for s in 0..self.base.dim {
            self.base.pool[me][s] = self.check(s, self.base.pool[me][s]
                + beta * (self.base.pool[she][s] - self.base.pool[me][s])
                + self.alpha * (self.ub(s) - self.lb(s)) * rand!(-0.5, 0.5));
        }
    }
    fn move_fireflies(&mut self) {
        for i in 0..self.base.pop_num {
            let mut moved = false;
            for j in 0..self.base.pop_num {
                if i == j || self.base.fitness[i] <= self.base.fitness[j] {
                    continue;
                }
                self.move_firefly(i, j);
                moved = true;
            }
            if !moved {
                for s in 0..self.base.dim {
                    self.base.pool[i][s] = self.check(s, self.base.pool[i][s]
                        + self.alpha * (self.ub(s) - self.lb(s)) * rand!(-0.5, 0.5));
                }
            }
            self.base.fitness(i);
        }
    }
}

impl<F: ObjFunc> Algorithm<F> for FA<F> {
    fn base(&self) -> &AlgorithmBase<F> { &self.base }
    fn base_mut(&mut self) -> &mut AlgorithmBase<F> { &mut self.base }
    fn generation(&mut self) {
        self.move_fireflies();
        self.find_best();
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        {FA, FASetting, Setting, Task},
        tests::{test, TestObj},
    };

    #[test]
    fn fa() {
        test(FA::new(
            TestObj::new(),
            FASetting {
                base: Setting {
                    task: Task::MinFit(1e-20),
                    ..Default::default()
                },
                ..Default::default()
            },
        ));
    }
}
