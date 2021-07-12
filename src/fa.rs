use crate::*;
use ndarray::{s, AsArray};

setting_builder! {
    /// Firefly Algorithm settings.
    pub struct FASetting {
        @base,
        @pop_num = 80,
        /// Alpha factor.
        alpha: f64 = 0.01,
        /// Minimum beta factor.
        beta_min: f64 = 0.2,
        /// Gamma factor.
        gamma: f64 = 1.,
        /// Initial beta factor.
        beta0: f64 = 1.,
    }
}

fn distance<'a, A>(me: A, she: A) -> f64
where
    A: AsArray<'a, f64>,
{
    let me = me.into();
    let she = she.into();
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

impl<F> FA<F>
where
    F: ObjFunc,
{
    fn move_firefly(&mut self, me: usize, she: usize) {
        let r = distance(
            self.base.pool.slice(s![me, ..]),
            self.base.pool.slice(s![she, ..]),
        );
        self.beta0 -= self.beta_min;
        let beta = self.beta0 * (-self.gamma * r).exp() + self.beta_min;
        for s in 0..self.base.dim {
            self.base.pool[[me, s]] = self.check(
                s,
                self.base.pool[[me, s]]
                    + beta * (self.base.pool[[she, s]] - self.base.pool[[me, s]])
                    + self.alpha * (self.ub(s) - self.lb(s)) * rand!(-0.5, 0.5),
            );
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
                    self.base.pool[[i, s]] = self.check(
                        s,
                        self.base.pool[[i, s]]
                            + self.alpha * (self.ub(s) - self.lb(s)) * rand!(-0.5, 0.5),
                    );
                }
            }
            self.base.fitness(i);
        }
    }
}

impl<F> Algorithm<F> for FA<F>
where
    F: ObjFunc,
{
    type Setting = FASetting;

    fn create(func: F, settings: Self::Setting) -> Self {
        let base = AlgorithmBase::new(func, settings.base);
        Self {
            alpha: settings.alpha,
            beta_min: settings.beta_min,
            gamma: settings.gamma,
            beta0: settings.beta0,
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
    fn generation(&mut self) {
        self.move_fireflies();
        self.find_best();
    }
}
