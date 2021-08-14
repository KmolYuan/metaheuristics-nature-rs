use crate::{random::*, *};
use core::ops::{Deref, DerefMut};
use ndarray::{s, Array1, AsArray};

setting_builder! {
    /// Firefly Algorithm settings.
    pub struct FASetting {
        @base,
        @pop_num = 80,
        /// Alpha factor.
        alpha: f64 = 0.05,
        /// Minimum beta factor.
        beta_min: f64 = 0.2,
        /// Gamma factor.
        gamma: f64 = 1.,
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

impl<F: ObjFunc> FA<F> {
    fn move_fireflies(&mut self) {
        for (i, j) in product(0..self.pop_num, 0..self.pop_num) {
            if self.fitness[i] <= self.fitness[j] {
                continue;
            }
            let mut tmp = Array1::zeros(self.dim);
            let pool_j = if i == j {
                self.best.view()
            } else {
                self.pool.slice(s![j, ..])
            };
            let r = distance(self.pool.slice(s![i, ..]), pool_j.view());
            let beta = (self.beta0 - self.beta_min) * (-self.gamma * r).exp() + self.beta_min;
            for s in 0..self.dim {
                let v = self.pool[[i, s]]
                    + beta * (pool_j[s] - self.pool[[i, s]])
                    + self.alpha * (self.ub(s) - self.lb(s)) * rand_float(-0.5, 0.5);
                tmp[s] = self.check(s, v);
            }
            let tmp_f = self.func.fitness(&tmp, &self.report);
            if tmp_f < self.fitness[i] {
                self.assign_from(i, tmp_f, &tmp);
            }
        }
    }
}

impl<F: ObjFunc> DerefMut for FA<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.base
    }
}

impl<F: ObjFunc> Deref for FA<F> {
    type Target = AlgorithmBase<F>;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl<F: ObjFunc> Algorithm<F> for FA<F> {
    type Setting = FASetting;

    fn create(func: F, settings: Self::Setting) -> Self {
        let base = AlgorithmBase::new(func, settings.base);
        Self {
            alpha: settings.alpha,
            beta_min: settings.beta_min,
            gamma: settings.gamma,
            beta0: 1.,
            base,
        }
    }

    #[inline(always)]
    fn generation(&mut self) {
        self.move_fireflies();
        self.find_best();
    }
}
