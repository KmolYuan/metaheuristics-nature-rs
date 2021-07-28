use crate::{random::*, *};
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

impl<F> FA<F>
where
    F: ObjFunc,
{
    fn move_fireflies(&mut self) {
        for (i, j) in product(0..self.base.pop_num, 0..self.base.pop_num) {
            if self.base.fitness[i] <= self.base.fitness[j] {
                continue;
            }
            let mut tmp = Array1::zeros(self.base.dim);
            let pool_j = if i == j {
                self.base.best.view()
            } else {
                self.base.pool.slice(s![j, ..])
            };
            let r = distance(self.base.pool.slice(s![i, ..]), pool_j);
            let beta = (self.beta0 - self.beta_min) * (-self.gamma * r).exp() + self.beta_min;
            for s in 0..self.base.dim {
                let v = self.base.pool[[i, s]]
                    + beta * (pool_j[s] - self.base.pool[[i, s]])
                    + self.alpha * (self.ub(s) - self.lb(s)) * rand_rng(-0.5, 0.5);
                tmp[s] = self.check(s, v);
            }
            let tmp_f = self.base.func.fitness(&tmp, &self.base.report);
            if tmp_f < self.base.fitness[i] {
                self.assign_from(i, tmp_f, &tmp);
            }
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
            beta0: 1.,
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
