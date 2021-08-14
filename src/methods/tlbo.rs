use crate::{random::*, *};
use core::ops::{Deref, DerefMut};
use ndarray::{s, Array1};

/// Teaching Learning Based Optimization settings.
/// This is a type alias to [`Setting`].
pub type TLBOSetting = Setting;

/// Teaching Learning Based Optimization type.
pub struct TLBO<F: ObjFunc> {
    tmp: Array1<f64>,
    base: AlgorithmBase<F>,
}

impl<F: ObjFunc> TLBO<F> {
    fn register(&mut self, i: usize) {
        let f_new = self.func.fitness(&self.tmp, &self.report);
        if f_new < self.fitness[i] {
            self.base.pool.slice_mut(s![i, ..]).assign(&self.tmp);
            self.fitness[i] = f_new;
        }
        if f_new < self.report.best_f {
            self.set_best(i);
        }
    }

    fn teaching(&mut self, i: usize) {
        let tf = f64::round(rand() + 1.);
        for s in 0..self.dim {
            let mut mean = 0.;
            for j in 0..self.pop_num {
                mean += self.pool[[j, s]];
            }
            mean /= self.dim as f64;
            let v =
                self.pool[[i, s]] + rand_float(1., self.dim as f64) * (self.best[s] - tf * mean);
            self.tmp[s] = self.check(s, v);
        }
        self.register(i);
    }

    fn learning(&mut self, i: usize) {
        let j = {
            let j = rand_int(0, self.pop_num - 1);
            if j >= i {
                j + 1
            } else {
                j
            }
        };
        for s in 0..self.dim {
            let diff = if self.fitness[j] < self.fitness[i] {
                self.pool[[i, s]] - self.pool[[j, s]]
            } else {
                self.pool[[j, s]] - self.pool[[i, s]]
            };
            let v = self.pool[[i, s]] + rand_float(1., self.dim as f64) * diff;
            self.tmp[s] = self.check(s, v);
        }
        self.register(i);
    }
}

impl<F: ObjFunc> DerefMut for TLBO<F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.base
    }
}

impl<F: ObjFunc> Deref for TLBO<F> {
    type Target = AlgorithmBase<F>;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl<F: ObjFunc> Algorithm<F> for TLBO<F> {
    type Setting = TLBOSetting;

    fn create(func: F, settings: Self::Setting) -> Self {
        let base = AlgorithmBase::new(func, settings);
        Self {
            tmp: Array1::zeros(base.dim),
            base,
        }
    }

    #[inline(always)]
    fn generation(&mut self) {
        for i in 0..self.pop_num {
            self.teaching(i);
            self.learning(i);
        }
    }
}
