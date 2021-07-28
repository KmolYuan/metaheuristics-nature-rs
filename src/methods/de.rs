use self::Strategy::*;
use crate::{random::*, *};
use ndarray::{s, Array1};

/// The Differential Evolution strategy.
/// Each strategy has different formulas on the recombination.
///
/// # Variable formula
///
/// This formula decide how to generate new variable *n*.
/// Where *vi* is the random indicator on the individuals,
/// they are different from each other.
///
/// + *f1*: best{n} + F * (v0{n} - v1{n})
/// + *f2*: v0{n} + F * (v1{n} - v2{n})
/// + *f3*: self{n} + F * (best{n} - self{n} + v0{n} - v1{n})
/// + *f4*: best{n} + F * (v0{n} + v1{n} - v2{n} - v3{n})
/// + *f5*: v4{n} + F * (v0{n} + v1{n} - v2{n} - v3{n})
///
/// # Crossing formula
///
/// + *c1*: Continue crossing with the variables order until failure.
/// + *c2*: Each variable has independent probability.
#[derive(Clone)]
pub enum Strategy {
    /// *f1* + *c1*
    S1,
    /// *f2* + *c1*
    S2,
    /// *f3* + *c1*
    S3,
    /// *f4* + *c1*
    S4,
    /// *f5* + *c1*
    S5,
    /// *f1* + *c2*
    S6,
    /// *f2* + *c2*
    S7,
    /// *f3* + *c2*
    S8,
    /// *f4* + *c2*
    S9,
    /// *f5* + *c2*
    S10,
}

setting_builder! {
    /// Differential Evolution settings.
    pub struct DESetting {
        @base,
        @pop_num = 400,
        /// Strategy of the formula.
        strategy: Strategy = S1,
        /// F factor.
        f: f64 = 0.6,
        /// Crossing probability.
        cross: f64 = 0.9,
    }
}

/// Differential Evolution type.
pub struct DE<F: ObjFunc> {
    f: f64,
    cross: f64,
    num: usize,
    formula: fn(&Self, &Array1<f64>, &Array1<usize>, usize) -> f64,
    setter: fn(&mut Self, &mut Array1<f64>, Array1<usize>, usize),
    base: AlgorithmBase<F>,
}

impl<F> DE<F>
where
    F: ObjFunc,
{
    fn f1(&self, _tmp: &Array1<f64>, v: &Array1<usize>, n: usize) -> f64 {
        self.base.best[n] + self.f * (self.base.pool[[v[0], n]] - self.base.pool[[v[1], n]])
    }

    fn f2(&self, _tmp: &Array1<f64>, v: &Array1<usize>, n: usize) -> f64 {
        self.base.pool[[v[0], n]] + self.f * (self.base.pool[[v[1], n]] - self.base.pool[[v[2], n]])
    }

    fn f3(&self, tmp: &Array1<f64>, v: &Array1<usize>, n: usize) -> f64 {
        tmp[n]
            + self.f
                * (self.base.best[n] - tmp[n] + self.base.pool[[v[0], n]]
                    - self.base.pool[[v[1], n]])
    }

    fn f4(&self, _tmp: &Array1<f64>, v: &Array1<usize>, n: usize) -> f64 {
        self.base.best[n] + self.f45(v, n)
    }

    fn f5(&self, _tmp: &Array1<f64>, v: &Array1<usize>, n: usize) -> f64 {
        self.base.pool[[v[4], n]] + self.f45(v, n)
    }

    fn f45(&self, v: &Array1<usize>, n: usize) -> f64 {
        (self.base.pool[[v[0], n]] + self.base.pool[[v[1], n]]
            - self.base.pool[[v[2], n]]
            - self.base.pool[[v[3], n]])
            * self.f
    }

    fn c1(&mut self, tmp: &mut Array1<f64>, v: Array1<usize>, mut n: usize) {
        for _ in 0..self.base.dim {
            tmp[n] = (self.formula)(self, tmp, &v, n);
            n = (n + 1) % self.base.dim;
            if !maybe(self.cross) {
                break;
            }
        }
    }

    fn c2(&mut self, tmp: &mut Array1<f64>, v: Array1<usize>, mut n: usize) {
        for lv in 0..self.base.dim {
            if !maybe(self.cross) || lv == self.base.dim - 1 {
                tmp[n] = (self.formula)(self, tmp, &v, n);
            }
            n = (n + 1) % self.base.dim;
        }
    }
}

impl<F> Algorithm<F> for DE<F>
where
    F: ObjFunc,
{
    type Setting = DESetting;

    fn create(func: F, settings: Self::Setting) -> Self {
        let base = AlgorithmBase::new(func, settings.base);
        let num = match settings.strategy {
            S1 | S3 | S6 | S8 => 2,
            S2 | S7 => 3,
            S4 | S9 => 4,
            S5 | S10 => 5,
        };
        Self {
            f: settings.f,
            cross: settings.cross,
            num,
            formula: match settings.strategy {
                S1 | S6 => Self::f1,
                S2 | S7 => Self::f2,
                S3 | S8 => Self::f3,
                S4 | S9 => Self::f4,
                S5 | S10 => Self::f5,
            },
            setter: match settings.strategy {
                S1 | S2 | S3 | S4 | S5 => Self::c1,
                S6 | S7 | S8 | S9 | S10 => Self::c2,
            },
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

    fn generation(&mut self) {
        'a: for i in 0..self.base.pop_num {
            // Generate Vector
            let mut v = Array1::zeros(self.num);
            for j in 0..self.num {
                v[j] = i;
                while v[j] == i || v.slice(s![..j]).iter().any(|&n| n == v[j]) {
                    v[j] = rand_rng(0, self.base.pop_num);
                }
            }
            // Recombination
            let mut tmp = self.base.pool.slice(s![i, ..]).to_owned();
            (self.setter)(self, &mut tmp, v, rand_rng(0, self.base.dim));
            for s in 0..self.base.dim {
                if tmp[s] > self.ub(s) || tmp[s] < self.lb(s) {
                    continue 'a;
                }
            }
            let tmp_f = self.base.func.fitness(&tmp, &self.base.report);
            if tmp_f < self.base.fitness[i] {
                self.assign_from(i, tmp_f, &tmp.clone());
            }
        }
        self.find_best();
    }
}
