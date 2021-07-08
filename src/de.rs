use self::Strategy::*;
use crate::*;
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
    v: Array1<usize>,
    tmp: Array1<f64>,
    formula: fn(&Self, usize) -> f64,
    setter: fn(&mut Self, usize),
    base: AlgorithmBase<F>,
}

impl<F> DE<F>
where
    F: ObjFunc,
{
    fn vector(&mut self, i: usize) {
        for j in 0..self.v.len() {
            self.v[j] = i;
            while self.v[j] == i || self.v.slice(s![..j]).iter().any(|&v| v == self.v[j]) {
                self.v[j] = rand!(0, self.base.pop_num);
            }
        }
    }

    fn f1(&self, n: usize) -> f64 {
        self.base.best[n]
            + self.f * (self.base.pool[[self.v[0], n]] - self.base.pool[[self.v[1], n]])
    }

    fn f2(&self, n: usize) -> f64 {
        self.base.pool[[self.v[0], n]]
            + self.f * (self.base.pool[[self.v[1], n]] - self.base.pool[[self.v[2], n]])
    }

    fn f3(&self, n: usize) -> f64 {
        self.tmp[n]
            + self.f
                * (self.base.best[n] - self.tmp[n] + self.base.pool[[self.v[0], n]]
                    - self.base.pool[[self.v[1], n]])
    }

    fn f4(&self, n: usize) -> f64 {
        self.base.best[n] + self.f45(n)
    }

    fn f5(&self, n: usize) -> f64 {
        self.base.pool[[self.v[4], n]] + self.f45(n)
    }

    fn f45(&self, n: usize) -> f64 {
        (self.base.pool[[self.v[0], n]] + self.base.pool[[self.v[1], n]]
            - self.base.pool[[self.v[2], n]]
            - self.base.pool[[self.v[3], n]])
            * self.f
    }

    fn c1(&mut self, mut n: usize) {
        for _ in 0..self.base.dim {
            self.tmp[n] = (self.formula)(self, n);
            n = (n + 1) % self.base.dim;
            if !maybe!(self.cross) {
                break;
            }
        }
    }

    fn c2(&mut self, mut n: usize) {
        for lv in 0..self.base.dim {
            if !maybe!(self.cross) || lv == self.base.dim - 1 {
                self.tmp[n] = (self.formula)(self, n);
            }
            n = (n + 1) % self.base.dim;
        }
    }

    fn recombination(&mut self, i: usize) {
        self.tmp.assign(&self.base.pool.slice(s![i, ..]));
        (self.setter)(self, rand!(0, self.base.dim));
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
            v: Array1::zeros(num),
            tmp: Array1::zeros(base.dim),
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

    fn base(&self) -> &AlgorithmBase<F> {
        &self.base
    }
    fn base_mut(&mut self) -> &mut AlgorithmBase<F> {
        &mut self.base
    }

    fn generation(&mut self) {
        'a: for i in 0..self.base.pop_num {
            self.vector(i);
            self.recombination(i);
            for s in 0..self.base.dim {
                if self.tmp[s] > self.ub(s) || self.tmp[s] < self.lb(s) {
                    continue 'a;
                }
            }
            let tmp_f = self.base.func.fitness(&self.tmp, &self.base.report);
            if tmp_f < self.base.fitness[i] {
                self.assign_from(i, tmp_f, &self.tmp.clone());
            }
        }
        self.find_best();
    }
}
