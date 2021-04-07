use crate::{zeros, rand, maybe, AlgorithmBase, Algorithm, Setting, ObjFunc};

#[derive(Copy, Clone)]
pub enum Strategy {
    S1,
    S2,
    S3,
    S4,
    S5,
    S6,
    S7,
    S8,
    S9,
    S10,
}

/// Differential Evolution settings.
pub struct DESetting {
    pub base: Setting,
    pub strategy: Strategy,
    pub f: f64,
    pub cr: f64,
}

impl Default for DESetting {
    fn default() -> Self {
        Self {
            base: Setting {
                pop_num: 400,
                ..Default::default()
            },
            strategy: Strategy::S1,
            f: 0.6,
            cr: 0.9,
        }
    }
}

/// Differential Evolution type.
pub struct DE<F: ObjFunc> {
    f: f64,
    cr: f64,
    v: Vec<usize>,
    tmp: Vec<f64>,
    formula: fn(&Self, usize) -> f64,
    setter: fn(&mut Self, usize),
    base: AlgorithmBase<F>,
}

impl<F: ObjFunc> DE<F> {
    pub fn new(func: F, settings: DESetting) -> Self {
        let base = AlgorithmBase::new(func, settings.base);
        let num = match settings.strategy {
            Strategy::S1 | Strategy::S3 | Strategy::S6 | Strategy::S8 => 2,
            Strategy::S2 | Strategy::S7 => 3,
            Strategy::S4 | Strategy::S9 => 4,
            Strategy::S5 | Strategy::S10 => 5,
        };
        Self {
            f: settings.f,
            cr: settings.cr,
            v: vec![0; num],
            tmp: zeros!(base.dim),
            formula: match settings.strategy {
                Strategy::S1 | Strategy::S6 => Self::f1,
                Strategy::S2 | Strategy::S7 => Self::f2,
                Strategy::S3 | Strategy::S8 => Self::f3,
                Strategy::S4 | Strategy::S9 => Self::f4,
                Strategy::S5 | Strategy::S10 => Self::f5,
            },
            setter: match settings.strategy {
                Strategy::S1 | Strategy::S2 | Strategy::S3
                | Strategy::S4 | Strategy::S5 => Self::s1,
                Strategy::S6 | Strategy::S7 | Strategy::S8
                | Strategy::S9 | Strategy::S10 => Self::s2,
            },
            base,
        }
    }
    fn vector(&mut self, i: usize) {
        for j in 0..self.v.len() {
            self.v[j] = i;
            while self.v[j] == i || self.v[..j].contains(&self.v[j]) {
                self.v[j] = rand!(0, self.base.pop_num);
            }
        }
    }
    fn f1(&self, n: usize) -> f64 {
        self.base.best[n] + self.f
            * (self.base.pool[self.v[0]][n] - self.base.pool[self.v[1]][n])
    }
    fn f2(&self, n: usize) -> f64 {
        self.base.pool[self.v[0]][n] + self.f
            * (self.base.pool[self.v[1]][n] - self.base.pool[self.v[3]][n])
    }
    fn f3(&self, n: usize) -> f64 {
        self.tmp[n] + self.f * (self.base.best[n] - self.tmp[n]
            + self.base.pool[self.v[0]][n] - self.base.pool[self.v[1]][n])
    }
    fn f4(&self, n: usize) -> f64 {
        self.base.best[n] + self.f
            * (self.base.pool[self.v[0]][n] + self.base.pool[self.v[1]][n]
            - self.base.pool[self.v[2]][n] - self.base.pool[self.v[3]][n])
    }
    fn f5(&self, n: usize) -> f64 {
        self.base.pool[self.v[4]][n] + self.f
            * (self.base.pool[self.v[0]][n] + self.base.pool[self.v[1]][n]
            - self.base.pool[self.v[2]][n] - self.base.pool[self.v[3]][n])
    }
    fn s1(&mut self, mut n: usize) {
        let mut lv = 0;
        loop {
            self.tmp[n] = (self.formula)(self, n);
            n = (n + 1) % self.base.dim;
            lv += 1;
            if !maybe!(self.cr) || lv >= self.base.dim {
                break;
            }
        }
    }
    fn s2(&mut self, mut n: usize) {
        for lv in 0..self.base.dim {
            if !maybe!(self.cr) || lv == self.base.dim - 1 {
                self.tmp[n] = (self.formula)(self, n);
            }
            n = (n + 1) % self.base.dim;
        }
    }
    fn recombination(&mut self, i: usize) {
        self.tmp = self.base.pool[i].clone();
        (self.setter)(self, rand!(0, self.base.dim));
    }
    fn check(&self) -> bool {
        for i in 0..self.base.dim {
            if self.tmp[i] > self.ub(i) || self.tmp[i] < self.lb(i) {
                return true;
            }
        }
        false
    }
}

impl<F: ObjFunc> Algorithm<F> for DE<F> {
    fn base(&self) -> &AlgorithmBase<F> { &self.base }
    fn base_mut(&mut self) -> &mut AlgorithmBase<F> { &mut self.base }
    fn init(&mut self) {
        self.init_pop();
        self.find_best();
    }
    fn generation(&mut self) {
        for i in 0..self.base.pop_num {
            self.vector(i);
            self.recombination(i);
            if self.check() {
                continue;
            }
            let tmp_f = self.base.func.fitness(self.base.gen, &self.tmp);
            if tmp_f < self.base.fitness[i] {
                self.assign_from(i, tmp_f, self.tmp.clone());
            }
        }
        self.find_best();
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        tests::{test, TestObj},
        {DE, DESetting, Setting, Task},
    };

    #[test]
    fn de() {
        test(DE::new(
            TestObj::new(),
            DESetting {
                base: Setting {
                    task: Task::MinFit,
                    stop_at: 1e-20,
                    ..Default::default()
                },
                ..Default::default()
            },
        ));
    }
}
