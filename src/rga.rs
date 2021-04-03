use crate::utility::{AlgorithmBase, Algorithm, Settings, ObjFunc, Task};
use crate::{zeros, rand};

/// Real-coded Genetic Algorithm settings.
pub struct RGASetting {
    pub base: Settings,
    pub cross: f64,
    pub mutate: f64,
    pub win: f64,
    pub delta: f64,
}

impl Default for RGASetting {
    fn default() -> Self {
        Self {
            base: Settings {
                pop_num: 500,
                ..Default::default()
            },
            cross: 0.95,
            mutate: 0.05,
            win: 0.95,
            delta: 5.,
        }
    }
}

/// Real-coded Genetic Algorithm type.
pub struct RGA<F: ObjFunc> {
    cross: f64,
    mutate: f64,
    win: f64,
    delta: f64,
    new_fitness: Vec<f64>,
    tmp1: Vec<f64>,
    tmp2: Vec<f64>,
    tmp3: Vec<f64>,
    f_tmp: Vec<f64>,
    new_pool: Vec<Vec<f64>>,
    base: AlgorithmBase<F>,
}

impl<F: ObjFunc> RGA<F> {
    pub fn new(func: F, settings: RGASetting) -> Self {
        let base = AlgorithmBase::new(func, settings.base);
        Self {
            cross: settings.cross,
            mutate: settings.mutate,
            win: settings.win,
            delta: settings.delta,
            new_fitness: zeros!(base.pop_num),
            tmp1: zeros!(base.dim),
            tmp2: zeros!(base.dim),
            tmp3: zeros!(base.dim),
            f_tmp: zeros!(3),
            new_pool: zeros!(base.pop_num, base.dim),
            base,
        }
    }
    fn check(&self, i: usize, v: f64) -> f64 {
        if self.ub(i) < v || self.lb(i) > v {
            rand!(self.lb(i), self.ub(i))
        } else { v }
    }
    fn crossover(&mut self) {
        for i in (0..(self.base.pop_num - 1)).step_by(2) {
            if !(rand!() < self.cross) {
                continue;
            }
            for s in 0..self.base.dim {
                self.tmp1[s] = 0.5 * self.base.pool[i][s] + 0.5 * self.base.pool[i + 1][s];
                self.tmp2[s] = self.check(s, 1.5 * self.base.pool[i][s] - 0.5 * self.base.pool[i + 1][s]);
                self.tmp3[s] = self.check(s, -0.5 * self.base.pool[i][s] + 1.5 * self.base.pool[i + 1][s]);
            }
            self.f_tmp[0] = self.base.func.fitness(self.base.gen, &self.tmp1);
            self.f_tmp[1] = self.base.func.fitness(self.base.gen, &self.tmp2);
            self.f_tmp[2] = self.base.func.fitness(self.base.gen, &self.tmp3);
            if self.f_tmp[0] > self.f_tmp[1] {
                self.f_tmp.swap(0, 1);
                std::mem::swap(&mut self.tmp1, &mut self.tmp2);
            }
            if self.f_tmp[0] > self.f_tmp[2] {
                self.f_tmp.swap(0, 2);
                std::mem::swap(&mut self.tmp1, &mut self.tmp3);
            }
            if self.f_tmp[1] > self.f_tmp[2] {
                self.f_tmp.swap(1, 2);
                std::mem::swap(&mut self.tmp2, &mut self.tmp3);
            }
            self.assign_from(i, self.f_tmp[0], self.tmp1.clone());
            self.assign_from(i + 1, self.f_tmp[1], self.tmp2.clone());
        }
    }
    fn get_delta(&self, y: f64) -> f64 {
        let r = if self.base.task == Task::MaxGen && self.base.stop_at > 0. {
            self.base.gen as f64 / self.base.stop_at
        } else { 1. };
        y * rand!() * (1. - r).powf(self.delta)
    }
    fn mutate(&mut self) {
        for i in 0..self.base.pop_num {
            if !(rand!() < self.mutate) {
                continue;
            }
            let s = rand!(self.base.dim);
            if rand!() < 0.5 {
                self.base.pool[i][s] += self.get_delta(self.ub(s) - self.base.pool[i][s]);
            } else {
                self.base.pool[i][s] -= self.get_delta(self.base.pool[i][s] - self.lb(s));
            }
            self.base.fitness[i] = self.base.func.fitness(self.base.gen, &self.base.pool[i]);
        }
        self.find_best();
    }
    fn select(&mut self) {
        for i in 0..self.base.pop_num {
            let j = rand!(self.base.pop_num);
            let k = rand!(self.base.pop_num);
            if self.base.fitness[j] > self.base.fitness[k] && rand!() < self.win {
                self.new_fitness[i] = self.base.fitness[k];
                self.new_pool[i] = self.base.pool[k].clone();
            } else {
                self.new_fitness[i] = self.base.fitness[j];
                self.new_pool[i] = self.base.pool[j].clone();
            }
            self.base.fitness = self.new_fitness.clone();
            self.base.pool = self.new_pool.clone();
            self.assign_from(rand!(self.base.pop_num), self.base.best_f, self.base.best.clone());
        }
    }
}

impl<F: ObjFunc> Algorithm<F> for RGA<F> {
    fn base(&self) -> &AlgorithmBase<F> { &self.base }
    fn base_mut(&mut self) -> &mut AlgorithmBase<F> { &mut self.base }
    fn init(&mut self) {
        self.init_pop();
        self.set_best(0);
    }
    fn generation(&mut self) {
        self.select();
        self.crossover();
        self.mutate();
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::TestObj;
    use crate::rga::{RGA, RGASetting};
    use crate::utility::{Algorithm, Settings, Task};

    #[test]
    fn rga() {
        let mut a = RGA::new(
            TestObj::new(),
            RGASetting {
                base: Settings {
                    task: Task::MinFit,
                    stop_at: 1e-20,
                    ..Default::default()
                },
                ..Default::default()
            }
        );
        let ans = a.run();
        let (x, y) = a.result();
        assert!(ans.abs() < 1e-20);
        assert!(x[0].abs() < 1e-20);
        assert!(x[1].abs() < 1e-20);
        assert!(y.abs() < 1e-20);
    }
}
