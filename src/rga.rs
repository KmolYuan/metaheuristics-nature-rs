use crate::{Algorithm, AlgorithmBase, ObjFunc, Setting, Task};

setting_builder! {
    /// Real-coded Genetic Algorithm settings.
    pub struct RGASetting {
        @base: Setting,
        cross: f64,
        mutate: f64,
        win: f64,
        delta: f64,
    }
}

impl Default for RGASetting {
    fn default() -> Self {
        Self {
            base: Setting::default().pop_num(500),
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
    tmp: Vec<Vec<f64>>,
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
            tmp: zeros!(3, base.dim),
            f_tmp: zeros!(3),
            new_pool: zeros!(base.pop_num, base.dim),
            base,
        }
    }
    fn crossover(&mut self) {
        for i in (0..(self.base.pop_num - 1)).step_by(2) {
            if !maybe!(self.cross) {
                continue;
            }
            for s in 0..self.base.dim {
                self.tmp[0][s] = 0.5 * self.base.pool[i][s] + 0.5 * self.base.pool[i + 1][s];
                self.tmp[1][s] = self.check(
                    s,
                    1.5 * self.base.pool[i][s] - 0.5 * self.base.pool[i + 1][s],
                );
                self.tmp[2][s] = self.check(
                    s,
                    -0.5 * self.base.pool[i][s] + 1.5 * self.base.pool[i + 1][s],
                );
            }
            for j in 0..3 {
                self.f_tmp[j] = self.base.func.fitness(self.base.gen, &self.tmp[j]);
            }
            if self.f_tmp[0] > self.f_tmp[1] {
                self.f_tmp.swap(0, 1);
                self.tmp.swap(0, 1);
            }
            if self.f_tmp[0] > self.f_tmp[2] {
                self.f_tmp.swap(0, 2);
                self.tmp.swap(0, 2);
            }
            if self.f_tmp[1] > self.f_tmp[2] {
                self.f_tmp.swap(1, 2);
                self.tmp.swap(1, 2);
            }
            self.assign_from(i, self.f_tmp[0], self.tmp[0].clone());
            self.assign_from(i + 1, self.f_tmp[1], self.tmp[1].clone());
        }
    }
    fn get_delta(&self, y: f64) -> f64 {
        let r = match self.base.task {
            Task::MaxGen(v) if v > 0 => self.base.gen as f64 / v as f64,
            _ => 1.,
        };
        y * rand!() * (1. - r).powf(self.delta)
    }
    fn mutate(&mut self) {
        for i in 0..self.base.pop_num {
            if !maybe!(self.mutate) {
                continue;
            }
            let s = rand!(0, self.base.dim);
            if maybe!(0.5) {
                self.base.pool[i][s] += self.get_delta(self.ub(s) - self.base.pool[i][s]);
            } else {
                self.base.pool[i][s] -= self.get_delta(self.base.pool[i][s] - self.lb(s));
            }
            self.base.fitness(i);
        }
        self.find_best();
    }
    fn select(&mut self) {
        for i in 0..self.base.pop_num {
            let j = rand!(0, self.base.pop_num);
            let k = rand!(0, self.base.pop_num);
            if self.base.fitness[j] > self.base.fitness[k] && maybe!(self.win) {
                self.new_fitness[i] = self.base.fitness[k];
                self.new_pool[i] = self.base.pool[k].clone();
            } else {
                self.new_fitness[i] = self.base.fitness[j];
                self.new_pool[i] = self.base.pool[j].clone();
            }
            self.base.fitness = self.new_fitness.clone();
            self.base.pool = self.new_pool.clone();
            self.assign_from(
                rand!(0, self.base.pop_num),
                self.base.best_f,
                self.base.best.clone(),
            );
        }
    }
}

impl<F: ObjFunc> Algorithm<F> for RGA<F> {
    fn base(&self) -> &AlgorithmBase<F> {
        &self.base
    }
    fn base_mut(&mut self) -> &mut AlgorithmBase<F> {
        &mut self.base
    }
    fn generation(&mut self) {
        self.select();
        self.crossover();
        self.mutate();
    }
    fn check(&self, s: usize, v: f64) -> f64 {
        if self.ub(s) < v || self.lb(s) > v {
            rand!(self.lb(s), self.ub(s))
        } else {
            v
        }
    }
}
