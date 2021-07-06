use crate::*;
use ndarray::{s, Array1, Array2};

setting_builder! {
    /// Real-coded Genetic Algorithm settings.
    pub struct RGASetting {
        @base,
        @pop_num = 500,
        cross: f64 = 0.95,
        mutate: f64 = 0.05,
        win: f64 = 0.95,
        delta: f64 = 5.,
    }
}

/// Real-coded Genetic Algorithm type.
pub struct RGA<F: ObjFunc> {
    cross: f64,
    mutate: f64,
    win: f64,
    delta: f64,
    new_fitness: Array1<f64>,
    new_pool: Array2<f64>,
    base: AlgorithmBase<F>,
}

impl<F> RGA<F>
where
    F: ObjFunc,
{
    fn crossover(&mut self) {
        for i in (0..(self.base.pop_num - 1)).step_by(2) {
            if !maybe!(self.cross) {
                continue;
            }
            let mut tmp = Array2::zeros((3, self.base.dim));
            let mut f_tmp = Array1::zeros(3);
            for s in 0..self.base.dim {
                tmp[[0, s]] = 0.5 * self.base.pool[[i, s]] + 0.5 * self.base.pool[[i + 1, s]];
                tmp[[1, s]] = self.check(
                    s,
                    1.5 * self.base.pool[[i, s]] - 0.5 * self.base.pool[[i + 1, s]],
                );
                tmp[[2, s]] = self.check(
                    s,
                    -0.5 * self.base.pool[[i, s]] + 1.5 * self.base.pool[[i + 1, s]],
                );
            }
            for j in 0..3 {
                f_tmp[j] = self.base.func.fitness(self.base.gen, tmp.slice(s![j, ..]));
            }
            if f_tmp[0] > f_tmp[1] {
                f_tmp.swap(0, 1);
                for j in 0..2 {
                    tmp.swap([0, j], [1, j]);
                }
            }
            if f_tmp[0] > f_tmp[2] {
                f_tmp.swap(0, 2);
                for j in 0..2 {
                    tmp.swap([0, j], [2, j]);
                }
            }
            if f_tmp[1] > f_tmp[2] {
                f_tmp.swap(1, 2);
                for j in 0..2 {
                    tmp.swap([1, j], [2, j]);
                }
            }
            self.assign_from(i, f_tmp[0], &tmp.slice(s![0, ..]).into_owned());
            self.assign_from(i + 1, f_tmp[1], &tmp.slice(s![1, ..]).into_owned());
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
                self.base.pool[[i, s]] += self.get_delta(self.ub(s) - self.base.pool[[i, s]]);
            } else {
                self.base.pool[[i, s]] -= self.get_delta(self.base.pool[[i, s]] - self.lb(s));
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
                self.new_pool
                    .slice_mut(s![i, ..])
                    .assign(&self.base.pool.slice(s![k, ..]));
            } else {
                self.new_fitness[i] = self.base.fitness[j];
                self.new_pool
                    .slice_mut(s![i, ..])
                    .assign(&self.base.pool.slice(s![j, ..]));
            }
            self.base.fitness.assign(&self.new_fitness);
            self.base.pool.assign(&self.new_pool);
            self.assign_from(
                rand!(0, self.base.pop_num),
                self.base.best_f,
                &self.base.best.clone(),
            );
        }
    }
}

impl<F> Algorithm<F> for RGA<F>
where
    F: ObjFunc,
{
    type Setting = RGASetting;

    fn create(func: F, settings: Self::Setting) -> Self {
        let base = AlgorithmBase::new(func, settings.base);
        Self {
            cross: settings.cross,
            mutate: settings.mutate,
            win: settings.win,
            delta: settings.delta,
            new_fitness: Array1::zeros(base.pop_num),
            new_pool: Array2::zeros((base.pop_num, base.dim)),
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
