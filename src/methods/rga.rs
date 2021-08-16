#[cfg(feature = "parallel")]
use crate::thread_pool::ThreadPool;
use crate::{random::*, *};
use ndarray::{s, Array1, Array2};

setting_builder! {
    /// Real-coded Genetic Algorithm settings.
    pub struct RGASetting for RGA {
        @base,
        @pop_num = 500,
        /// Crossing probability.
        cross: f64 = 0.95,
        /// Mutation probability.
        mutate: f64 = 0.05,
        /// Winning probability.
        win: f64 = 0.95,
        /// Delta factor.
        delta: f64 = 5.,
    }
}

/// Real-coded Genetic Algorithm type.
pub struct RGA {
    cross: f64,
    mutate: f64,
    win: f64,
    delta: f64,
    new_fitness: Array1<f64>,
    new_pool: Array2<f64>,
}

impl RGA {
    fn crossover<F: ObjFunc>(&mut self, ctx: &mut Context<F>) {
        for i in (0..(ctx.pop_num - 1)).step_by(2) {
            if !maybe(self.cross) {
                continue;
            }
            let mut tmp = Array2::zeros((3, ctx.dim));
            let mut f_tmp = Array1::zeros(3);
            for s in 0..ctx.dim {
                tmp[[0, s]] = 0.5 * ctx.pool[[i, s]] + 0.5 * ctx.pool[[i + 1, s]];
                let v = 1.5 * ctx.pool[[i, s]] - 0.5 * ctx.pool[[i + 1, s]];
                tmp[[1, s]] = Self::check(ctx, s, v);
                let v = -0.5 * ctx.pool[[i, s]] + 1.5 * ctx.pool[[i + 1, s]];
                tmp[[2, s]] = Self::check(ctx, s, v);
            }
            #[cfg(feature = "parallel")]
            let mut tasks = ThreadPool::new();
            for j in 0..3 {
                #[cfg(feature = "parallel")]
                {
                    tasks.insert(
                        j,
                        ctx.func.clone(),
                        ctx.report.clone(),
                        tmp.slice(s![j, ..]),
                    );
                }
                #[cfg(not(feature = "parallel"))]
                {
                    f_tmp[j] = ctx.func.fitness(tmp.slice(s![j, ..]), &ctx.report);
                }
            }
            #[cfg(feature = "parallel")]
            for (j, f) in tasks {
                f_tmp[j] = f;
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
            ctx.assign_from(i, f_tmp[0], tmp.slice(s![0, ..]));
            ctx.assign_from(i + 1, f_tmp[1], tmp.slice(s![1, ..]));
        }
    }

    fn get_delta<F: ObjFunc>(&self, ctx: &Context<F>, y: f64) -> f64 {
        let r = match ctx.task {
            Task::MaxGen(v) if v > 0 => ctx.report.gen as f64 / v as f64,
            _ => 1.,
        };
        y * rand() * (1. - r).powf(self.delta)
    }

    fn mutate<F: ObjFunc>(&mut self, ctx: &mut Context<F>) {
        for i in 0..ctx.pop_num {
            if !maybe(self.mutate) {
                continue;
            }
            let s = rand_int(0, ctx.dim);
            if maybe(0.5) {
                ctx.pool[[i, s]] += self.get_delta(ctx, ctx.ub(s) - ctx.pool[[i, s]]);
            } else {
                ctx.pool[[i, s]] -= self.get_delta(ctx, ctx.pool[[i, s]] - ctx.lb(s));
            }
            ctx.fitness(i);
        }
        ctx.find_best();
    }

    fn select<F: ObjFunc>(&mut self, ctx: &mut Context<F>) {
        for i in 0..ctx.pop_num {
            let j = rand_int(0, ctx.pop_num);
            let k = rand_int(0, ctx.pop_num);
            // FIXME
            assert!(i < ctx.pop_num);
            assert!(j < ctx.pop_num);
            assert!(k < ctx.pop_num);
            if ctx.fitness[j] > ctx.fitness[k] && maybe(self.win) {
                self.new_fitness[i] = ctx.fitness[k];
                self.new_pool
                    .slice_mut(s![i, ..])
                    .assign(&ctx.pool.slice(s![k, ..]));
            } else {
                self.new_fitness[i] = ctx.fitness[j];
                self.new_pool
                    .slice_mut(s![i, ..])
                    .assign(&ctx.pool.slice(s![j, ..]));
            }
            ctx.fitness.assign(&self.new_fitness);
            ctx.pool.assign(&self.new_pool);
            ctx.assign_from_best(rand_int(0, ctx.pop_num));
        }
    }

    #[inline(always)]
    fn check<F: ObjFunc>(ctx: &Context<F>, s: usize, v: f64) -> f64 {
        if ctx.ub(s) < v || v < ctx.lb(s) {
            rand_float(ctx.lb(s), ctx.ub(s))
        } else {
            v
        }
    }
}

impl Algorithm for RGA {
    type Setting = RGASetting;

    fn create(settings: &Self::Setting) -> Self {
        Self {
            cross: settings.cross,
            mutate: settings.mutate,
            win: settings.win,
            delta: settings.delta,
            new_pool: Array2::zeros((1, 1)),
            new_fitness: Array1::ones(1) * f64::INFINITY,
        }
    }

    #[inline(always)]
    fn init<F: ObjFunc>(&mut self, ctx: &mut Context<F>) {
        self.new_pool = ctx.pool.clone();
        self.new_fitness = ctx.fitness.clone();
    }

    #[inline(always)]
    fn generation<F: ObjFunc>(&mut self, ctx: &mut Context<F>) {
        self.select(ctx);
        self.crossover(ctx);
        self.mutate(ctx);
    }
}
