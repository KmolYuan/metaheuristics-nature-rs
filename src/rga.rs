use std::iter::{repeat, FromIterator};
use crate::utility::{AlgorithmBase, Algorithm, Settings, ObjFunc};
use crate::zeros;

pub struct RGASetting {
    base: Settings,
    cross: f64,
    mutate: f64,
    win: f64,
    delta: f64,
}

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
}

impl<F: ObjFunc> Algorithm<F> for RGA<F> {
    fn base(&self) -> &AlgorithmBase<F> { &self.base }
    fn base_mut(&mut self) -> &mut AlgorithmBase<F> { &mut self.base }
    fn init(&mut self) {
        unimplemented!()
    }
    fn generation(&mut self) {
        unimplemented!()
    }
}
