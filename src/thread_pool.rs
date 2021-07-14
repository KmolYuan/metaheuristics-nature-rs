use crate::{ObjFunc, Report};
use ndarray::AsArray;
use std::{
    collections::{hash_map::IntoIter, HashMap},
    sync::Arc,
    thread::{spawn, JoinHandle},
};

/// A join handler collector.
#[derive(Default)]
pub struct ThreadPool {
    tasks: HashMap<usize, JoinHandle<f64>>,
}

impl ThreadPool {
    /// Create a new thread pool.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert<'a, F, V>(&mut self, i: usize, f: Arc<F>, report: Report, v: V)
    where
        F: ObjFunc,
        V: AsArray<'a, f64>,
    {
        let v = Arc::new(v.into().to_owned());
        self.tasks.insert(i, spawn(move || f.fitness(&*v, &report)));
    }
}

impl IntoIterator for ThreadPool {
    type Item = (usize, f64);
    type IntoIter = IntoIter<usize, f64>;

    fn into_iter(self) -> Self::IntoIter {
        let m = self
            .tasks
            .into_iter()
            .map(|(i, j)| (i, j.join().unwrap()))
            .collect::<HashMap<_, _>>();
        m.into_iter()
    }
}
