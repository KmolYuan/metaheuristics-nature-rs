//! If the `parallel` feature is enabled,
//! this module provides a thread pool to spawn the objective function and collect the results.

use crate::{ObjFunc, Report};
use alloc::{
    collections::{btree_map::IntoIter, BTreeMap},
    sync::Arc,
};
use ndarray::AsArray;
use std::ops::Deref;
#[cfg(feature = "parallel")]
use std::thread::{spawn, JoinHandle};

/// A join handler collector.
///
/// This type implements [`IntoIterator`] that consume the pool,
/// and the tasks can be wait by a for-loop.
///
/// ```
/// use std::sync::Arc;
/// use metaheuristics_nature::{Report, ObjFunc, thread_pool::ThreadPool};
/// # use ndarray::{Array1, AsArray, ArrayView1, array};
/// # struct MyFunc(Array1<f64>, Array1<f64>);
/// # impl MyFunc {
/// #     fn new() -> Self { Self(Array1::zeros(3), Array1::ones(3) * 50.) }
/// # }
/// # impl ObjFunc for MyFunc {
/// #     type Result = f64;
/// #     fn fitness<'a, A>(&self, v: A, _: &Report) -> f64
/// #     where
/// #         A: AsArray<'a, f64>,
/// #     {
/// #         let v = v.into();
/// #         v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
/// #     }
/// #     fn result<'a, V>(&self, v: V) -> Self::Result
/// #     where
/// #         V: AsArray<'a, f64>
/// #     {
/// #         self.fitness(v, &Default::default())
/// #     }
/// #     fn ub(&self) -> ArrayView1<f64> { self.1.view() }
/// #     fn lb(&self) -> ArrayView1<f64> { self.0.view() }
/// # }
///
/// let mut tasks = ThreadPool::new();
/// tasks.insert(0, Arc::new(MyFunc::new()), Report::default(), &array![0., 0., 0.]);
///
/// for (i, f) in tasks {
///     assert_eq!(i, 0);
///     assert_eq!(f, 0.);
/// }
/// ```
#[derive(Default)]
pub struct ThreadPool {
    #[cfg(feature = "parallel")]
    tasks: BTreeMap<usize, JoinHandle<f64>>,
    #[cfg(not(feature = "parallel"))]
    tasks: BTreeMap<usize, f64>,
}

impl ThreadPool {
    /// Create a new thread pool.
    pub fn new() -> Self {
        Self::default()
    }

    /// Spawn a objective function task.
    pub fn insert<'a, F, V>(&mut self, i: usize, f: Arc<F>, report: Report, v: V)
    where
        F: ObjFunc,
        V: AsArray<'a, f64>,
    {
        let v = Arc::new(v.into().to_owned());
        #[cfg(feature = "parallel")]
        let fit = spawn(move || f.fitness(v.deref(), &report));
        #[cfg(not(feature = "parallel"))]
        let fit = f.fitness(v.deref(), &report);
        self.tasks.insert(i, fit);
    }
}

impl IntoIterator for ThreadPool {
    type Item = (usize, f64);
    type IntoIter = IntoIter<usize, f64>;

    fn into_iter(self) -> Self::IntoIter {
        let m = self
            .tasks
            .into_iter()
            .map(|(i, j)| {
                #[cfg(feature = "parallel")]
                let fit = j.join().unwrap();
                #[cfg(not(feature = "parallel"))]
                let fit = j;
                (i, fit)
            })
            .collect::<BTreeMap<_, _>>();
        m.into_iter()
    }
}
