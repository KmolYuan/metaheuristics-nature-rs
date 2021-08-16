//! If the `parallel` feature is enabled,
//! this module provides a thread pool to spawn the objective function and collect the results.

use crate::{AsArray, ObjFunc, Report};
use alloc::{
    collections::{btree_map::IntoIter, BTreeMap},
    sync::Arc,
};
#[cfg(feature = "parallel")]
extern crate std;
#[cfg(feature = "parallel")]
use std::thread::{spawn, JoinHandle};

/// A join handler collector.
///
/// This type implements [`IntoIterator`] that consume the pool,
/// and the tasks can be wait by a for-loop.
///
/// ```
/// use std::sync::Arc;
/// use metaheuristics_nature::{Report, thread_pool::ThreadPool};
/// # use metaheuristics_nature::{ObjFunc, Array1, AsArray};
/// # struct MyFunc([f64; 3], [f64; 3]);
/// # impl MyFunc {
/// #     fn new() -> Self { Self([0.; 3], [50.; 3]) }
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
/// #     fn ub(&self) -> &[f64] { &self.1 }
/// #     fn lb(&self) -> &[f64] { &self.0 }
/// # }
///
/// let mut tasks = ThreadPool::new();
/// tasks.insert(0, Arc::new(MyFunc::new()), Report::default(), &Array1::zeros(3));
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
        {
            let job = spawn(move || f.fitness(&*v, &report));
            self.tasks.insert(i, job);
        }
        #[cfg(not(feature = "parallel"))]
        {
            let fit = f.fitness(&*v, &report);
            self.tasks.insert(i, fit);
        }
    }
}

impl IntoIterator for ThreadPool {
    type Item = (usize, f64);
    type IntoIter = IntoIter<usize, f64>;

    fn into_iter(self) -> Self::IntoIter {
        self.tasks
            .into_iter()
            .map(|(i, j)| {
                #[cfg(feature = "parallel")]
                let j = j.join().unwrap();
                (i, j)
            })
            .collect::<BTreeMap<_, _>>()
            .into_iter()
    }
}
