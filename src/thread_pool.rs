//! This module provides a thread pool to spawn the objective function and collect the results.
//!
//! Although the "parallel" feature maybe disabled,
//! the thread pool is still can work with single thread.
#[cfg(feature = "parallel")]
extern crate std;

use crate::{AsArray, ObjFunc, Report};
use alloc::{sync::Arc, vec::Vec};
#[cfg(feature = "parallel")]
use std::thread::{spawn, JoinHandle};

/// A join handler collector.
///
/// If the feature "parallel" is enabled,
/// the jobs will be spawned when inserted.
/// Otherwise, this container will run the objective function immediately.
///
/// This type implements [`IntoIterator`] that consume the pool,
/// and the tasks can be wait by a for-loop.
///
/// ```
/// use std::sync::Arc;
/// use metaheuristics_nature::{thread_pool::ThreadPool, Report};
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
    tasks: Vec<(usize, JoinHandle<f64>)>,
    #[cfg(not(feature = "parallel"))]
    tasks: Vec<(usize, f64)>,
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
        #[cfg(feature = "parallel")]
        {
            let v = v.into().to_shared();
            let job = spawn(move || f.fitness(&v, &report));
            self.tasks.push((i, job));
        }
        #[cfg(not(feature = "parallel"))]
        {
            let fit = f.fitness(v, &report);
            self.tasks.push((i, fit));
        }
    }
}

impl IntoIterator for ThreadPool {
    type Item = (usize, f64);
    type IntoIter = <Vec<Self::Item> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        #[cfg(feature = "parallel")]
        {
            self.tasks
                .into_iter()
                .map(|(i, j)| (i, j.join().unwrap()))
                .collect::<Vec<_>>()
                .into_iter()
        }
        #[cfg(not(feature = "parallel"))]
        self.tasks.into_iter()
    }
}
