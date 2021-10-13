//! This module provides a thread pool to spawn the objective function and collect the results.
//!
//! Although the "parallel" feature maybe disabled,
//! the thread pool is still can work with single thread.
#[cfg(feature = "parallel")]
extern crate std;

use crate::{
    utility::{AsArray, Respond},
    ObjFunc, Report,
};
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
/// use metaheuristics_nature::{thread_pool::ThreadPool, utility::Array1, Report};
/// use std::sync::Arc;
/// # use metaheuristics_nature::ObjFunc;
/// # struct MyFunc([f64; 3], [f64; 3]);
/// # impl MyFunc {
/// #     fn new() -> Self { Self([0.; 3], [50.; 3]) }
/// # }
/// # impl ObjFunc for MyFunc {
/// #     type Result = f64;
/// #     type Respond = f64;
/// #     fn fitness(&self, v: &[f64], _: &Report) -> Self::Respond {
/// #         v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
/// #     }
/// #     fn result(&self, v: &[f64]) -> Self::Result {
/// #         self.fitness(v, &Default::default())
/// #     }
/// #     fn ub(&self) -> &[f64] { &self.1 }
/// #     fn lb(&self) -> &[f64] { &self.0 }
/// # }
///
/// let mut tasks = ThreadPool::new();
/// tasks.insert(
///     0,
///     Arc::new(MyFunc::new()),
///     Report::default(),
///     &Array1::zeros(3),
/// );
///
/// for (i, f) in tasks {
///     assert_eq!(i, 0);
///     assert_eq!(f, 0.);
/// }
/// ```
pub struct ThreadPool<R: Respond> {
    #[cfg(feature = "parallel")]
    tasks: Vec<(usize, JoinHandle<R>)>,
    #[cfg(not(feature = "parallel"))]
    tasks: Vec<(usize, R)>,
}

impl<R: Respond> Default for ThreadPool<R> {
    fn default() -> Self {
        Self { tasks: Vec::new() }
    }
}

impl<R: Respond> ThreadPool<R> {
    /// Create a new thread pool.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the length of the pool.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.tasks.len()
    }

    /// Return true if the pool is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.tasks.is_empty()
    }

    /// Spawn a objective function task.
    pub fn insert<'a, F, V>(&mut self, i: usize, f: Arc<F>, report: Report, v: V)
    where
        F: ObjFunc<Respond = R>,
        V: AsArray<'a, f64>,
    {
        #[cfg(feature = "parallel")]
        {
            let v = v.into().to_shared();
            let job = spawn(move || f.fitness(v.as_slice().unwrap(), &report));
            self.tasks.push((i, job));
        }
        #[cfg(not(feature = "parallel"))]
        {
            let v = v.into();
            let ans = f.fitness(v.as_slice().unwrap(), &report);
            self.tasks.push((i, ans));
        }
    }

    /// Join and consume the pool, get an iterable result list.
    ///
    /// Instead of calling [`IntoIterator::into_iter`] method,
    /// this method will ignore the index (job number).
    ///
    /// ```
    /// # use std::sync::Arc;
    /// # use metaheuristics_nature::{ObjFunc, thread_pool::ThreadPool, utility::Array1, Report};
    /// # struct MyFunc([f64; 3], [f64; 3]);
    /// # impl MyFunc {
    /// #     fn new() -> Self { Self([0.; 3], [50.; 3]) }
    /// # }
    /// # impl ObjFunc for MyFunc {
    /// #     type Result = f64;
    /// #     type Respond = f64;
    /// #     fn fitness(&self, v: &[f64], _: &Report) -> Self::Respond {
    /// #         v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
    /// #     }
    /// #     fn result(&self, v: &[f64]) -> Self::Result {
    /// #         self.fitness(v, &Default::default())
    /// #     }
    /// #     fn ub(&self) -> &[f64] { &self.1 }
    /// #     fn lb(&self) -> &[f64] { &self.0 }
    /// # }
    /// # let mut tasks = ThreadPool::new();
    /// # tasks.insert(0, Arc::new(MyFunc::new()), Report::default(), &Array1::zeros(3));
    /// let mut ans = Vec::with_capacity(1);
    /// ans.extend(tasks.join());
    /// # assert_eq!(ans[0], 0.);
    /// ```
    pub fn join(self) -> impl Iterator<Item = R> {
        #[cfg(feature = "parallel")]
        {
            self.tasks.into_iter().map(|(_, f)| f.join().unwrap())
        }
        #[cfg(not(feature = "parallel"))]
        {
            self.tasks.into_iter().map(|(_, f)| f)
        }
    }
}

impl<R: Respond> IntoIterator for ThreadPool<R> {
    type Item = (usize, R);
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
        {
            self.tasks.into_iter()
        }
    }
}
