//! The utility API for advanced usage.
//!
//! There is a [`prelude`] module that will import all the items of this crate.
//!
//! ```
//! use metaheuristics_nature::utility::prelude::*;
//! ```
pub use self::{algorithm::*, context::*, fitness::*, random::*, setting::*, solver_builder::*};

mod algorithm;
mod context;
mod fitness;
mod random;
mod setting;
mod solver_builder;

/// The re-export of some rayon functions. Rayon's prelude is in [`prelude`].
///
/// + `spawn` is a function sends a sub-task to the background and does not care about its return,
///   so this function will not block the current thread,
///   same as the [`std::thread::spawn`] but it still managed by rayon thread-pool.
///   This function requires a **static lifetime**.
/// + `scope` is the function runs a local callable with return value,
///   and there is a `Scope` object lets multiple callables **join** at this function.
///   So this function will wait all subtasks to return, which will block the current thread.
///   This function can be **passed by reference**.
///
/// The following example shows a block-on model that creates a simple fork-join structure.
///
/// ```
/// use metaheuristics_nature::{utility::thread::scope, Rga, Solver};
/// # use metaheuristics_nature::tests::TestObj as MyFunc;
///
/// scope(|s| {
///     // Fork here
///     // Task 1
///     s.spawn(|_| {
///         let s = Solver::build(Rga::default())
///             .task(|ctx| ctx.gen == 20)
///             .solve(MyFunc::new());
///         /* ... */
///     });
///     // Task 2
///     s.spawn(|_| ());
///     // Join here (then return)
/// });
/// ```
///
/// # First-in-First-out (FIFO)
///
/// Rayon's thread-pool is last-in-first-out (LIFO),
/// when the number of tasks becomes greater in a short time,
/// it means that the subtask order is upside-down.
/// If you want to keep the sending order, please use the FIFO version.
///
/// # Panics
///
/// Unify with rayon's thread-pool has another benefit is that
/// the propagation of panics is connected with each others,
/// which prevents the main thread waiting for a crashed task.
#[cfg(feature = "parallel")]
#[cfg_attr(doc_cfg, doc(cfg(feature = "parallel")))]
pub mod thread {
    #[doc(no_inline)]
    pub use rayon::{scope, scope_fifo, spawn, spawn_fifo};
}

/// A prelude module for algorithm implementation.
///
/// This module includes all items of this crate, some hidden types,
/// and external items from "ndarray" and "rayon" (if `parallel` feature enabled).
///
/// # Fitness
///
/// To customize a fitness type, please see [`Fitness`].
///
/// # Algorithm
///
/// To implement an algorithm, please see [`Algorithm`].
pub mod prelude {
    pub use super::*;
    pub use crate::*;

    #[doc(no_inline)]
    pub use ndarray::{s, Array1, Array2, AsArray, Axis, Zip};
    #[cfg(feature = "parallel")]
    #[doc(no_inline)]
    #[cfg_attr(doc_cfg, doc(cfg(feature = "parallel")))]
    pub use rayon::prelude::*;
}
