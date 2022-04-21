//! A collection of nature-inspired meta-heuristic algorithms.
//!
//! # Terminology
//!
//! For unifying the terms, in this documentation,
//!
//! + "Iteration" is called "generation".
//! + "Function" that evaluates value is called "objective function".
//! + "Return value" of the objective function is called "fitness".
//!
//! # Algorithm
//!
//! There are two traits [`utility::Algorithm`] and [`utility::Setting`].
//! The previous is used to design the optimization method,
//! and the latter is the setting interface.
//!
//! [`Solver`] is a simple interface for obtaining the solution, or analyzing the result.
//! This type allows you to use the pre-defined methods without importing any traits.
//!
//! All provided methods are listed in the module [`methods`].
//!
//! For making your owned method, please see [`utility::prelude`].
//!
//! # Objective Function
//!
//! For a quick demo with callable object, please see [`Fx`] or [`FxAdaptive`].
//!
//! You can define your question as an objective function through implementing [`ObjFunc`],
//! and then the upper bound, lower bound, and objective function [`ObjFunc::fitness`] should be defined.
//!
//! The final answer is [`ObjFunc::result`], which is calculated from the design parameters.
//!
//! # Random Function
//!
//! This crate use 64bit PRNG algorithm ([`utility::Rng`]) to generate uniform random values,
//! before that, a random seed is required.
//! The seed is generated by `getrandom`, please see its [support platform](getrandom#supported-targets).
//!
//! # Features
//!
//! + `std`: Default feature. Enable standard library function, such as timing and threading.
//! + `parallel`: Enable parallel function, let objective function running without ordered, uses `rayon`.
//!   Disable it for the platform that doesn't supported threading,
//!   or if your objective function is not complicate enough.
//!   This feature require `std` feature.
//! + `libm`: If the standard library is not provided, some math functions might missing.
//!   This will disable some pre-implemented algorithms.
//!   However, there is a math library implemented in pure Rust, the name is same as `libm`.
//!   This feature can re-enable (or replace) the math functions by using the `libm` crate.
//!
//! # Compatibility
//!
//! If you are using this crate for providing objective function,
//! other downstream crates of yours may have some problems with compatibility.
//!
//! The most important thing is using a stable version, specifying the major version number.
//! Then re-export (`pub use`) this crate for the downstream crates.
#![cfg_attr(doc_cfg, feature(doc_cfg))]
#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
extern crate alloc;
#[cfg(not(feature = "std"))]
extern crate core as std;

pub use crate::{fx_func::*, methods::*, obj_func::*, solver::*};

/// A tool macro used to generate multiple builder functions (methods).
///
/// For example,
///
/// ```
/// # use metaheuristics_nature::impl_builders;
/// # type Ty = bool;
/// # struct S {
/// #     name1: Ty,
/// #     name2: Ty,
/// # }
/// impl S {
///     impl_builders! {
///         /// Doc 1
///         fn name1(Ty)
///         /// Doc 2
///         fn name2(Ty)
///     }
/// }
/// ```
///
/// will become
///
/// ```
/// # type Ty = bool;
/// # struct S {
/// #     name1: Ty,
/// #     name2: Ty,
/// # }
/// impl S {
///     /// Doc 1
///     pub fn name1(mut self, name1: Ty) -> Self {
///         self.name1 = name1;
///         self
///     }
///     /// Doc 2
///     pub fn name2(mut self, name2: Ty) -> Self {
///         self.name2 = name2;
///         self
///     }
/// }
/// ```
#[macro_export]
macro_rules! impl_builders {
    ($($(#[$meta:meta])* fn $name:ident($ty:ty))+) => {$(
        $(#[$meta])*
        pub fn $name(mut self, $name: $ty) -> Self {
            self.$name = $name;
            self
        }
    )+};
}

mod fx_func;
pub mod methods;
mod obj_func;
mod solver;
pub mod tests;
pub mod utility;

/// The re-export of rayon items. Rayon's prelude is in [`utility::prelude`].
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
/// use metaheuristics_nature::{rayon::scope, Rga, Solver};
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
#[cfg(feature = "parallel")]
#[cfg_attr(doc_cfg, doc(cfg(feature = "parallel")))]
pub mod rayon {
    #[doc(no_inline)]
    pub use rayon::*;
}
