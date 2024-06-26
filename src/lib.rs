#![doc = include_str!("../README.md")]
//! # Terminologies
//!
//! For unifying the terms, in this documentation,
//!
//! + "Iteration" is called "generation". (Avoid confusion with iterators)
//! + "Function" that evaluates the design is called "objective function".
//! + "Return value" of the objective function is called "fitness".
//!
//! # Algorithms
//!
//! There are two traits [`Algorithm`] and [`AlgCfg`].
//! The previous is used to design the optimization method,
//! and the latter is the setting interface.
//!
//! [`Solver`] is a simple interface for obtaining the solution, or analyzing
//! the result. This type allows you to use the pre-defined methods without
//! importing any traits.
//!
//! All provided methods are listed in the module [`methods`].
//!
//! For making your owned method, please see [`prelude`].
//!
//! # Objective Function
//!
//! For a quick demo with callable object, please see [`Fx`].
//!
//! You can define your question as an objective function through implementing
//! [`ObjFunc`], and then the upper bound, lower bound, and an objective
//! function [`ObjFunc::fitness()`] returns [`Fitness`] should be defined.
//!
//! # Random Function
//!
//! This crate uses a 64bit ChaCha algorithm ([`random::Rng`]) to generate
//! uniform random values. Before that, a random seed is required. The seed is
//! generated by `getrandom` crate, please see its support platform.
//!
//! # Features
//!
//! The crate features:
//! + `std`: Default feature. Enable standard library function, such as timing
//!   and threading. If `std` is disabled, crate "libm" will be enabled for the
//!   math functions.
//! + `rayon`: Enable parallel computation via `rayon`. Disable it for the
//!   platform that doesn't supported threading, or if your objective function
//!   is not complicate enough. This feature require `std` feature.
//! + `clap`: Add CLI argument support for the provided algorithms and their
//!   options.
//!
//! # Compatibility
//!
//! If you are using this crate for providing objective function,
//! other downstream crates of yours may have some problems with compatibility.
//!
//! The most important thing is using a stable version, specifying the major
//! version number. Then re-export (`pub use`) this crate for the downstream
//! crates.
//!
//! This crate does the same things on `rand` and `rayon`.
#![cfg_attr(doc_cfg, feature(doc_auto_cfg))]
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;
#[cfg(not(feature = "std"))]
extern crate core as std;
pub use rand;
#[cfg(feature = "rayon")]
pub use rayon;

pub use self::{
    algorithm::*, ctx::*, fitness::*, fx_func::*, methods::*, obj_func::*, solver::*,
    solver_builder::*,
};

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
        pub fn $name(self, $name: $ty) -> Self {
            Self { $name, ..self }
        }
    )+};
}

/// A prelude module for algorithm implementation.
///
/// This module includes all items of this crate, some hidden types,
/// and external items from "ndarray" and "rayon" (if `rayon` feature enabled).
pub mod prelude {
    pub use super::*;
    pub use crate::{pareto::*, random::*};

    #[cfg(feature = "rayon")]
    #[doc(no_inline)]
    pub use crate::rayon::prelude::*;
    #[cfg(not(feature = "std"))]
    pub use num_traits::Float as _;
}

mod algorithm;
mod ctx;
mod fitness;
mod fx_func;
pub mod methods;
mod obj_func;
pub mod pareto;
pub mod random;
mod solver;
mod solver_builder;
pub mod tests;

/// A marker trait for parallel computation.
///
/// Require `Sync + Send` if the `rayon` feature is enabled, otherwise require
/// nothing.
#[cfg(not(feature = "rayon"))]
pub trait MaybeParallel {}
#[cfg(not(feature = "rayon"))]
impl<T> MaybeParallel for T {}

/// A marker trait for parallel computation.
///
/// Require `Sync + Send` if the `rayon` feature is enabled, otherwise require
/// nothing.
#[cfg(feature = "rayon")]
pub trait MaybeParallel: Sync + Send {}
#[cfg(feature = "rayon")]
impl<T: Sync + Send> MaybeParallel for T {}

#[cfg(feature = "rayon")]
macro_rules! maybe_send_box {
    ($($traits:tt)+) => {
        Box<dyn $($traits)+ + Send>
    };
}
#[cfg(not(feature = "rayon"))]
macro_rules! maybe_send_box {
    ($($traits:tt)+) => {
        Box<dyn $($traits)+ >
    };
}
pub(crate) use maybe_send_box;
