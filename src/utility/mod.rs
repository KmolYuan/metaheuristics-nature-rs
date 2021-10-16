//! The utility API used to create a new algorithm.
//!
//! When building a new method, just import [`prelude`] of this module.
//!
//! ```
//! use metaheuristics_nature::utility::prelude::*;
//! ```
pub use self::{algorithm::Algorithm, context::Context, respond::Respond, setting::BasicSetting};
pub use crate::random::*;

mod algorithm;
mod context;
mod respond;
pub(crate) mod setting;

/// Product two iterators together.
///
/// For example, `[a, b, c]` and `[1, 2, 3]` will become `[a1, a2, a3, b1, b2, b3, c1, c2, c3]`.
pub fn product<A, I1, I2>(iter1: I1, iter2: I2) -> impl Iterator<Item = (A, A)>
where
    A: Clone,
    I1: Iterator<Item = A>,
    I2: Iterator<Item = A> + Clone,
{
    iter1.flat_map(move |e| core::iter::repeat(e).zip(iter2.clone()))
}

/// A prelude module for algorithm implementation.
///
/// This module includes all items of this crate,
/// and some items from "ndarray".
pub mod prelude {
    pub use super::*;
    pub use crate::{random::*, thread_pool::ThreadPool, *};
    pub use ndarray::{s, Array1, Array2, AsArray};
}
