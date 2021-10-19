//! The utility API used to create a new algorithm.
//!
//! When building a new method, just import [`prelude`] of this module.
//!
//! ```
//! use metaheuristics_nature::utility::prelude::*;
//! ```
pub use self::{algorithm::Algorithm, context::Context, respond::Respond, setting::BasicSetting};

mod algorithm;
mod context;
mod respond;
pub(crate) mod setting;

/// Product two iterators together.
///
/// ```
/// use metaheuristics_nature::utility::product;
///
/// let ans = product(['a', 'b', 'c'].into_iter().cloned(), 0..3).collect::<Vec<_>>();
/// assert_eq!(
///     vec![
///         ('a', 0),
///         ('a', 1),
///         ('a', 2),
///         ('b', 0),
///         ('b', 1),
///         ('b', 2),
///         ('c', 0),
///         ('c', 1),
///         ('c', 2),
///     ],
///     ans
/// );
/// ```
pub fn product<A, B, I1, I2>(iter1: I1, iter2: I2) -> impl Iterator<Item = (A, B)>
where
    A: Clone,
    I1: Iterator<Item = A>,
    I2: Iterator<Item = B> + Clone,
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
    pub use ndarray::{s, Array1, Array2, AsArray, Axis, Zip};
    #[cfg(feature = "parallel")]
    pub use rayon::prelude::*;
}
