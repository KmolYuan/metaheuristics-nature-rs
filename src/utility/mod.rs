//! The utility API used to create a new algorithm.
//!
//! When building a new method, just import this module as prelude.
//!
//! ```
//! use metaheuristics_nature::{utility::*, *};
//! ```
//!
//! In other hand, if you went to fork the task manually by using parallel structure,
//! import [`thread_pool::ThreadPool`](crate::thread_pool::ThreadPool) is required.
pub use self::{
    algorithm::Algorithm,
    context::Context,
    respond::Respond,
    setting::{BasicSetting, Setting},
};
pub use crate::random::*;
use crate::ObjFunc;
pub use ndarray::{s, Array1, Array2, AsArray};

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
