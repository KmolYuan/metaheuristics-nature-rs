//! The utility API used to create a new algorithm.
//!
//! When building a new method, just import [`prelude`] of this module.
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
