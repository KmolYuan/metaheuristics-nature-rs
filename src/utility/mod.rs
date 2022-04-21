//! The utility API for advanced usage.
//!
//! There is a [`prelude`] module that will import all the items of this crate.
//!
//! ```
//! use metaheuristics_nature::utility::prelude::*;
//! ```
pub use self::{context::*, fitness::*, random::*, solver_builder::*};

mod context;
mod fitness;
mod random;
mod solver_builder;

/// A prelude module for algorithm implementation.
///
/// This module includes all items of this crate, some hidden types,
/// and external items from "ndarray" and "rayon" (if `rayon` feature enabled).
pub mod prelude {
    pub use super::*;
    pub use crate::*;

    #[doc(no_inline)]
    pub use crate::ndarray::*;
    #[cfg(feature = "rayon")]
    #[cfg_attr(doc_cfg, doc(cfg(feature = "rayon")))]
    #[doc(no_inline)]
    pub use crate::rayon::prelude::*;
}
