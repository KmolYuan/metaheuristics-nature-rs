use crate::{Algorithm, ObjFunc};

/// A trait that provides a conversion to original setting.
///
/// The setting type is actually a builder of the [`Setting::Algorithm`] type.
pub trait Setting {
    /// Associated algorithm.
    type Algorithm<F: ObjFunc>: Algorithm<F> + 'static;

    /// Create the algorithm.
    fn algorithm<F: ObjFunc>(self) -> Self::Algorithm<F>;

    /// Default population number.
    fn default_pop() -> usize {
        200
    }
}
