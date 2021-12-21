/// A trait that provides a conversion to original setting.
///
/// The setting type is actually a builder of the [`Setting::Algorithm`] type.
pub trait Setting {
    /// Associated algorithm.
    ///
    /// This type should implement [`Algorithm`](crate::utility::Algorithm) trait.
    type Algorithm;

    /// Create the algorithm.
    fn algorithm(self) -> Self::Algorithm;

    /// Default population number.
    fn default_pop() -> usize {
        200
    }
}
