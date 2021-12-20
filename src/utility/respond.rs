/// The return value of the objective function.
///
/// Usually, the fitness can use [`f64`] type as the return value.
pub trait Respond: Sync + Send + Clone + PartialOrd + PartialEq + 'static {
    /// Infinity value of the respond.
    const INFINITY: Self;

    /// The fitness value.
    fn value(&self) -> f64;
}

impl Respond for f64 {
    const INFINITY: Self = Self::INFINITY;

    #[inline(always)]
    fn value(&self) -> f64 {
        *self
    }
}
