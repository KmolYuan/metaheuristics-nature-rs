/// The return value of the objective function.
///
/// Usually, the fitness can use [`f64`] and `(f64, bool)` type as the return value.
pub trait Respond: Sync + Send + Clone + 'static {
    /// Infinity value of the respond.
    const INFINITY: Self;
    /// Create from fitness value.
    fn from_value(v: f64, feasible: bool) -> Self;
    /// The fitness value.
    fn value(&self) -> f64;
    /// Return true if this respond is feasible.
    fn feasible(&self) -> bool;
}

impl Respond for f64 {
    const INFINITY: Self = Self::INFINITY;
    #[inline(always)]
    fn from_value(v: f64, _: bool) -> Self {
        v
    }
    #[inline(always)]
    fn value(&self) -> f64 {
        *self
    }
    #[inline(always)]
    fn feasible(&self) -> bool {
        false
    }
}

impl Respond for (f64, bool) {
    const INFINITY: Self = (f64::INFINITY, false);
    #[inline(always)]
    fn from_value(v: f64, feasible: bool) -> Self {
        (f64::from_value(v, feasible), feasible)
    }
    #[inline(always)]
    fn value(&self) -> f64 {
        self.0.value()
    }
    #[inline(always)]
    fn feasible(&self) -> bool {
        self.1
    }
}
