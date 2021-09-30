/// The return value of the objective function.
pub trait Respond: Clone {
    /// Infinity value of the respond.
    const INFINITY: Self;
    /// Create from fitness value.
    fn from_value(v: f64, feasible: bool) -> Self;
    /// The fitness value.
    fn value(&self) -> f64;
    /// Return true if this respond is feasible.
    fn feasible(&self) -> Option<bool>;
}

impl Respond for f64 {
    const INFINITY: Self = Self::INFINITY;

    fn from_value(v: f64, _: bool) -> Self {
        v
    }

    fn value(&self) -> f64 {
        *self
    }

    fn feasible(&self) -> Option<bool> {
        None
    }
}

impl<T: Respond> Respond for (T, bool) {
    const INFINITY: Self = (T::INFINITY, false);

    fn from_value(v: f64, feasible: bool) -> Self {
        (T::from_value(v, feasible), feasible)
    }

    fn value(&self) -> f64 {
        self.0.value()
    }

    fn feasible(&self) -> Option<bool> {
        Some(self.1)
    }
}
