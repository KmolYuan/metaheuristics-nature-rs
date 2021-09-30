/// The return value of the objective function.
pub trait Respond {
    /// The fitness value.
    fn value(&self) -> f64;
    /// Return true if this respond is feasible.
    fn feasible(&self) -> Option<bool>;
}

impl Respond for f64 {
    fn value(&self) -> f64 {
        *self
    }

    fn feasible(&self) -> Option<bool> {
        None
    }
}

impl<T: Respond> Respond for (T, bool) {
    fn value(&self) -> f64 {
        self.0.value()
    }

    fn feasible(&self) -> Option<bool> {
        Some(self.1)
    }
}
