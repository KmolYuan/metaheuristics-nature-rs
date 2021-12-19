/// Data of generation sampling.
///
/// [`ObjFunc`](crate::ObjFunc) type can read the information from this type.
#[derive(Clone, Debug)]
pub struct Report {
    /// Generation.
    pub gen: u64,
    /// Best fitness.
    pub best_f: f64,
    /// Is the best fitness feasible.
    pub best_feasible: bool,
    /// Gradient of the best fitness, between the current and the previous.
    pub diff: f64,
}

impl Default for Report {
    fn default() -> Self {
        Self {
            gen: 0,
            best_f: f64::INFINITY,
            best_feasible: false,
            diff: 0.,
        }
    }
}
