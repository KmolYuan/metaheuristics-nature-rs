/// The data of generation sampling.
#[derive(Clone, Debug)]
pub struct Report {
    /// Generation.
    pub gen: u32,
    /// Best fitness.
    pub best_f: f64,
    /// Is the best fitness feasible.
    pub best_feasible: bool,
    /// Gradient of the best fitness, between the current and the previous.
    pub diff: f64,
    /// Average of the finite-fitness individuals.
    ///
    /// The first value might be [`f64::NAN`].
    pub average: f64,
    /// Adaptive factor.
    pub adaptive: f64,
    /// Time duration.
    #[cfg(feature = "std")]
    #[cfg_attr(doc_cfg, doc(cfg(feature = "std")))]
    pub time: f64,
}

impl Default for Report {
    fn default() -> Self {
        Self {
            gen: 0,
            best_f: f64::INFINITY,
            best_feasible: false,
            diff: 0.,
            average: f64::NAN,
            adaptive: 0.,
            #[cfg(feature = "std")]
            time: 0.,
        }
    }
}
