#[cfg(feature = "std")]
extern crate std;
#[cfg(feature = "std")]
use std::time::Instant;

/// The data of generation sampling.
#[derive(Clone, Debug)]
pub struct Report {
    /// Generation.
    pub gen: u32,
    /// Best fitness.
    pub best_f: f64,
    /// Gradient of the best fitness, between the current and the previous.
    pub diff: f64,
    /// Average of the finite-fitness individuals.
    ///
    /// The first value might be [`f64::NaN`].
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
            diff: 0.,
            average: f64::NAN,
            adaptive: 0.,
            #[cfg(feature = "std")]
            time: 0.,
        }
    }
}

impl Report {
    #[cfg(feature = "std")]
    pub(crate) fn update_time(&mut self, time: Instant) {
        self.time = (Instant::now() - time).as_secs_f64();
    }
}
