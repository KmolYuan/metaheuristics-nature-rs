#[cfg(feature = "std")]
extern crate std;
#[cfg(feature = "std")]
use std::time::Instant;

/// The data of generation sampling.
#[derive(Clone, Debug)]
pub struct Report {
    /// Generation.
    pub gen: u32,
    /// The best fitness.
    pub best_f: f64,
    /// The gradient of the best fitness.
    pub diff: f64,
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
            #[cfg(feature = "std")]
            time: 0.,
        }
    }
}

impl Report {
    pub(crate) fn next_gen(&mut self) {
        self.gen += 1;
    }

    pub(crate) fn set_diff(&mut self, diff: f64) {
        self.diff = diff;
    }

    #[cfg(feature = "std")]
    pub(crate) fn update_time(&mut self, time: Instant) {
        self.time = (Instant::now() - time).as_secs_f64();
    }
}
