use crate::utility::Algorithm;
use crate::ObjFunc;

/// Setting base. This type store the basic configurations that provides to the algorithm framework.
///
/// Please see [setting!] for more usage.
#[derive(Debug, PartialEq)]
pub struct BasicSetting {
    /// Termination condition.
    pub task: Task,
    /// Population number.
    pub pop_num: usize,
    /// Report frequency. (per generation)
    pub rpt: u32,
    /// Calculate the average of the fitness at [`Report`](crate::Report).
    /// Default to false.
    pub average: bool,
    /// Threshold of the adaptive factor. Default to disable this function.
    pub adaptive: Adaptive,
}

impl Default for BasicSetting {
    fn default() -> Self {
        Self {
            task: Task::MaxGen(200),
            pop_num: 200,
            rpt: 1,
            average: false,
            adaptive: Adaptive::Disable,
        }
    }
}

/// A trait that provides a conversion to original setting.
///
/// The setting type is actually a builder of the [`Setting::Algorithm`] type.
///
/// Before the implementation,
/// the builder function of the setting type can be implemented by [`setting!`].
pub trait Setting<F: ObjFunc> {
    /// Associated algorithm.
    type Algorithm: Algorithm<F>;
    /// Convert to original setting.
    fn base(&self) -> &BasicSetting;
    /// Create the algorithm.
    fn create(self) -> Self::Algorithm;
}

/// Terminal condition of the algorithm setting.
#[derive(Clone, Debug, PartialEq)]
pub enum Task {
    /// Max generation.
    MaxGen(u32),
    /// Minimum fitness.
    MinFit(f64),
    /// Max time in second.
    #[cfg(feature = "std")]
    #[cfg_attr(doc_cfg, doc(cfg(feature = "std")))]
    MaxTime(f32),
    /// Minimum delta value.
    SlowDown(f64),
}

/// Adaptive factor option.
///
/// The adaptive function will provide a factor for "adaptive penalty factor".
///
/// The factor is calculated by dividing the "feasible individuals" by the "total individuals",
/// where the "feasible individuals" is decided by the threshold.
#[derive(Clone, Debug, PartialEq)]
pub enum Adaptive {
    /// Use constant threshold.
    Constant(f64),
    /// Use the average of the finite fitness as threshold.
    Average,
    /// Custom mark from objective function.
    ///
    /// See [`Respond`](crate::utility::Respond) for more information.
    Custom,
    /// Disable this option.
    Disable,
}
