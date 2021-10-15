macro_rules! impl_base_setting {
    ($name:ident, $ty:ty) => {
        /// Set the base option.
        fn $name(mut self, $name: $ty) -> Self {
            self.base_mut().$name = $name;
            self
        }
    };
}

/// Setting base. This type store the basic configurations that provides to the algorithm framework.
///
/// This type should be included in the custom setting, which implements [`Setting`].
#[derive(Debug, PartialEq)]
pub struct BasicSetting {
    /// Termination condition.
    pub task: Task,
    /// Population number.
    pub pop_num: usize,
    /// Report frequency. (per generation)
    pub rpt: u64,
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
pub trait Setting: Sized {
    /// Associated algorithm.
    ///
    /// This type should implement [`Algorithm`](crate::utility::Algorithm) trait.
    type Algorithm;
    /// Get original setting.
    fn base(&self) -> &BasicSetting;
    /// Get mutable original setting.
    fn base_mut(&mut self) -> &mut BasicSetting;
    /// Create the algorithm.
    fn create(self) -> Self::Algorithm;

    impl_base_setting!(task, Task);
    impl_base_setting!(pop_num, usize);
    impl_base_setting!(rpt, u64);
    impl_base_setting!(average, bool);
    impl_base_setting!(adaptive, Adaptive);
}

/// Terminal condition of the algorithm setting.
#[derive(Clone, Debug, PartialEq)]
pub enum Task {
    /// Max generation.
    MaxGen(u64),
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
    /// The return type [`ObjFunc::Respond`](crate::ObjFunc::Respond) can be set to `(f64, bool)`.
    ///
    /// See [`Respond`](crate::utility::Respond) for more information.
    Custom,
    /// Disable this option.
    Disable,
}
