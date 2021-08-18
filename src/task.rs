/// The terminal condition of the algorithm setting.
#[derive(Clone)]
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
