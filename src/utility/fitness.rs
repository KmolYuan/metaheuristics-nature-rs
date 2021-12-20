/// The return value of the objective function.
///
/// Usually, we can use numeric [`f64`] / [`f32`] type as the return value.
/// More advanced, any type that implements the requirement trait can be used,
/// so the fitness can add special mark during comparison.
pub trait Fitness: Sync + Send + Default + Clone + PartialOrd {}
impl<T> Fitness for T where T: Sync + Send + Default + Clone + PartialOrd {}
