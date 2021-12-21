/// The return value of the objective function ([`ObjFunc`](crate::ObjFunc)).
///
/// Usually, we can use numeric [`f64`] / [`f32`] type as the return value.
/// But more advanced is that any type that implements the requirement trait can be used,
/// so the fitness can add special mark during comparison.
///
/// In the following example, an "important" marker has higher priority in the comparison.
///
/// ```
/// use metaheuristics_nature::{utility::Fitness, ObjFunc};
/// use std::cmp::Ordering;
/// # struct MyFunc([f64; 2], [f64; 2]);
///
/// #[derive(Clone, PartialEq, Default)]
/// struct MarkerFitness {
///     f: f64,
///     important: bool,
/// }
///
/// impl PartialOrd for MarkerFitness {
///     fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
///         match (self.important, other.important) {
///             (true, false) => Some(Ordering::Greater),
///             (false, true) => Some(Ordering::Less),
///             (false, false) | (true, true) => self.f.partial_cmp(&other.f),
///         }
///     }
/// }
///
/// impl ObjFunc for MyFunc {
///     type Result = f64;
///     type Fitness = MarkerFitness;
///
///     fn fitness(&self, v: &[f64], f: f64) -> Self::Fitness {
///         MarkerFitness {
///             f: v[0],
///             important: v[0] + v[1] * f < 1.,
///         }
///     }
///
///     fn result(&self, v: &[f64]) -> Self::Result {
///         self.fitness(v, 0.).f
///     }
///
///     fn ub(&self) -> &[f64] {
///         &self.1
///     }
///
///     fn lb(&self) -> &[f64] {
///         &self.0
///     }
/// }
/// ```
pub trait Fitness: Sync + Send + Default + Clone + PartialOrd {}
impl<T> Fitness for T where T: Sync + Send + Default + Clone + PartialOrd {}
