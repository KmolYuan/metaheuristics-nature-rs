use std::fmt::Debug;

/// The return value of the objective function ([`ObjFunc`](crate::ObjFunc)).
///
/// Usually, we can use numeric [`f64`] / [`f32`] type as the return value.
/// But more advanced is that any type that implements the requirement trait can
/// be used, so the fitness can add special mark during comparison
/// ([`PartialOrd`]).
///
/// In the following example, an "important" marker has higher priority in the
/// comparison.
///
/// ```
/// use metaheuristics_nature::{utility::Fitness, Bounded, ObjFunc};
/// use std::cmp::Ordering;
///
/// struct MyFunc;
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
/// impl Bounded for MyFunc {
///     fn bound(&self) -> &[[f64; 2]] {
///         &[[0., 50.]; 2]
///     }
/// }
///
/// impl ObjFunc for MyFunc {
///     type Fitness = MarkerFitness;
///
///     fn fitness(&self, x: &[f64]) -> Self::Fitness {
///         MarkerFitness { f: x[0], important: x[0] + x[1] < 1. }
///     }
/// }
/// ```
pub trait Fitness: Sync + Send + Default + Clone + PartialOrd + Debug {}
impl<T> Fitness for T where T: Sync + Send + Default + Clone + PartialOrd + Debug {}
