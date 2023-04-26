/// The return value of the objective function ([`ObjFunc`](crate::ObjFunc)).
///
/// Usually, we can use numeric [`f64`] / [`f32`] type as the return value.
/// But more advanced is that any type that implements the requirement trait can
/// be used, so the fitness can add special mark during comparison
/// ([`PartialOrd`]).
///
/// # Validity
///
/// `f.partial_cmp(&f).is_none()` should returns true if the fitness is invalid.
///
/// # Example
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
///
/// # See Also
///
/// [`Product`] provides a field for a final result.
pub trait Fitness: Sync + Send + Default + Clone + PartialOrd {}
impl<T> Fitness for T where T: Sync + Send + Default + Clone + PartialOrd {}

/// A [`Fitness`] type carrying final results.
///
/// You can use [`Solver::result()`](crate::Solver) to access product field.
#[derive(Default, Clone)]
pub struct Product<P, F: Fitness> {
    /// Product
    pub product: P,
    /// Fitness
    pub fitness: F,
}

impl<P, F: Fitness> Product<P, F> {
    /// Create a product.
    pub fn new(product: P, fitness: F) -> Self {
        Self { product, fitness }
    }
}

impl<P, F: Fitness> PartialEq for Product<P, F> {
    fn eq(&self, other: &Self) -> bool {
        self.fitness.eq(&other.fitness)
    }
}

impl<P, F: Fitness> PartialOrd for Product<P, F> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.fitness.partial_cmp(&other.fitness)
    }
}
