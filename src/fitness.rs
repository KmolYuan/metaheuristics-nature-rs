use crate::prelude::*;
use alloc::boxed::Box;

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
/// use metaheuristics_nature::{Bounded, Fitness, ObjFunc};
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
/// impl Fitness for MarkerFitness {}
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
/// See also [`Product`].
pub trait Fitness: MaybeParallel + Clone + PartialOrd + 'static {
    /// Mark the value to non-best.
    fn mark_not_best(&mut self) {}
}

/// [`Fitness`] just implemented for the float number by default.
impl<T> Fitness for T where T: num_traits::Float + MaybeParallel + 'static {}

/// Trait for dominance comparison.
///
/// By default, the trait is implemented for types that implement `PartialOrd`,
/// which means that `a <= b` is equivalent to `a.is_dominated(b)` for using
/// single objective.
///
/// # Example
///
/// If your type has multiple objectives, you can use the [`pareto::Pareto`]
/// container and implement [`Dominance::eval()`] to decide the final fitness
/// value.
///
/// ```
/// use metaheuristics_nature::{pareto::Pareto, Dominance};
///
/// #[derive(Clone)]
/// struct MyObject {
///     cost: f64,
///     weight: f64,
/// }
///
/// impl Dominance for MyObject {
///     type Best = Pareto<Self>;
///     fn is_dominated(&self, rhs: &Self) -> bool {
///         self.cost <= rhs.cost && self.weight <= rhs.weight
///     }
///     fn eval(&self) -> impl PartialOrd + 'static {
///         self.cost.max(self.weight)
///     }
/// }
/// ```
pub trait Dominance: Clone {
    /// The best element container.
    type Best: pareto::Best<Item = Self>;
    /// Check if `self` dominates `rhs`.
    fn is_dominated(&self, rhs: &Self) -> bool;
    /// Evaluate the final fitness value.
    ///
    /// Used in [`pareto::Best::update()`] and [`pareto::Best::result()`].
    fn eval(&self) -> impl PartialOrd + 'static;
    // /// TODO: Mark the value to non-best.
    // fn mark_not_best(&mut self) {}
}

impl<T: PartialOrd + Clone + 'static> Dominance for T {
    type Best = pareto::SingleBest<T>;
    fn is_dominated(&self, rhs: &Self) -> bool {
        self <= rhs
    }
    fn eval(&self) -> impl PartialOrd + 'static {
        self.clone()
    }
}

/// A [`Fitness`] type carrying final results.
///
/// You can use [`Solver::as_result()`]/[`Solver::into_result()`] to access
/// product field.
#[derive(Default, Clone, Debug)]
pub struct Product<P, F: Fitness> {
    fitness: F,
    product: Option<Box<P>>,
}

impl<P, F: Fitness> Product<P, F> {
    /// Create a product.
    pub fn new(fitness: F, product: P) -> Self {
        Self { fitness, product: Some(Box::new(product)) }
    }

    /// Get the fitness value.
    pub fn fitness(&self) -> F {
        self.fitness.clone()
    }

    /// Get the reference to the final result.
    pub fn as_result(&self) -> &P {
        self.product.as_ref().unwrap()
    }

    /// Consume and get the final result.
    pub fn into_result(self) -> P {
        *self.product.unwrap()
    }

    /// Get the fitness value and the final result.
    pub fn into_inner(self) -> (F, P) {
        let Self { fitness, product } = self;
        (fitness, *product.unwrap())
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

impl<P: MaybeParallel + Clone + 'static, F: Fitness> Fitness for Product<P, F> {
    fn mark_not_best(&mut self) {
        self.product.take();
    }
}
