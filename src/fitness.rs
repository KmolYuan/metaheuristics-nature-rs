use crate::prelude::*;
use alloc::boxed::Box;

/// Trait for dominance comparison.
///
/// By default, the trait is implemented for types that implement `PartialOrd +
/// Clone + 'static`, which means a clonable non-lifetime type comparable with
/// `a <= b` is equivalent to [`a.is_dominated(b)`](Fitness::is_dominated) for
/// using single objective.
///
/// # Example
///
/// If your type has multiple objectives, you can use the [`Pareto`]
/// container and implement [`Fitness::eval()`] to decide the final fitness
/// value.
///
/// ```
/// use metaheuristics_nature::{pareto::Pareto, Fitness};
///
/// #[derive(Clone)]
/// struct MyObject {
///     cost: f64,
///     weight: f64,
/// }
///
/// impl Fitness for MyObject {
///     type Best<T: Fitness> = Pareto<T>;
///     type Eval = f64;
///     fn is_dominated(&self, rhs: &Self) -> bool {
///         self.cost <= rhs.cost && self.weight <= rhs.weight
///     }
///     fn eval(&self) -> Self::Eval {
///         self.cost.max(self.weight)
///     }
/// }
/// ```
pub trait Fitness: MaybeParallel + Clone + 'static {
    /// The best element container.
    /// + Use [`SingleBest`] for single objective.
    /// + Use [`Pareto`] for multi-objective.
    type Best<T: Fitness>: Best<Item = T>;
    /// A value to compare the final fitness value.
    type Eval: PartialOrd + 'static;
    /// Check if `self` dominates `rhs`.
    fn is_dominated(&self, rhs: &Self) -> bool;
    /// Evaluate the final fitness value.
    ///
    /// Used in [`Best::update()`] and [`Best::as_result()`].
    fn eval(&self) -> Self::Eval;
    /// Mark the value to non-best, used to drop the non-best results.
    fn mark_not_best(&mut self) {}
}

impl<T: MaybeParallel + PartialOrd + Clone + 'static> Fitness for T {
    type Best<A: Fitness> = SingleBest<A>;
    type Eval = Self;
    fn is_dominated(&self, rhs: &Self) -> bool {
        self <= rhs
    }
    fn eval(&self) -> Self::Eval {
        self.clone()
    }
}

/// A [`Fitness`] type carrying final results.
///
/// You can use [`Solver::as_best_xs()`] / [`Solver::as_best_fit()`] /
/// [`Solver::get_best_eval()`] to access product field.
#[derive(Default, Clone, Debug)]
pub struct Product<F, P> {
    fit: F,
    product: Option<Box<P>>,
}

impl<P, F> Product<F, P> {
    /// Create a product.
    pub fn new(fit: F, product: P) -> Self {
        Self { fit, product: Some(Box::new(product)) }
    }

    /// Get the fitness value.
    pub fn fitness(&self) -> F
    where
        F: Clone,
    {
        self.fit.clone()
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
    pub fn into_err_result(self) -> (F, P) {
        let Self { fit, product } = self;
        (fit, *product.unwrap())
    }
}

impl<P, F> Fitness for Product<F, P>
where
    P: MaybeParallel + Clone + 'static,
    F: Fitness,
{
    type Best<T: Fitness> = F::Best<T>;
    type Eval = F::Eval;
    fn is_dominated(&self, rhs: &Self) -> bool {
        self.fit.is_dominated(&rhs.fit)
    }
    fn eval(&self) -> Self::Eval {
        self.fit.eval()
    }
    fn mark_not_best(&mut self) {
        self.product.take();
    }
}
