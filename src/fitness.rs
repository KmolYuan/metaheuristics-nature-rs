use crate::prelude::*;
use alloc::sync::Arc;

/// Trait for dominance comparison.
///
/// By default, the trait is implemented for types that implement `PartialOrd +
/// Clone + 'static`, which means a clonable non-lifetime type comparable with
/// `a < b` is equivalent to [`a.is_dominated(b)`](Fitness::is_dominated) for
/// using single objective.
///
/// # Example
///
/// Single objective problems can simply use the `f32`/`f64` number type.
///
/// Multi-objective problems can specify the [`Pareto`] container as
/// [`Fitness::Best`] and implement [`Fitness::eval()`] to decide the final
/// fitness value.
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
    /// Used in [`Best::as_result()`] and [`Best::update()`] when reaching the
    /// limit.
    fn eval(&self) -> Self::Eval;
}

impl<T: MaybeParallel + PartialOrd + Clone + 'static> Fitness for T {
    type Best<A: Fitness> = SingleBest<A>;
    type Eval = Self;
    fn is_dominated(&self, rhs: &Self) -> bool {
        self < rhs
    }
    fn eval(&self) -> Self::Eval {
        self.clone()
    }
}

/// A [`Fitness`] type carrying a multi-objective [`Fitness`] value. Make it
/// become a single objective task via using [`Fitness::eval()`].
///
/// This wrapper type is overrided [`Fitness::Best`] to [`SingleBest`]. A
/// multi-objective fitness type can be tested in single mode by setting
/// [`ObjFunc::Ys`] to `MakeSingle<MyMOFit>` and wrapping the final result with
/// `MakeSingle(MyMOFit { .. })`.
#[derive(Clone, Debug)]
#[repr(transparent)]
pub struct MakeSingle<Y: Fitness>(pub Y)
where
    Y::Eval: Fitness;

impl<Y: Fitness> Fitness for MakeSingle<Y>
where
    Y::Eval: Fitness,
{
    type Best<T: Fitness> = SingleBest<T>;
    type Eval = Y::Eval;
    fn is_dominated(&self, rhs: &Self) -> bool {
        self.eval().is_dominated(&rhs.eval())
    }
    #[inline]
    fn eval(&self) -> Self::Eval {
        self.0.eval()
    }
}

/// A [`Fitness`] type carrying final results.
///
/// You can use [`Solver::as_best_xs()`] / [`Solver::as_best_fit()`] /
/// [`Solver::get_best_eval()`] to access product field.
#[derive(Clone, Debug)]
pub struct WithProduct<Y, P: ?Sized> {
    ys: Y,
    product: Arc<P>,
}

impl<Y, P: ?Sized> WithProduct<Y, P> {
    /// Create a product from an existing [`Arc`] object, where `P` can be
    /// unknown size.
    pub fn new_from_arc(ys: Y, product: Arc<P>) -> Self {
        Self { ys, product }
    }

    /// Get the reference to the final result.
    pub fn as_result(&self) -> &P {
        self.product.as_ref()
    }
}

impl<Y, P> WithProduct<Y, P> {
    /// Create a product.
    pub fn new(ys: Y, product: P) -> Self {
        Self::new_from_arc(ys, Arc::new(product))
    }

    /// Get the fitness value.
    pub fn ys(&self) -> Y
    where
        Y: Clone,
    {
        self.ys.clone()
    }

    /// Consume and get the final result.
    pub fn into_result(self) -> P
    where
        P: Clone,
    {
        self.into_err_result().1
    }

    /// Get the fitness value and the final result.
    pub fn into_err_result(self) -> (Y, P)
    where
        P: Clone,
    {
        (self.ys, Arc::unwrap_or_clone(self.product))
    }
}

impl<Y: Fitness, P> Fitness for WithProduct<Y, P>
where
    P: MaybeParallel + Clone + 'static,
{
    type Best<T: Fitness> = Y::Best<T>;
    type Eval = Y::Eval;
    fn is_dominated(&self, rhs: &Self) -> bool {
        self.ys.is_dominated(&rhs.ys)
    }
    fn eval(&self) -> Self::Eval {
        self.ys.eval()
    }
}
