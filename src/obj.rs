use crate::utility::prelude::*;

/// A problem is well bounded.
///
/// Provide constant array reference or dynamic slice for the variables.
pub trait Bounded: Sync + Send {
    /// The upper bound and lower bound in `[[lower, upper]; number_of_vars]`
    /// form.
    ///
    /// This function should be cheap.
    fn bound(&self) -> &[[f64; 2]];
}

/// A trait for the objective function.
///
/// ```
/// use metaheuristics_nature::{Bounded, ObjFunc};
///
/// struct MyFunc;
///
/// impl Bounded for MyFunc {
///     fn bound(&self) -> &[[f64; 2]] {
///         &[[0., 50.]; 3]
///     }
/// }
///
/// impl ObjFunc for MyFunc {
///     type Fitness = f64;
///
///     fn fitness(&self, x: &[f64]) -> Self::Fitness {
///         x[0] * x[0] + x[1] * x[1] + x[2] * x[2]
///     }
/// }
/// ```
///
/// The objective function returns fitness value that used to evaluate the
/// objective. The lower bound and upper bound represents the number of
/// variables at the same time.
///
/// This trait is designed as immutable and there should only has shared data.
pub trait ObjFunc: Bounded {
    /// Representation of the fitness value.
    type Fitness: Fitness;

    /// Return fitness, the smaller value represents a good result.
    ///
    /// # How to design the fitness value?
    ///
    /// Regularly, the evaluation value **should not** lower than zero,
    /// because it is not easy to control with multiplication,
    /// and a negative infinity can directly break the result.
    /// Instead, a positive enhanced floating point value is the better choice,
    /// and the zero is the best result.
    ///
    /// In another hand, some marker can help you to compare with other design,
    /// please see [`Fitness`] for more information.
    ///
    /// # Penalty
    ///
    /// In another hand, positive infinity represents the worst, or illogical
    /// result. In fact, the searching area (or we called feasible solution)
    /// should keeping not bad results, instead of evaluating them as the
    /// worst one, because of we can keep the searching inspection around
    /// the best result, to finding our potential winner.
    ///
    /// In order to distinguish how bad the result is, we can add a penalty
    /// value, which represents the "fault" on the result.
    ///
    /// Under most circumstances, the result is not good enough, appearing on
    /// its fitness value. But sometimes a result is badly than our normal
    /// results, if we mark them as the worst one (infinity), it will become
    /// a great "wall", which is not suitable for us to search across it.
    ///
    /// So that, we use secondary evaluation function to measure the result from
    /// other requirements, we call it "constraint" or "penalty function".
    /// The penalty value usually multiply a weight factor for increasing its
    /// influence.
    ///
    /// # Adaptive Value
    ///
    /// Sometimes a value that adjust with converge states can help to restrict
    /// the searching. The "adaptive function" can be set in
    /// [`SolverBuilder::callback()`] method.
    fn fitness(&self, xs: &[f64]) -> Self::Fitness;
}

/// A trait same as [`ObjFunc`] but returns a "product" and then evaluates it.
///
/// This is a higher level interface than [`ObjFunc`], it will auto-implement
/// for this trait.
pub trait ObjFactory: Bounded {
    /// "Product" type.
    type Product;
    /// Representation of the evaluation.
    type Eval: Fitness;

    /// Return a product of the problem.
    fn produce(&self, xs: &[f64]) -> Self::Product;

    /// This function same as [`ObjFunc::fitness()`] function but receive the
    /// product type.
    fn evaluate(&self, product: Self::Product) -> Self::Eval;
}

impl<F: ObjFactory> ObjFunc for F {
    type Fitness = <Self as ObjFactory>::Eval;

    fn fitness(&self, xs: &[f64]) -> Self::Fitness {
        self.evaluate(self.produce(xs))
    }
}