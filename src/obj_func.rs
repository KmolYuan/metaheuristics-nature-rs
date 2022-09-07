use crate::utility::prelude::*;

/// A trait for the objective function.
///
/// ```
/// use metaheuristics_nature::ObjFunc;
///
/// struct MyFunc([[f64; 2]; 3]);
///
/// impl MyFunc {
///     fn new() -> Self {
///         Self([[0., 50.]; 3])
///     }
/// }
///
/// impl ObjFunc for MyFunc {
///     type Result = f64;
///     type Fitness = f64;
///
///     fn fitness(&self, x: &[f64], _: f64) -> Self::Fitness {
///         x[0] * x[0] + x[1] * x[1] + x[2] * x[2]
///     }
///
///     fn result(&self, xs: &[f64]) -> Self::Result {
///         self.fitness(xs, 0.)
///     }
///
///     fn bound(&self) -> &[[f64; 2]] {
///         &self.0
///     }
/// }
/// ```
///
/// The objective function returns fitness value that used to evaluate the
/// objective. The lower bound and upper bound represents the number of
/// variables at the same time.
///
/// This trait is designed as immutable and there should only has shared data.
pub trait ObjFunc: Sync + Send {
    /// The result type.
    type Result;
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
    /// [`SolverBuilder::adaptive`] method.
    fn fitness(&self, xs: &[f64], f: f64) -> Self::Fitness;

    /// Return the final result of the problem.
    ///
    /// The parameters `xs` is the best parameter we found.
    fn result(&self, xs: &[f64]) -> Self::Result;

    /// The upper bound and lower bound.
    ///
    /// This function should be cheap.
    fn bound(&self) -> &[[f64; 2]];
}
