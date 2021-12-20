use crate::utility::Fitness;

/// The base of the objective function.
///
/// ```
/// use metaheuristics_nature::ObjFunc;
///
/// struct MyFunc([f64; 3], [f64; 3]);
///
/// impl MyFunc {
///     fn new() -> Self {
///         Self([0.; 3], [50.; 3])
///     }
/// }
///
/// impl ObjFunc for MyFunc {
///     type Result = f64;
///     type Fitness = f64;
///
///     fn fitness(&self, v: &[f64], _: f64) -> Self::Fitness {
///         v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
///     }
///
///     fn result(&self, v: &[f64]) -> Self::Result {
///         self.fitness(v, 0.)
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
/// The objective function returns fitness value that used to evaluate the objective.
///
/// The lower bound and upper bound represents the number of variables at the same time.
///
/// This trait is designed as immutable.
pub trait ObjFunc: Sync + Send + 'static {
    /// The result type.
    type Result;
    /// Representation of the fitness value.
    type Fitness: Fitness;

    /// Return fitness, the smaller value represents a good result.
    ///
    /// # How to design the fitness value?
    ///
    /// Regularly, the evaluation value **should not** lower than zero,
    /// because a negative infinity can directly break the result.
    /// Instead, a positive enhanced floating point value is the better choice.
    ///
    /// # Penalty
    ///
    /// In another hand, positive infinity represents the worst, or illogical result.
    /// In fact, the searching area (or we called feasible solution) should keeping not bad results,
    /// instead of evaluating them as the worst one,
    /// because of we can keep the searching inspection around the best result,
    /// to finding our potential winner.
    ///
    /// In order to distinguish how bad the result is, we can add a penalty value,
    /// which represents the "fault" on the result.
    ///
    /// Under most circumstances, the result is not good enough, appearing on its fitness value.
    /// But sometimes a result is badly than our normal results,
    /// if we mark them as the worst one (infinity), it will become a great "wall",
    /// which is not suitable for us to search across it.
    ///
    /// So that, we use secondary evaluation function to measure the result from other requirements,
    /// we call it "constraint" or "penalty function".
    /// The penalty value usually multiply a weight factor for increasing its influence.
    fn fitness(&self, v: &[f64], f: f64) -> Self::Fitness;

    /// Return the final result of the problem.
    ///
    /// The parameters `v` is the best parameter we found.
    fn result(&self, v: &[f64]) -> Self::Result;

    /// Get upper bound.
    fn ub(&self) -> &[f64];

    /// Get lower bound.
    fn lb(&self) -> &[f64];
}
