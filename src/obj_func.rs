use ndarray::{ArrayView1, AsArray};

/// The base of the objective function.
///
/// For example:
/// ```
/// use metaheuristics_nature::ObjFunc;
/// use ndarray::{AsArray, ArrayView1, Array1};
///
/// struct MyFunc(Array1<f64>, Array1<f64>);
///
/// impl MyFunc {
///     fn new() -> Self { Self(Array1::zeros(3), Array1::ones(3) * 50.) }
/// }
///
/// impl ObjFunc for MyFunc {
///     type Result = f64;
///
///     fn fitness<'a, A>(&self, _gen: u32, v: A) -> f64
///     where
///         A: AsArray<'a, f64>,
///     {
///         let v = v.into();
///         v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
///     }
///
///     fn result<'a, V>(&self, v: V) -> Self::Result
///     where
///         V: AsArray<'a, f64>
///     {
///         self.fitness(0, v)
///     }
///
///     fn ub(&self) -> ArrayView1<f64> { self.1.view() }
///     fn lb(&self) -> ArrayView1<f64> { self.0.view() }
/// }
/// ```
/// The objective function returns fitness value that used to evaluate the objective.
///
/// The lower bound and upper bound represents the number of variables at the same time.
///
/// This trait is designed as immutable.
pub trait ObjFunc {
    /// The result type.
    type Result;

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
    fn fitness<'a, A>(&self, gen: u32, v: A) -> f64
    where
        A: AsArray<'a, f64>;

    /// Return the final result of the problem.
    fn result<'a, V>(&self, v: V) -> Self::Result
    where
        V: AsArray<'a, f64>;

    /// Get upper bound.
    fn ub(&self) -> ArrayView1<f64>;

    /// Get lower bound.
    fn lb(&self) -> ArrayView1<f64>;
}
