use ndarray::{ArrayView1, AsArray};

/// The base of the objective function.
///
/// For example:
/// ```
/// use metaheuristics_nature::ObjFunc;
/// use ndarray::{AsArray, ArrayView1, Array1};
/// struct MyFunc(Array1<f64>, Array1<f64>);
/// impl MyFunc {
///     fn new() -> Self { Self(Array1::zeros(3), Array1::ones(3) * 50.) }
/// }
/// impl ObjFunc for MyFunc {
///     type Result = f64;
///     fn fitness<'a, A>(&self, gen: u32, v: A) -> f64
///     where
///         A: AsArray<'a, f64>,
///     {
///         let v = v.into();
///         v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
///     }
///     fn result<'a, A>(&self, v: A) -> Self::Result
///     where
///         A: AsArray<'a, f64>
///     {
///         self.fitness(0, v)
///     }
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
    /// Return fitness, the smaller value represents good.
    fn fitness<'a, A>(&self, gen: u32, v: A) -> f64
    where
        A: AsArray<'a, f64>;
    /// Return the final result of the problem.
    fn result<'a, A>(&self, v: A) -> Self::Result
    where
        A: AsArray<'a, f64>;
    /// Get upper bound.
    fn ub(&self) -> ArrayView1<f64>;
    /// Get lower bound.
    fn lb(&self) -> ArrayView1<f64>;
}
