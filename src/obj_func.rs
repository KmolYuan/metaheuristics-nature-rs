use crate::prelude::*;

/// A problem is well bounded.
///
/// Provide constant array reference or dynamic slice for the variables.
pub trait Bounded: Sync + Send {
    /// The upper bound and lower bound in `[[lower, upper]; number_of_vars]`
    /// form.
    ///
    /// This function should be cheap.
    fn bound(&self) -> &[[f64; 2]];

    /// Get the number of variables (dimension) of the problem.
    #[inline]
    fn dim(&self) -> usize {
        self.bound().len()
    }

    /// Get the upper bound and the lower bound values.
    #[inline]
    fn bound_of(&self, s: usize) -> [f64; 2] {
        self.bound()[s]
    }

    ///Get the width of the upper bound and the lower bound.
    fn bound_width(&self, s: usize) -> f64 {
        let [min, max] = self.bound_of(s);
        max - min
    }

    /// Get the upper bound and the lower bound as a range.
    ///
    /// The variable is constrain with lower <= x <= upper.
    fn bound_range(&self, s: usize) -> core::ops::RangeInclusive<f64> {
        let [min, max] = self.bound_of(s);
        min..=max
    }

    /// Get the lower bound.
    #[inline]
    fn lb(&self, s: usize) -> f64 {
        self.bound_of(s)[0]
    }

    /// Get the upper bound.
    #[inline]
    fn ub(&self, s: usize) -> f64 {
        self.bound_of(s)[1]
    }

    /// Check the bounds of the index `s` with the value `v`, and set the value
    /// to max and min if out of bound.
    fn clamp(&self, s: usize, v: f64) -> f64 {
        let [min, max] = self.bound_of(s);
        v.clamp(min, max)
    }
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
///     type Ys = f64;
///
///     fn fitness(&self, x: &[f64]) -> Self::Ys {
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
    /// Type of the fitness value.
    ///
    /// # Wrappers
    ///
    /// There are some wrappers for the fitness value: [`WithProduct`] and
    /// [`MakeSingle`].
    type Ys: Fitness;

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
    fn fitness(&self, xs: &[f64]) -> Self::Ys;
}
