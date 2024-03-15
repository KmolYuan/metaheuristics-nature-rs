//! Pareto front implementation.
use crate::random::Rng;
use alloc::vec::Vec;

/// Trait for dominance comparison.
///
/// # Example
///
/// ```
/// use metaheuristics_nature::pareto::Dominance;
///
/// struct MyObject {
///     cost: f64,
///     weight: f64,
/// }
///
/// impl Dominance for MyObject {
///     fn is_dominated(&self, rhs: &Self) -> bool {
///         self.cost <= rhs.cost && self.weight <= rhs.weight
///     }
/// }
/// ```
pub trait Dominance {
    /// Check if `self` dominates `rhs`.
    fn is_dominated(&self, rhs: &Self) -> bool;
    /// Evaluate the final fitness value.
    fn eval(&self) -> f64;
}

impl Dominance for f64 {
    fn is_dominated(&self, rhs: &Self) -> bool {
        self <= rhs
    }
    fn eval(&self) -> f64 {
        *self
    }
}
impl Dominance for f32 {
    fn is_dominated(&self, rhs: &Self) -> bool {
        self <= rhs
    }
    fn eval(&self) -> f64 {
        *self as f64
    }
}

/// Single best element container.
#[derive(Debug, Default)]
pub struct SingleBest<T: Dominance> {
    xs: Option<Vec<f64>>,
    fit: Option<T>,
}

/// Pareto front container.
#[derive(Debug)]
pub struct Pareto<T: Dominance> {
    xs: Vec<Vec<f64>>,
    fit: Vec<T>,
    limit: usize,
}

impl<T: Dominance> Default for Pareto<T> {
    fn default() -> Self {
        Self::new(10)
    }
}

impl<T: Dominance> Pareto<T> {
    /// Create a new Pareto front container.
    pub const fn new(limit: usize) -> Self {
        Self { xs: Vec::new(), fit: Vec::new(), limit }
    }

    /// Set the limit of the Pareto front.
    ///
    /// This method will not change the solution set until the next update.
    pub fn set_limit(&mut self, limit: usize) {
        self.limit = limit;
    }
}

/// A trait for best element container.
pub trait Best {
    /// The type of the best element
    type Item: Dominance;
    /// Update the best element.
    fn update(&mut self, xs: Vec<f64>, fit: Self::Item);
    /// Sample a random best element.
    fn sample(&self, rng: &Rng) -> Option<(&[f64], &Self::Item)>;
}

impl<T: Dominance> Best for SingleBest<T> {
    type Item = T;

    fn update(&mut self, xs: Vec<f64>, fit: Self::Item) {
        if let (Some(best), Some(best_f)) = (&mut self.xs, &mut self.fit) {
            if fit.is_dominated(best_f) {
                *best = xs;
                *best_f = fit;
            }
        } else {
            self.xs = Some(xs);
            self.fit = Some(fit);
        }
    }

    fn sample(&self, _rng: &Rng) -> Option<(&[f64], &Self::Item)> {
        self.xs.as_deref().zip(self.fit.as_ref())
    }
}

impl<T: Dominance> Best for Pareto<T> {
    type Item = T;

    fn update(&mut self, xs: Vec<f64>, fit: Self::Item) {
        let mut is_dominated = false;
        // Remove dominated solutions
        for i in 0..self.xs.len() {
            let dominated = fit.is_dominated(&self.fit[i]);
            is_dominated |= dominated;
            if dominated {
                self.xs.swap_remove(i);
                self.fit.swap_remove(i);
            }
        }
        if is_dominated {
            self.xs.push(xs);
            self.fit.push(fit);
        }
        if self.xs.len() > self.limit {
            // Remove the oldest solution
            let (i, _) = (self.fit.iter().map(T::eval).enumerate())
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();
            self.xs.remove(i);
            self.fit.remove(i);
        }
    }

    fn sample(&self, rng: &Rng) -> Option<(&[f64], &Self::Item)> {
        let i = rng.range(0..self.xs.len());
        Some((self.xs.get(i)?, self.fit.get(i)?))
    }
}
