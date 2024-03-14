//! Pareto front implementation.
use crate::random::Rng;

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
}

impl<T: PartialOrd> Dominance for T {
    fn is_dominated(&self, rhs: &Self) -> bool {
        self <= rhs
    }
}

/// Single best element container.
#[derive(Debug, Default)]
pub struct SingleBest<T: Dominance> {
    best: Option<T>,
}

/// Pareto front container.
#[derive(Debug, Default)]
pub struct Pareto<T: Dominance> {
    list: Vec<T>,
}

/// A trait for best element container.
pub trait Best {
    /// The type of the best element
    type Item: Dominance;
    /// Update the best element.
    fn update(&mut self, x: Self::Item);
    /// Sample a random best element.
    fn sample(&self, rng: &Rng) -> Option<&Self::Item>;
}

impl<T: Dominance> Best for SingleBest<T> {
    type Item = T;

    fn update(&mut self, x: Self::Item) {
        if let Some(best) = &mut self.best {
            if x.is_dominated(best) {
                *best = x;
            }
        } else {
            self.best = Some(x);
        }
    }

    fn sample(&self, _rng: &Rng) -> Option<&Self::Item> {
        self.best.as_ref()
    }
}

impl<T: Dominance> Best for Pareto<T> {
    type Item = T;

    fn update(&mut self, x: Self::Item) {
        let mut is_dominated = false;
        self.list.retain(|y| {
            let dominated = x.is_dominated(y);
            is_dominated |= dominated;
            !dominated
        });
        if is_dominated {
            self.list.push(x);
        }
    }

    fn sample(&self, rng: &Rng) -> Option<&Self::Item> {
        use rand::seq::SliceRandom as _;
        rng.gen(|r| self.list.choose(r))
    }
}
