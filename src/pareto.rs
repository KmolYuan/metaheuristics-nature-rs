//! Pareto front implementation.
use crate::prelude::*;
use alloc::vec::Vec;
use core::iter::zip;

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
    fn update(&mut self, xs: &[f64], fit: &Self::Item);
    /// Update the best elements from a batch.
    fn update_all(&mut self, xs: &[Vec<f64>], fit: &[&Self::Item]) {
        zip(xs, fit).for_each(|(xs, fit)| self.update(xs, fit));
    }
    /// Sample a random best element.
    fn sample(&self, rng: &Rng) -> Option<(&[f64], &Self::Item)>;
    /// Get the final best element.
    fn result(&self) -> Option<(&[f64], &Self::Item)>;
}

impl<T: Dominance> Best for SingleBest<T> {
    type Item = T;

    fn update(&mut self, xs: &[f64], fit: &Self::Item) {
        if let (Some(best), Some(best_f)) = (&mut self.xs, &mut self.fit) {
            if fit.is_dominated(best_f) {
                *best = xs.to_vec();
                *best_f = fit.clone();
            }
        } else {
            self.xs = Some(xs.to_vec());
            self.fit = Some(fit.clone());
        }
    }

    fn sample(&self, _rng: &Rng) -> Option<(&[f64], &Self::Item)> {
        self.result()
    }

    fn result(&self) -> Option<(&[f64], &Self::Item)> {
        self.xs.as_deref().zip(self.fit.as_ref())
    }
}

impl<T: Dominance> Best for Pareto<T> {
    type Item = T;

    fn update(&mut self, xs: &[f64], fit: &Self::Item) {
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
        // Add the new solution
        if is_dominated {
            self.xs.push(xs.to_vec());
            self.fit.push(fit.clone());
        }
        // Prune the solution set
        if self.xs.len() > self.limit {
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

    fn result(&self) -> Option<(&[f64], &Self::Item)> {
        zip(&self.xs, &self.fit)
            .map(|(xs, fit)| (xs, fit, fit.eval()))
            .min_by(|(.., a), (.., b)| a.partial_cmp(b).unwrap())
            .map(|(xs, fit, _)| (xs.as_slice(), fit))
    }
}
