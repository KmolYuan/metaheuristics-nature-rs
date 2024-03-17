//! Pareto front implementation.
use crate::prelude::*;
use alloc::vec::Vec;
use core::iter::zip;

/// Single best element container.
#[derive(Debug)]
pub struct SingleBest<T: Fitness> {
    xs: Option<Vec<f64>>,
    fit: Option<T>,
}

impl<T: Fitness> SingleBest<T> {
    /// Get the final best element.
    pub fn current_eval(&self) -> <T as Fitness>::Eval {
        Best::current_eval(self)
    }
}

/// Pareto front container.
#[derive(Debug)]
pub struct Pareto<T: Fitness> {
    xs: Vec<Vec<f64>>,
    fit: Vec<T>,
    limit: usize,
}

impl<T: Fitness> Pareto<T> {
    /// Get the final best element.
    pub fn current_eval(&self) -> <T as Fitness>::Eval {
        Best::current_eval(self)
    }
}

/// A trait for best element container.
pub trait Best: MaybeParallel {
    /// The type of the best element
    type Item: Fitness;
    /// Create a new best element container.
    fn from_limit(limit: usize) -> Self;
    /// Update the best element.
    fn update(&mut self, xs: &[f64], fit: &Self::Item);
    /// Update the best elements from a batch.
    fn update_all(&mut self, pool: &[Vec<f64>], pool_f: &[Self::Item]) {
        zip(pool, pool_f).for_each(|(xs, fit)| self.update(xs, fit));
    }
    /// Sample a random best element.
    fn sample(&self, rng: &Rng) -> (&[f64], &Self::Item);
    /// Sample a random design variables.
    ///
    /// # Panics
    ///
    /// Panics if the best element is not available.
    fn sample_xs(&self, rng: &Rng) -> &[f64] {
        self.sample(rng).0
    }
    /// Get the final best element.
    fn as_result(&self) -> (&[f64], &Self::Item);
    /// Get the final best fitness value.
    fn as_result_fit(&self) -> &Self::Item {
        self.as_result().1
    }
    /// Get the final best evaluation value.
    fn current_eval(&self) -> <Self::Item as Fitness>::Eval {
        self.as_result_fit().eval()
    }
}

impl<T: Fitness> Best for SingleBest<T> {
    type Item = T;

    fn from_limit(_limit: usize) -> Self {
        Self { xs: None, fit: None }
    }

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

    fn sample(&self, _rng: &Rng) -> (&[f64], &Self::Item) {
        self.as_result()
    }

    fn as_result(&self) -> (&[f64], &Self::Item) {
        match (&self.xs, &self.fit) {
            (Some(xs), Some(fit)) => (xs, fit),
            _ => panic!("No best element available"),
        }
    }
}

impl<T: Fitness> Best for Pareto<T> {
    type Item = T;

    fn from_limit(limit: usize) -> Self {
        let xs = Vec::with_capacity(limit);
        let fit = Vec::with_capacity(limit);
        Self { xs, fit, limit }
    }

    fn update(&mut self, xs: &[f64], fit: &Self::Item) {
        let mut is_dominated = self.xs.is_empty();
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

    fn sample(&self, rng: &Rng) -> (&[f64], &Self::Item) {
        let i = rng.range(0..self.xs.len());
        (&self.xs[i], &self.fit[i])
    }

    fn as_result(&self) -> (&[f64], &Self::Item) {
        match zip(&self.xs, &self.fit)
            .map(|(xs, fit)| (xs, fit, fit.eval()))
            .min_by(|(.., a), (.., b)| a.partial_cmp(b).unwrap())
        {
            Some((xs, fit, _)) => (xs, fit),
            None => panic!("No best element available"),
        }
    }
}
