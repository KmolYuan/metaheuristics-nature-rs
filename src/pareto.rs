//! Single/Multi-objective best containers.
use crate::prelude::*;
use alloc::vec::Vec;
use core::iter::zip;

/// Single best element container.
#[derive(Debug)]
pub struct SingleBest<T: Fitness> {
    xs: Option<Vec<f64>>,
    ys: Option<T>,
}

impl<T: Fitness> SingleBest<T> {
    /// Get the final best element.
    pub fn get_eval(&self) -> <T as Fitness>::Eval {
        Best::get_eval(self)
    }
}

/// Pareto front container for multi-objective optimization.
#[derive(Debug)]
pub struct Pareto<T: Fitness> {
    xs: Vec<Vec<f64>>,
    ys: Vec<T>,
    limit: usize,
}

impl<T: Fitness> Pareto<T> {
    /// Get the final best element.
    pub fn get_eval(&self) -> <T as Fitness>::Eval {
        Best::get_eval(self)
    }

    /// Get the number of best elements.
    pub fn len(&self) -> usize {
        self.xs.len()
    }

    /// Check if the best elements are empty.
    pub fn is_empty(&self) -> bool {
        self.xs.is_empty()
    }
}

/// A trait for best element container.
pub trait Best: MaybeParallel {
    /// The type of the best element
    type Item: Fitness;
    /// Create a new best element container.
    fn from_limit(limit: usize) -> Self;
    /// Update the best element.
    fn update(&mut self, xs: &[f64], ys: &Self::Item);
    /// Update the best elements from a batch.
    fn update_all(&mut self, pool: &[Vec<f64>], pool_y: &[Self::Item]) {
        zip(pool, pool_y).for_each(|(xs, ys)| self.update(xs, ys));
    }
    /// Sample a random best element.
    fn sample(&self, rng: &mut Rng) -> (&[f64], &Self::Item);
    /// Sample a random design variables.
    ///
    /// # Panics
    ///
    /// Panics if the best element is not available.
    fn sample_xs(&self, rng: &mut Rng) -> &[f64] {
        self.sample(rng).0
    }
    /// Get the final best element.
    fn as_result(&self) -> (&[f64], &Self::Item);
    /// Get the final best fitness value.
    fn as_result_fit(&self) -> &Self::Item {
        self.as_result().1
    }
    /// Convert the best element into the target item.
    ///
    /// See also [`Best::as_result_fit()`] for getting its reference.
    fn into_result_fit(self) -> Self::Item;
    /// Get the final best evaluation value.
    fn get_eval(&self) -> <Self::Item as Fitness>::Eval {
        self.as_result_fit().eval()
    }
}

impl<T: Fitness> Best for SingleBest<T> {
    type Item = T;

    fn from_limit(_limit: usize) -> Self {
        Self { xs: None, ys: None }
    }

    fn update(&mut self, xs: &[f64], ys: &Self::Item) {
        if let (Some(best), Some(best_f)) = (&mut self.xs, &mut self.ys) {
            if ys.is_dominated(best_f) {
                *best = xs.to_vec();
                *best_f = ys.clone();
            }
        } else {
            self.xs = Some(xs.to_vec());
            self.ys = Some(ys.clone());
        }
    }

    fn sample(&self, _rng: &mut Rng) -> (&[f64], &Self::Item) {
        self.as_result()
    }

    fn as_result(&self) -> (&[f64], &Self::Item) {
        match (&self.xs, &self.ys) {
            (Some(xs), Some(ys)) => (xs, ys),
            _ => panic!("No best element available"),
        }
    }

    fn into_result_fit(self) -> Self::Item {
        self.ys.unwrap()
    }
}

impl<T: Fitness> Best for Pareto<T> {
    type Item = T;

    fn from_limit(limit: usize) -> Self {
        let xs = Vec::with_capacity(limit);
        let ys = Vec::with_capacity(limit);
        Self { xs, ys, limit }
    }

    fn update(&mut self, xs: &[f64], ys: &Self::Item) {
        // Remove dominated solutions
        let mut has_dominated = false;
        for i in (0..self.xs.len()).rev() {
            let ys_curr = &self.ys[i];
            if ys.is_dominated(ys_curr) {
                has_dominated = true;
                self.xs.swap_remove(i);
                self.ys.swap_remove(i);
            } else if !has_dominated && ys_curr.is_dominated(ys) {
                return;
            }
        }
        // Add the new solution
        self.xs.push(xs.to_vec());
        self.ys.push(ys.clone());
        // Prune the solution set
        if self.xs.len() > self.limit {
            let (i, _) = (self.ys.iter().map(T::eval).enumerate())
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();
            self.xs.swap_remove(i);
            self.ys.swap_remove(i);
        }
    }

    fn sample(&self, rng: &mut Rng) -> (&[f64], &Self::Item) {
        let i = rng.range(0..self.xs.len());
        (&self.xs[i], &self.ys[i])
    }

    fn as_result(&self) -> (&[f64], &Self::Item) {
        match zip(&self.xs, &self.ys)
            .map(|(xs, ys)| (xs, ys, ys.eval()))
            .min_by(|(.., a), (.., b)| a.partial_cmp(b).unwrap())
        {
            Some((xs, ys, _)) => (xs, ys),
            None => panic!("No best element available"),
        }
    }

    fn into_result_fit(self) -> Self::Item {
        match (self.ys.into_iter())
            .map(|ys| (ys.eval(), ys))
            .min_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap())
        {
            Some((_, ys)) => ys,
            None => panic!("No best element available"),
        }
    }
}
