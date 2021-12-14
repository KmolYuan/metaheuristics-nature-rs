//! The random function for building algorithm.
use core::mem::size_of;
use getrandom::getrandom;
use oorandom::Rand64;

/// An uniformed random number generator.
pub struct Rng(Rand64);

impl Rng {
    /// Create generator by a given seed.
    /// If none, create the seed from CPU random function.
    pub fn new(seed: Option<u128>) -> Self {
        match seed {
            Some(seed) => Self(Rand64::new(seed)),
            None => {
                let mut buf = [0; size_of::<u128>()];
                getrandom(&mut buf).unwrap();
                Self(Rand64::new(u128::from_le_bytes(buf)))
            }
        }
    }

    /// Generate a random values between `0..1` (exclusive).
    pub fn rand(&mut self) -> f64 {
        self.0.rand_float()
    }

    /// Generate a random boolean by positive (`true`) factor.
    pub fn maybe(&mut self, v: f64) -> bool {
        self.rand() < v
    }

    /// Generate a random values by range.
    pub fn rand_float(&mut self, lb: f64, ub: f64) -> f64 {
        self.rand() * (ub - lb) + lb
    }

    /// Generate a random values by range.
    pub fn rand_int(&mut self, lb: usize, ub: usize) -> usize {
        (self.rand() * (ub - lb) as f64) as usize + lb
    }

    /// Generate (fill) a random vector.
    pub fn rand_vector(&mut self, v: &mut [usize], start: usize, lb: usize, ub: usize) {
        for i in start..v.len() {
            v[i] = self.rand_int(lb, ub);
            while v[..i].contains(&v[i]) {
                v[i] = self.rand_int(lb, ub);
            }
        }
    }
}
