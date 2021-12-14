//! The random function for building algorithm.
use core::{mem::size_of, ops::Range};
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

    /// Generate a random floating point value by range.
    pub fn float(&mut self, rng: Range<f64>) -> f64 {
        self.rand() * (rng.end - rng.start) + rng.start
    }

    /// Generate a random integer value by range.
    pub fn int(&mut self, rng: Range<usize>) -> usize {
        (self.rand() * (rng.end - rng.start) as f64) as usize + rng.start
    }

    /// Generate (fill) a random vector.
    ///
    /// The start position of the vector can be set.
    pub fn vector(&mut self, v: &mut [usize], start: usize, rng: Range<usize>) {
        for i in start..v.len() {
            v[i] = self.int(rng.clone());
            while v[..i].contains(&v[i]) {
                v[i] = self.int(rng.clone());
            }
        }
    }
}
