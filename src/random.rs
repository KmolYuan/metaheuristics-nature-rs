//! The random function for building algorithm.
use core::{
    mem::size_of,
    ops::Range,
    sync::atomic::{AtomicU64, Ordering},
};
use getrandom::getrandom;
use oorandom::Rand64;

fn concat_u128(v0: u64, v1: u64) -> u128 {
    let mut buf = [0; size_of::<u128>()];
    buf[0..size_of::<u64>()].clone_from_slice(&v0.to_le_bytes());
    buf[size_of::<u64>()..size_of::<u128>()].clone_from_slice(&v1.to_le_bytes());
    u128::from_le_bytes(buf)
}

fn split_u128(v: u128) -> (u64, u64) {
    let buf = v.to_le_bytes();
    let mut buf1 = [0; size_of::<u64>()];
    let mut buf2 = [0; size_of::<u64>()];
    buf1.clone_from_slice(&buf[0..size_of::<u64>()]);
    buf2.clone_from_slice(&buf[size_of::<u64>()..size_of::<u128>()]);
    (u64::from_le_bytes(buf1), u64::from_le_bytes(buf2))
}

/// An uniformed random number generator.
///
/// This generator doesn't require mutability,
/// because the state is saved as atomic values.
pub struct Rng([AtomicU64; 4]);

impl Rng {
    /// Create generator by a given seed.
    /// If none, create the seed from CPU random function.
    pub fn new(seed: Option<u128>) -> Self {
        let seed = match seed {
            Some(seed) => seed,
            None => {
                let mut buf = [0; size_of::<u128>()];
                getrandom(&mut buf).unwrap();
                u128::from_le_bytes(buf)
            }
        };
        let (s1, s2) = Rand64::new(seed).state();
        let (s00, s01) = split_u128(s1);
        let (s10, s11) = split_u128(s2);
        let a = [
            AtomicU64::new(s00),
            AtomicU64::new(s01),
            AtomicU64::new(s10),
            AtomicU64::new(s11),
        ];
        Self(a)
    }

    /// Generate a random values between `0..1` (exclusive).
    pub fn rand(&self) -> f64 {
        let s1 = concat_u128(
            self.0[0].load(Ordering::Relaxed),
            self.0[1].load(Ordering::Relaxed),
        );
        let s2 = concat_u128(
            self.0[2].load(Ordering::Relaxed),
            self.0[3].load(Ordering::Relaxed),
        );
        let mut rng = Rand64::from_state((s1, s2));
        let v = rng.rand_float();
        let (s1, s2) = rng.state();
        let (s00, s01) = split_u128(s1);
        let (s10, s11) = split_u128(s2);
        self.0[0].store(s00, Ordering::Relaxed);
        self.0[1].store(s01, Ordering::Relaxed);
        self.0[2].store(s10, Ordering::Relaxed);
        self.0[3].store(s11, Ordering::Relaxed);
        v
    }

    /// Generate a random boolean by positive (`true`) factor.
    pub fn maybe(&self, v: f64) -> bool {
        self.rand() < v
    }

    /// Generate a random floating point value by range.
    pub fn float(&self, rng: Range<f64>) -> f64 {
        self.rand() * (rng.end - rng.start) + rng.start
    }

    /// Generate a random integer value by range.
    pub fn int(&self, rng: Range<usize>) -> usize {
        (self.rand() * (rng.end - rng.start) as f64) as usize + rng.start
    }

    /// Generate (fill) a random vector.
    ///
    /// The start position of the vector can be set.
    pub fn vector(&self, v: &mut [usize], start: usize, rng: Range<usize>) {
        for i in start..v.len() {
            v[i] = self.int(rng.clone());
            while v[..i].contains(&v[i]) {
                v[i] = self.int(rng.clone());
            }
        }
    }
}
