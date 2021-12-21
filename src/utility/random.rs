use core::{
    mem::size_of,
    ops::Range,
    sync::atomic::{AtomicU64, Ordering},
};
use getrandom::getrandom;
use oorandom::Rand64;

union Converter128 {
    n: u128,
    s: [u64; 2],
}

struct AtomicU128 {
    s1: AtomicU64,
    s2: AtomicU64,
}

impl AtomicU128 {
    fn new(v: u128) -> Self {
        let c = Converter128 { n: v };
        Self {
            s1: AtomicU64::new(unsafe { c.s[0] }),
            s2: AtomicU64::new(unsafe { c.s[1] }),
        }
    }

    fn load(&self, order: Ordering) -> u128 {
        let c = Converter128 {
            s: [self.s1.load(order), self.s2.load(order)],
        };
        unsafe { c.n }
    }

    fn store(&self, val: u128, order: Ordering) {
        let c = Converter128 { n: val };
        self.s1.store(unsafe { c.s[0] }, order);
        self.s2.store(unsafe { c.s[1] }, order);
    }
}

/// An uniformed random number generator.
///
/// This generator doesn't require mutability,
/// because the state is saved as atomic values.
pub struct Rng {
    /// Seed of this generator.
    pub seed: u128,
    s1: AtomicU128,
    s2: AtomicU128,
}

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
        Self {
            seed,
            s1: AtomicU128::new(s1),
            s2: AtomicU128::new(s2),
        }
    }

    /// Generate a random values between `0..1` (exclusive).
    pub fn rand(&self) -> f64 {
        let mut rng = Rand64::from_state((
            self.s1.load(Ordering::Relaxed),
            self.s2.load(Ordering::Relaxed),
        ));
        let v = rng.rand_float();
        let (s1, s2) = rng.state();
        self.s1.store(s1, Ordering::Relaxed);
        self.s2.store(s2, Ordering::Relaxed);
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
