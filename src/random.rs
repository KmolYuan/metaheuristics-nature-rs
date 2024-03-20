//! Random number generator module.
use alloc::vec::Vec;
use rand::{
    distributions::{
        uniform::{SampleRange, SampleUniform},
        Distribution, Standard,
    },
    Rng as _, SeedableRng as _,
};
use rand_chacha::ChaCha20Rng as ChaCha;

/// The seed type of the ChaCha algorithm.
pub type Seed = [u8; 32];

/// The seed option.
#[derive(Copy, Clone)]
pub enum SeedOpt {
    /// Seed from non-crypto u64
    U64(u64),
    /// Crypto seed series
    Seed(Seed),
    /// Auto-decided crypto seed
    None,
}

impl From<Option<u64>> for SeedOpt {
    fn from(opt: Option<u64>) -> Self {
        match opt {
            Some(seed) => Self::U64(seed),
            None => Self::None,
        }
    }
}

impl From<u64> for SeedOpt {
    fn from(seed: u64) -> Self {
        Self::U64(seed)
    }
}

impl From<Seed> for SeedOpt {
    fn from(seed: Seed) -> Self {
        Self::Seed(seed)
    }
}

/// An uniformed random number generator.
#[derive(Clone, Debug)]
pub struct Rng {
    rng: ChaCha,
}

impl Rng {
    /// Create generator by a given seed.
    /// If none, create the seed from CPU random function.
    pub fn new(seed: SeedOpt) -> Self {
        let rng = match seed {
            SeedOpt::Seed(seed) => ChaCha::from_seed(seed),
            SeedOpt::U64(seed) => ChaCha::seed_from_u64(seed),
            SeedOpt::None => ChaCha::from_entropy(),
        };
        Self { rng }
    }

    /// Seed of this generator.
    #[inline]
    pub fn seed(&self) -> Seed {
        self.rng.get_seed()
    }

    /// Stream for parallel threading.
    ///
    /// Use the iterators `.zip()` method to fork this RNG set.
    pub fn stream(&mut self, n: usize) -> Vec<Self> {
        // Needs to "run" the RNG to avoid constantly opening new branches
        const JUMP: u128 = 2 << 64;
        self.rng
            .set_word_pos(self.rng.get_word_pos().wrapping_add(JUMP));
        let stream = self.rng.get_stream();
        (1..=n)
            .map(|i| {
                let mut rng = self.clone();
                rng.rng.set_stream(stream.wrapping_add(i as _));
                rng
            })
            .collect()
    }

    /// A low-level access to the RNG type.
    ///
    /// Please import necessary traits first.
    pub fn gen_with<R>(&mut self, f: impl FnOnce(&mut ChaCha) -> R) -> R {
        f(&mut self.rng)
    }

    /// Generate a random value by standard distribution.
    pub fn gen<T>(&mut self) -> T
    where
        Standard: Distribution<T>,
    {
        self.gen_with(|r| r.gen())
    }

    /// Generate a classic random value between `0..1` (exclusive range).
    #[inline]
    pub fn rand(&mut self) -> f64 {
        self.ub(1.)
    }

    /// Generate a random boolean by positive (`true`) factor.
    #[inline]
    pub fn maybe(&mut self, p: f64) -> bool {
        self.gen_with(|r| r.gen_bool(p))
    }

    /// Generate a random value by range.
    #[inline]
    pub fn range<T, R>(&mut self, range: R) -> T
    where
        T: SampleUniform,
        R: SampleRange<T>,
    {
        self.gen_with(|r| r.gen_range(range))
    }

    /// Sample from a distribution.
    #[inline]
    pub fn sample<T, D>(&mut self, distr: D) -> T
    where
        D: Distribution<T>,
    {
        self.gen_with(|r| r.sample(distr))
    }

    /// Generate a random value by upper bound (exclusive range).
    ///
    /// The lower bound is zero.
    #[inline]
    pub fn ub<U>(&mut self, ub: U) -> U
    where
        U: Default + SampleUniform,
        core::ops::Range<U>: SampleRange<U>,
    {
        self.range(U::default()..ub)
    }

    /// Generate a random value by range.
    #[inline]
    pub fn clamp<T, R>(&mut self, v: T, range: R) -> T
    where
        T: SampleUniform + PartialOrd,
        R: SampleRange<T> + core::ops::RangeBounds<T>,
    {
        if range.contains(&v) {
            v
        } else {
            self.range(range)
        }
    }

    /// Sample with Gaussian distribution.
    #[inline]
    pub fn normal<F>(&mut self, mean: F, std: F) -> F
    where
        F: num_traits::Float,
        rand_distr::StandardNormal: Distribution<F>,
    {
        self.sample(rand_distr::Normal::new(mean, std).unwrap())
    }

    /// Shuffle a slice.
    pub fn shuffle<S: rand::seq::SliceRandom + ?Sized>(&mut self, s: &mut S) {
        self.gen_with(|r| s.shuffle(r));
    }

    /// Generate a random array with no-repeat values.
    pub fn array<A, C, const N: usize>(&mut self, candi: C) -> [A; N]
    where
        A: Default + Copy + PartialEq + SampleUniform,
        C: IntoIterator<Item = A>,
    {
        let mut candi = candi.into_iter().collect::<Vec<_>>();
        self.shuffle(candi.as_mut_slice());
        candi[..N].try_into().unwrap()
    }
}
