//! Random number generator module.
use alloc::vec::Vec;
use rand::{
    distributions::{
        uniform::{SampleRange, SampleUniform},
        Distribution,
    },
    Rng as _, SeedableRng as _,
};
use rand_chacha::ChaCha8Rng;

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
#[derive(Clone)]
pub struct Rng {
    rng: ChaCha8Rng,
}

impl Rng {
    /// Create generator by a given seed.
    /// If none, create the seed from CPU random function.
    pub fn new(seed: SeedOpt) -> Self {
        let rng = match seed {
            SeedOpt::Seed(seed) => ChaCha8Rng::from_seed(seed),
            SeedOpt::U64(seed) => ChaCha8Rng::seed_from_u64(seed),
            SeedOpt::None => ChaCha8Rng::from_entropy(),
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
        let stream = self.rng.get_stream();
        self.rng.set_stream(stream.wrapping_add(n as u64));
        (0..n)
            .map(|i| {
                let mut rng = self.clone();
                rng.rng.set_stream(stream.wrapping_add(i as u64));
                rng
            })
            .collect()
    }

    /// Low-level access to the RNG type.
    ///
    /// Please import necessary traits first.
    pub fn gen<R>(&mut self, f: impl FnOnce(&mut ChaCha8Rng) -> R) -> R {
        f(&mut self.rng)
    }

    /// Generate a classic random value between `0..1` (exclusive range).
    #[inline]
    pub fn rand(&mut self) -> f64 {
        self.ub(1.)
    }

    /// Generate a random boolean by positive (`true`) factor.
    #[inline]
    pub fn maybe(&mut self, p: f64) -> bool {
        self.rng.gen_bool(p)
    }

    /// Generate a random value by range.
    #[inline]
    pub fn range<T, R>(&mut self, range: R) -> T
    where
        T: SampleUniform,
        R: SampleRange<T>,
    {
        self.rng.gen_range(range)
    }

    /// Sample from a distribution.
    #[inline]
    pub fn sample<T, D>(&mut self, distr: D) -> T
    where
        D: Distribution<T>,
    {
        self.rng.sample(distr)
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
        s.shuffle(&mut self.rng)
    }

    /// Generate a random array with no-repeat values.
    pub fn array<A, C, const N: usize>(&mut self, candi: C) -> [A; N]
    where
        A: Default + Copy + PartialEq + SampleUniform,
        C: IntoIterator<Item = A>,
    {
        self.array_by([A::default(); N], 0, candi)
    }

    /// Fill a mutable slice with no-repeat values.
    ///
    /// The start position of the vector can be set.
    pub fn array_by<A, V, C>(&mut self, mut v: V, start: usize, candi: C) -> V
    where
        A: PartialEq + SampleUniform,
        V: AsMut<[A]>,
        C: IntoIterator<Item = A>,
    {
        let (pre, curr) = v.as_mut().split_at_mut(start);
        let mut candi = candi
            .into_iter()
            .filter(|e| !pre.contains(e))
            .collect::<Vec<_>>();
        self.shuffle(candi.as_mut_slice());
        core::iter::zip(curr, candi).for_each(|(a, b)| *a = b);
        v
    }
}
