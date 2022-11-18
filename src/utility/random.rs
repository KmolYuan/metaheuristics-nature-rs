use alloc::vec::Vec;
use core::{
    mem::transmute,
    sync::atomic::{AtomicU64, Ordering},
};
use num_traits::{Float, Zero};
use rand::{
    distributions::{
        uniform::{SampleRange, SampleUniform},
        Distribution,
    },
    seq::SliceRandom as _,
    Rng as _, SeedableRng as _,
};
use rand_chacha::ChaCha8Rng;

/// The seed type of the ChaCha algorithm.
pub type Seed = [u8; 32];

/// The seed option.
pub enum SeedOption {
    /// Seed from non-crypto u64
    U64(u64),
    /// Crypto seed series
    Seed(Seed),
    /// Auto-decided crypto seed
    None,
}

impl From<Option<u64>> for SeedOption {
    fn from(opt: Option<u64>) -> Self {
        match opt {
            Some(seed) => Self::U64(seed),
            None => Self::None,
        }
    }
}

impl From<u64> for SeedOption {
    fn from(seed: u64) -> Self {
        Self::U64(seed)
    }
}

impl From<Seed> for SeedOption {
    fn from(seed: Seed) -> Self {
        Self::Seed(seed)
    }
}

struct AtomicU128 {
    s1: AtomicU64,
    s2: AtomicU64,
}

impl AtomicU128 {
    fn new(v: u128) -> Self {
        let [a, b] = unsafe { transmute::<_, [_; 2]>(v) };
        Self { s1: AtomicU64::new(a), s2: AtomicU64::new(b) }
    }

    fn load(&self, order: Ordering) -> u128 {
        unsafe { transmute([self.s1.load(order), self.s2.load(order)]) }
    }

    fn store(&self, v: u128, order: Ordering) {
        let [a, b] = unsafe { transmute::<_, [_; 2]>(v) };
        self.s1.store(a, order);
        self.s2.store(b, order);
    }
}

/// An uniformed random number generator.
///
/// This generator doesn't require mutability,
/// because the state is saved as atomic values.
pub struct Rng {
    seed: Seed,
    stream: AtomicU64,
    word_pos: AtomicU128,
}

impl Rng {
    /// Create generator by a given seed.
    /// If none, create the seed from CPU random function.
    pub fn new(seed: SeedOption) -> Self {
        let rng = match seed {
            SeedOption::U64(seed) => ChaCha8Rng::seed_from_u64(seed),
            SeedOption::Seed(seed) => ChaCha8Rng::from_seed(seed),
            SeedOption::None => ChaCha8Rng::from_entropy(),
        };
        Self {
            seed: rng.get_seed(),
            stream: AtomicU64::new(rng.get_stream()),
            word_pos: AtomicU128::new(rng.get_word_pos()),
        }
    }

    /// Seed of this generator.
    pub fn seed(&self) -> Seed {
        self.seed
    }

    /// Low-level access to the RNG type.
    ///
    /// Please import necessary traits first.
    pub fn gen<R>(&self, f: impl FnOnce(&mut ChaCha8Rng) -> R) -> R {
        let mut rng = ChaCha8Rng::from_seed(self.seed);
        rng.set_stream(self.stream.load(Ordering::SeqCst));
        rng.set_word_pos(self.word_pos.load(Ordering::SeqCst));
        let r = f(&mut rng);
        self.stream.store(rng.get_stream(), Ordering::SeqCst);
        self.word_pos.store(rng.get_word_pos(), Ordering::SeqCst);
        r
    }

    /// Generate a classic random value between `0..1` (exclusive range).
    #[inline]
    pub fn rand(&self) -> f64 {
        self.ub(1.)
    }

    /// Generate a random boolean by positive (`true`) factor.
    #[inline]
    pub fn maybe(&self, p: f64) -> bool {
        self.gen(|r| r.gen_bool(p))
    }

    /// Generate a random value by range.
    #[inline]
    pub fn range<T, R>(&self, range: R) -> T
    where
        T: SampleUniform,
        R: SampleRange<T>,
    {
        self.gen(|r| r.gen_range(range))
    }

    /// Sample from a distribution.
    #[inline]
    pub fn sample<T, D>(&self, distr: D) -> T
    where
        D: Distribution<T>,
    {
        self.gen(|r| r.sample(distr))
    }

    /// Generate a random value by upper bound (exclusive range).
    ///
    /// The lower bound is zero.
    #[inline]
    pub fn ub<U>(&self, ub: U) -> U
    where
        U: Zero + SampleUniform,
        core::ops::Range<U>: SampleRange<U>,
    {
        self.range(U::zero()..ub)
    }

    /// Sample with Gaussian distribution.
    #[inline]
    pub fn normal<F>(&self, mean: F, std: F) -> F
    where
        F: Float,
        rand_distr::StandardNormal: Distribution<F>,
    {
        self.sample(rand_distr::Normal::new(mean, std).unwrap())
    }

    /// Shuffle a slice.
    pub fn shuffle<A>(&self, s: &mut [A]) {
        self.gen(|r| s.shuffle(r));
    }

    /// Generate a random array with no-repeat values.
    pub fn array<A, C, const N: usize>(&self, candi: C) -> [A; N]
    where
        A: Zero + Copy + PartialEq + SampleUniform,
        C: IntoIterator<Item = A>,
    {
        self.array_by([A::zero(); N], 0, candi)
    }

    /// Fill a mutable slice with no-repeat values.
    ///
    /// The start position of the vector can be set.
    pub fn array_by<A, V, C>(&self, mut v: V, start: usize, candi: C) -> V
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
        self.shuffle(&mut candi);
        curr.iter_mut().zip(candi).for_each(|(a, b)| *a = b);
        v
    }
}
