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
    Rng as _, SeedableRng,
};
use rand_chacha::ChaCha8Rng;
use rand_distr::Normal;

/// The seed type of the ChaCha algorithm.
pub type Seed = <ChaCha8Rng as SeedableRng>::Seed;

struct AtomicU128 {
    s1: AtomicU64,
    s2: AtomicU64,
}

impl AtomicU128 {
    fn new(v: u128) -> Self {
        let [a, b] = unsafe { transmute::<_, [u64; 2]>(v) };
        Self { s1: AtomicU64::new(a), s2: AtomicU64::new(b) }
    }

    fn load(&self, order: Ordering) -> u128 {
        unsafe { transmute([self.s1.load(order), self.s2.load(order)]) }
    }

    fn store(&self, v: u128, order: Ordering) {
        let [a, b] = unsafe { transmute::<_, [u64; 2]>(v) };
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
    pub fn new(seed: Option<Seed>) -> Self {
        let rng = match seed {
            Some(seed) => ChaCha8Rng::from_seed(seed),
            None => ChaCha8Rng::from_entropy(),
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

    fn gen<R>(&self, f: impl FnOnce(&mut ChaCha8Rng) -> R) -> R {
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
    pub fn rand_norm<F>(&self, mean: F, std: F) -> F
    where
        F: Float,
        rand_distr::StandardNormal: Distribution<F>,
    {
        self.sample(Normal::new(mean, std).unwrap())
    }

    /// Generate (fill) a random vector.
    ///
    /// The start position of the vector can be set.
    pub fn vector<A, V, R>(&self, mut v: V, start: usize, rng: R) -> V
    where
        A: PartialEq + SampleUniform,
        V: AsMut<[A]>,
        R: SampleRange<A> + Clone,
    {
        let s = v.as_mut();
        for i in start..s.len() {
            s[i] = self.range(rng.clone());
            while s[..i].contains(&s[i]) {
                s[i] = self.range(rng.clone());
            }
        }
        v
    }
}
