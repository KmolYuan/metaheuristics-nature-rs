use self::ziggurat::{ZIG_NORM_F, ZIG_NORM_R, ZIG_NORM_X};
use core::{
    mem::size_of,
    ops::Range,
    sync::atomic::{AtomicU64, Ordering},
};
use getrandom::getrandom;
use oorandom::Rand64;

mod ziggurat;

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
        let c = Converter128 { s: [self.s1.load(order), self.s2.load(order)] };
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
    seed: u128,
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

    /// Seed of this generator.
    pub fn seed(&self) -> u128 {
        self.seed
    }

    #[inline]
    fn gen<R>(&self, f: impl Fn(&mut Rand64) -> R) -> R {
        let mut rng = Rand64::from_state((
            self.s1.load(Ordering::Relaxed),
            self.s2.load(Ordering::Relaxed),
        ));
        let v = f(&mut rng);
        let (s1, s2) = rng.state();
        self.s1.store(s1, Ordering::Relaxed);
        self.s2.store(s2, Ordering::Relaxed);
        v
    }

    /// Generate a random values between `0..1` (exclusive).
    #[inline]
    pub fn rand(&self) -> f64 {
        self.gen(|rng| rng.rand_float())
    }

    /// Generate a random boolean by positive (`true`) factor.
    #[inline]
    pub fn maybe(&self, v: f64) -> bool {
        self.rand() < v
    }

    /// Generate a random floating point value by range.
    #[inline]
    pub fn float(&self, range: Range<f64>) -> f64 {
        self.rand() * (range.end - range.start) + range.start
    }

    /// Generate a random integer value by range.
    #[inline]
    pub fn int(&self, range: Range<usize>) -> usize {
        (self.rand() * (range.end - range.start) as f64) as usize + range.start
    }

    /// Sample with Gaussian distribution.
    #[inline]
    pub fn rand_norm(&self, mean: f64, std: f64) -> f64 {
        self.gen(|rng| mean + std * ziggurat(rng))
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

// Ziggurat algorithm, copy from `rand`
fn ziggurat(rng: &mut Rand64) -> f64 {
    loop {
        let bits = rng.rand_u64();
        let i = bits as usize & 0xff;

        let u = into_float_with_exponent(bits >> 12, 1) - 3.;
        let x = u * ZIG_NORM_X[i];
        #[cfg(all(feature = "std", not(feature = "libm")))]
        let test_x = x.abs();
        #[cfg(feature = "libm")]
        let test_x = libm::fabs(x);

        // algebraically equivalent to |u| < x_tab[i+1]/x_tab[i] (or u < x_tab[i+1]/x_tab[i])
        if test_x < ZIG_NORM_X[i + 1] {
            return x;
        }
        if i == 0 {
            return zero_case(rng, u);
        }
        #[cfg(all(feature = "std", not(feature = "libm")))]
        let pdf = (-x * x * 0.5).exp();
        #[cfg(feature = "libm")]
        let pdf = libm::exp(-x * x * 0.5);
        // algebraically equivalent to f1 + DRanU()*(f0 - f1) < 1
        if ZIG_NORM_F[i + 1] + (ZIG_NORM_F[i] - ZIG_NORM_F[i + 1]) * rand_f64(rng) < pdf {
            return x;
        }
    }
}

fn rand_f64(rng: &mut Rand64) -> f64 {
    let mut u = rng.rand_u64();
    u >>= 64 - f64::MANTISSA_DIGITS + 1;
    u as f64
}

fn rand_open01(rng: &mut Rand64) -> f64 {
    let float_size = size_of::<f64>() as u32 * 8;
    let value = rng.rand_u64();
    let fraction = value >> (float_size - (f64::MANTISSA_DIGITS - 1));
    into_float_with_exponent(fraction, 0) - (1. - f64::EPSILON * 0.5)
}

fn into_float_with_exponent(i: u64, exponent: i32) -> f64 {
    let exponent_bits = ((1023 + exponent) as u64) << (f64::MANTISSA_DIGITS - 1);
    f64::from_bits(i | exponent_bits)
}

fn zero_case(rng: &mut Rand64, u: f64) -> f64 {
    let mut x = 1.;
    let mut y = 0.;
    while -2. * y < x * x {
        #[cfg(all(feature = "std", not(feature = "libm")))]
        let x_ = rand_open01(rng).ln();
        #[cfg(feature = "libm")]
        let x_ = libm::log(rand_open01(rng));
        #[cfg(all(feature = "std", not(feature = "libm")))]
        let y_ = rand_open01(rng).ln();
        #[cfg(feature = "libm")]
        let y_ = libm::log(rand_open01(rng));
        x = x_ / ZIG_NORM_R;
        y = y_;
    }
    if u < 0. {
        x - ZIG_NORM_R
    } else {
        ZIG_NORM_R - x
    }
}
