//! The random function for building algorithm.
use core::{
    mem::size_of,
    sync::atomic::{AtomicU64, Ordering},
};
use getrandom::getrandom;
use oorandom::Rand32;

// Currently not support u128.
static STATE1: AtomicU64 = AtomicU64::new(0);
static STATE2: AtomicU64 = AtomicU64::new(0);

/// Generate a random values between [0., 1.).
pub fn rand() -> f64 {
    let mut gen = if STATE1.load(Ordering::SeqCst) == 0 && STATE2.load(Ordering::SeqCst) == 0 {
        // First time
        let mut buf = [0; size_of::<u64>()];
        getrandom(&mut buf).unwrap();
        Rand32::new(u64::from_le_bytes(buf))
    } else {
        Rand32::from_state((STATE1.load(Ordering::SeqCst), STATE2.load(Ordering::SeqCst)))
    };
    let v = gen.rand_float() as f64;
    let state = gen.state();
    STATE1.store(state.0, Ordering::SeqCst);
    STATE2.store(state.1, Ordering::SeqCst);
    v
}

/// Generate a random boolean by positive (`true`) factor.
#[inline]
pub fn maybe(v: f64) -> bool {
    rand() < v
}

/// Generate a random values by range.
#[inline]
pub fn rand_float(lb: f64, ub: f64) -> f64 {
    rand() * (ub - lb) + lb
}

/// Generate a random values by range.
#[inline]
pub fn rand_int(lb: usize, ub: usize) -> usize {
    (rand() * (ub - lb) as f64) as usize + lb
}

/// Generate a random vector.
pub fn rand_vector(v: &mut [usize], start: usize, lb: usize, ub: usize) {
    for i in start..v.len() {
        v[i] = rand_int(lb, ub);
        while v[..i].contains(&v[i]) {
            v[i] = rand_int(lb, ub);
        }
    }
}
