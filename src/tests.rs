#![doc(hidden)]
use crate::utility::prelude::*;

const OFFSET: f64 = 7.;

#[cfg(test)]
macro_rules! assert_xs {
    ($a:expr, $b:expr) => {
        $a.iter()
            .zip($b)
            .for_each(|(a, b)| assert!((a - b).abs() < f64::EPSILON * 2., "a: {a}, b: {b}"));
    };
}

// An example case for doctest
pub struct TestObj;

impl TestObj {
    pub const fn new() -> Self {
        Self
    }
}

impl Bounded for TestObj {
    fn bound(&self) -> &[[f64; 2]] {
        &[[-50., 50.]; 4]
    }
}

impl ObjFactory for TestObj {
    type Product = f64;
    type Eval = f64;

    fn produce(&self, x: &[f64]) -> Self::Product {
        OFFSET + x[0] * x[0] + 8. * x[1] * x[1] + x[2] * x[2] + x[3] * x[3]
    }

    fn evaluate(&self, product: Self::Product) -> Self::Eval {
        product
    }
}

#[cfg(test)]
fn test<S>() -> Solver<TestObj>
where
    S: Setting + Default,
{
    let mut report = alloc::vec::Vec::new();
    let s = Solver::build(S::default(), TestObj)
        .seed(0)
        .task(|ctx| ctx.best_f - OFFSET < 1e-20)
        .callback(|ctx| report.push(ctx.best_f))
        .solve()
        .unwrap();
    assert!(!report.is_empty());
    assert_eq!(s.result(), OFFSET);
    assert_eq!(s.best_fitness(), s.result());
    s
}

#[test]
fn de() {
    let xs = &[
        -2.0748169104736576e-8,
        -1.6005237632916924e-11,
        9.478282419194714e-9,
        4.742594068439913e-10,
    ];
    assert_xs!(test::<De>().best_parameters(), xs);
}

#[test]
fn pso() {
    let xs = &[
        -1.4795736794868348e-8,
        -3.2916821404040475e-10,
        1.6756327620723574e-8,
        6.68287879514832e-9,
    ];
    assert_xs!(test::<Pso>().best_parameters(), xs);
}

#[test]
fn fa() {
    let xs = &[
        5.902285600635401e-9,
        -1.8476700893832892e-9,
        -1.8298153191604698e-8,
        -1.1359971225137802e-8,
    ];
    assert_xs!(test::<Fa>().best_parameters(), xs);
}

#[test]
fn rga() {
    let xs = &[
        1.4177448066711399e-8,
        -2.0079456682615164e-9,
        -1.0537742668291769e-8,
        1.2656807995916594e-9,
    ];
    assert_xs!(test::<Rga>().best_parameters(), xs);
}

#[test]
fn tlbo() {
    let xs = &[
        -1.3786481627246014e-8,
        -1.830262228970864e-10,
        -1.6444290908967627e-8,
        -1.0078724638383442e-9,
    ];
    assert_xs!(test::<Tlbo>().best_parameters(), xs);
}
