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

impl ObjFunc for TestObj {
    type Fitness = Product<f64, f64>;

    fn fitness(&self, xs: &[f64]) -> Self::Fitness {
        let y = OFFSET + xs[0] * xs[0] + 8. * xs[1] * xs[1] + xs[2] * xs[2] + xs[3] * xs[3];
        Product::new(y, y)
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
        .task(|ctx| ctx.best_f.fitness() - OFFSET < 1e-20)
        .callback(|ctx| report.push(ctx.best_f.clone()))
        .solve()
        .unwrap();
    assert!(!report.is_empty());
    assert_eq!(*s.as_result(), OFFSET);
    assert_eq!(s.best_fitness().fitness(), *s.as_result());
    s
}

#[test]
fn de() {
    let xs = &[
        -2.049101271522225e-8,
        5.081366300155457e-9,
        -1.4078821888482487e-8,
        9.517202998115888e-9,
    ];
    assert_xs!(test::<De>().best_parameters(), xs);
}

#[test]
fn pso() {
    let xs = &[
        -1.562453410836174e-8,
        -5.727866910332448e-9,
        1.7997333114342357e-8,
        -1.666517790554776e-8,
    ];
    assert_xs!(test::<Pso>().best_parameters(), xs);
}

#[test]
fn fa() {
    let xs = &[
        5.673461573820622e-10,
        8.303840948050095e-10,
        -3.937243668403862e-9,
        -9.977049224294723e-9,
    ];
    assert_xs!(test::<Fa>().best_parameters(), xs);
}

#[test]
fn rga() {
    let xs = &[
        -9.043074575290101e-9,
        -2.2331863304051235e-10,
        9.221720958016999e-9,
        -6.9001946979772355e-9,
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
