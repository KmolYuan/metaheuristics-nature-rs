#![doc(hidden)]
use crate::utility::prelude::*;

const OFFSET: f64 = 7.;

// An example case for doctest
pub struct TestObj;

impl TestObj {
    pub const fn new() -> Self {
        Self
    }
}

impl ObjFunc for TestObj {
    type Result = f64;
    type Fitness = f64;

    fn fitness(&self, x: &[f64], _: f64) -> Self::Fitness {
        OFFSET + x[0] * x[0] + 8. * x[1] * x[1] + x[2] * x[2] + x[3] * x[3]
    }

    fn result(&self, v: &[f64]) -> Self::Result {
        self.fitness(v, 0.)
    }

    fn bound(&self) -> &[[f64; 2]] {
        &[[-50., 50.]; 4]
    }
}

#[cfg(test)]
fn test<S>()
where
    S: Setting + Default,
    S::Algorithm: Algorithm<TestObj>,
{
    let s = Solver::build(S::default())
        .task(|ctx| ctx.best_f - OFFSET < 1e-20)
        .solve(TestObj)
        .unwrap();
    let ans = s.result();
    let xs = s.best_parameters();
    let y = s.best_fitness();
    let report = s.report();
    assert!(!report.is_empty(), "{}", report.len());
    assert!((ans - OFFSET).abs() < 1e-20, "{}", ans);
    for (i, x) in xs.iter().enumerate() {
        assert!(x.abs() < 1e-6, "x{} = {}", i, x);
    }
    assert_eq!(y.abs(), ans);
}

#[test]
fn de() {
    test::<De>();
}

#[test]
fn pso() {
    test::<Pso<_>>();
}

#[test]
fn fa() {
    test::<Fa>();
}

#[test]
fn rga() {
    test::<Rga<_>>();
}

#[test]
fn tlbo() {
    test::<Tlbo>();
}
