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
        -1.4096013263903092e-8,
        5.9294512524583e-9,
        -1.1682725196001081e-8,
        -1.686041215945611e-8,
    ];
    assert_eq!(test::<De>().best_parameters(), xs);
}

#[test]
fn pso() {
    let xs = &[
        -5.8952389010136305e-9,
        -6.556757596629172e-9,
        3.995683568281232e-9,
        -5.049790405009771e-9,
    ];
    assert_eq!(test::<Pso>().best_parameters(), xs);
}

#[test]
fn fa() {
    let xs = &[
        1.6004926526143065e-8,
        -3.990108320609924e-9,
        -1.025895099819533e-8,
        -1.331288383548763e-8,
    ];
    assert_eq!(test::<Fa>().best_parameters(), xs);
}

#[test]
fn rga() {
    let xs = &[
        -2.0994884944079517e-8,
        -3.802650390006022e-9,
        -3.352451856642831e-10,
        1.4403675944455944e-8,
    ];
    assert_eq!(test::<Rga>().best_parameters(), xs);
}

#[test]
fn tlbo() {
    let xs = &[
        -1.3786481627246014e-8,
        -1.830262228970864e-10,
        -1.6444290908967627e-8,
        -1.0078724638383442e-9,
    ];
    assert_eq!(test::<Tlbo>().best_parameters(), xs);
}
