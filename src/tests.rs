#![doc(hidden)]
use crate::prelude::*;

const OFFSET: f64 = 7.;

/// An example for doctest.
pub struct TestObj;

impl TestObj {
    /// A dummy constructor.
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
    type Ys = Product<f64, f64>;

    fn fitness(&self, xs: &[f64]) -> Self::Ys {
        let y = OFFSET + xs[0] * xs[0] + 8. * xs[1] * xs[1] + xs[2] * xs[2] + xs[3] * xs[3];
        Product::new(y, y)
    }
}

/// A multi-objective example for doctest.
pub struct TestMO;

impl TestMO {
    /// A dummy constructor.
    pub const fn new() -> Self {
        Self
    }
}

impl Bounded for TestMO {
    fn bound(&self) -> &[[f64; 2]] {
        &[[-50., 50.]; 2]
    }
}

#[derive(Clone)]
pub struct TestMOFit {
    cost: f64,
    weight: f64,
}

impl Fitness for TestMOFit {
    type Best<T: Fitness> = Pareto<T>;
    type Eval = f64;

    fn is_dominated(&self, rhs: &Self) -> bool {
        self.cost <= rhs.cost && self.weight <= rhs.weight
    }

    fn eval(&self) -> Self::Eval {
        self.cost.max(self.weight)
    }
}

impl ObjFunc for TestMO {
    type Ys = Product<TestMOFit, ()>;

    fn fitness(&self, xs: &[f64]) -> Self::Ys {
        let ys = TestMOFit { cost: xs[0] * xs[0], weight: xs[1] * xs[1] };
        Product::new(ys, ())
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
        .task(|ctx| ctx.best.as_result_fit().eval() - OFFSET < 1e-20)
        .callback(|ctx| report.push(ctx.best.get_eval()))
        .solve();
    assert!(!report.is_empty());
    assert_eq!(s.get_best_eval(), OFFSET);
    s
}

#[cfg(test)]
macro_rules! assert_xs {
    ($case:expr) => {
        for x in $case.as_best_xs() {
            assert!(x.abs() < 2.1e-8, "x: {x}");
        }
    };
}

#[test]
fn de() {
    assert_xs!(test::<De>());
}

#[test]
fn pso() {
    assert_xs!(test::<Pso>());
}

#[test]
fn fa() {
    assert_xs!(test::<Fa>());
}

#[test]
fn rga() {
    assert_xs!(test::<Rga>());
}

#[test]
fn tlbo() {
    assert_xs!(test::<Tlbo>());
}

#[cfg(feature = "rayon")]
#[test]
fn test_rng() {
    let mut rng1 = Rng::new(SeedOpt::U64(0));
    rng1.stream(30);
    let mut rng2 = rng1.clone();
    let non_parallel = rng1
        .stream(100)
        .into_iter()
        .filter_map(|mut rng| {
            if rng.maybe(0.8) && rng.maybe(0.5) && rng.maybe(0.3) {
                Some(rng.rand())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let parallel = rng2
        .stream(100)
        .into_par_iter()
        .filter_map(|mut rng| {
            if rng.maybe(0.8) && rng.maybe(0.5) && rng.maybe(0.3) {
                Some(rng.rand())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(non_parallel, parallel);
}
