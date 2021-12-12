use crate::{utility::prelude::*, *};

const OFFSET: f64 = 7.;

struct TestObj([f64; 4], [f64; 4]);

impl Default for TestObj {
    fn default() -> Self {
        Self([0.; 4], [50.; 4])
    }
}

impl ObjFunc for TestObj {
    type Result = f64;
    type Respond = f64;

    fn fitness(&self, v: &[f64], _: &Report) -> Self::Respond {
        OFFSET + v[0] * v[0] + 8. * v[1] * v[1] + v[2] * v[2] + v[3] * v[3]
    }

    fn result(&self, v: &[f64]) -> Self::Result {
        self.fitness(v, &Default::default())
    }

    fn ub(&self) -> &[f64] {
        &self.1
    }
    fn lb(&self) -> &[f64] {
        &self.0
    }
}

fn test<S>()
where
    S: Setting + Default,
    S::Algorithm: Algorithm<TestObj>,
{
    let s = Solver::build(S::default())
        .task(Task::MinFit(OFFSET))
        .solve(TestObj::default(), |_| true);
    let ans = s.result();
    let x = s.best_parameters();
    let y = s.best_fitness();
    let reports = s.reports();
    assert!(reports.len() > 0, "{}", reports.len());
    assert!((ans - OFFSET).abs() < 1e-20, "{}", ans);
    for i in 0..4 {
        assert!(x[i].abs() < 1e-6, "x{} = {}", i, x[i]);
    }
    assert_eq!(y.abs(), ans);
}

#[test]
fn de() {
    test::<De>();
}

#[test]
fn pso() {
    test::<Pso>();
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
