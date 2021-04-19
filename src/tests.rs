use crate::{
    ObjFunc, Solver, Task, {DESetting, DE}, {FASetting, FA}, {PSOSetting, PSO}, {RGASetting, RGA},
    {TLBOSetting, TLBO},
};
use ndarray::{Array1, ArrayView1, AsArray};

struct TestObj(Array1<f64>, Array1<f64>);

impl TestObj {
    fn new() -> Self {
        Self(Array1::zeros(4), Array1::ones(4) * 50.)
    }
}

impl ObjFunc for TestObj {
    type Result = f64;
    fn fitness<'a, A>(&self, _gen: u32, v: A) -> f64
    where
        A: AsArray<'a, f64>,
    {
        let v = v.into();
        v[0] * v[0] + 8. * v[1] * v[1] + v[2] * v[2] + v[3] * v[3]
    }
    fn result<'a, A>(&self, v: A) -> f64
    where
        A: AsArray<'a, f64>,
    {
        self.fitness(0, v)
    }
    fn ub(&self) -> ArrayView1<f64> {
        self.1.view()
    }
    fn lb(&self) -> ArrayView1<f64> {
        self.0.view()
    }
}

fn test<S>(mut a: S)
where
    S: Solver<TestObj>,
{
    let ans = a.run();
    let (x, y) = a.result();
    let history = a.history();
    assert!(history.len() > 0, "{}", history.len());
    assert!(ans.abs() < 1e-20, "{}", ans);
    for i in 0..4 {
        assert!(x[i].abs() < 1e-10, "x{} = {}", i, x[i]);
    }
    assert_eq!(y.abs(), ans);
}

#[test]
fn de() {
    test(DE::new(
        TestObj::new(),
        DESetting::default().task(Task::MinFit(1e-20)),
    ));
}

#[test]
fn pso() {
    test(PSO::new(
        TestObj::new(),
        PSOSetting::default().task(Task::MinFit(1e-20)),
    ));
}

#[test]
fn fa() {
    test(FA::new(
        TestObj::new(),
        FASetting::default().task(Task::MinFit(1e-20)),
    ));
}

#[test]
fn rga() {
    test(RGA::new(
        TestObj::new(),
        RGASetting::default().task(Task::MinFit(1e-20)),
    ));
}

#[test]
fn tlbo() {
    test(TLBO::new(
        TestObj::new(),
        TLBOSetting::default().task(Task::MinFit(1e-20)),
    ));
}
